"""Ollama vision extraction + JSON normalisation.

This is a direct Python port of src/lib/statement.ts — all the same
field-inference, amount-cleaning, and card-name fallback logic is
preserved so the output schema is identical.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import ollama

from worker.pdf_render import RenderedPage

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5-vl")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "240"))
PAGES_PER_REQUEST = max(1, int(os.environ.get("EXTRACTION_PAGES_PER_REQUEST", "4")))

_SYSTEM_PROMPT = (
    "You extract bank statement transactions from page images. "
    "Infer the fields from the statement itself. Do not force a fixed schema. "
    "Return every transaction row you can find. Preserve original field names when possible. "
    "When a credit card statement shows card identity fields, always extract Card Name as cardName "
    "and Name on Card as nameOnCard. If the statement has one card section, put them at the top level. "
    "If transactions are split by cardholder/card, also keep row-level cardName/nameOnCard fields. "
    "For amount fields, return only the amount value. Remove CR, DR, debit, and credit labels from the amount cell. "
    "Return only valid JSON. Do not wrap it in markdown fences. "
    'Preferred shape: {"cardName":"UOB PRVI MILES VISA CARD","nameOnCard":"LOH JUN WEI",'
    '"fields":[{"key":"postDate","label":"Post Date"},{"key":"amount","label":"Amount","kind":"amount"}],'
    '"transactions":[{"values":{"postDate":"02 MAY","amount":"123.45"}}],"notes":[]}. '
    "If you return a plain array of row objects, each object must be one transaction. "
    "If a value is missing, use null. Do not include non-transaction summary rows as transactions."
)

_AMOUNT_KEYS_RE = re.compile(
    r"amount|amt|debit|credit|withdrawal|deposit|payment|charge|balance",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_statement_from_pages(
    file_name: str,
    pages: list[RenderedPage],
) -> dict[str, Any]:
    """Call the Ollama vision model and return a normalised extraction dict."""
    if not pages:
        raise ValueError("No rendered pages were provided for extraction.")

    extractions: list[dict[str, Any]] = []
    for start in range(0, len(pages), PAGES_PER_REQUEST):
        batch = pages[start : start + PAGES_PER_REQUEST]
        print(
            "[Extraction] Calling vision model for "
            f"pages {batch[0].page_number}-{batch[-1].page_number} "
            f"of {len(pages)} using {OLLAMA_MODEL}",
            flush=True,
        )
        extractions.append(_extract_statement_batch(file_name, batch))

    return _merge_extractions(extractions)


def _extract_statement_batch(
    file_name: str,
    pages: list[RenderedPage],
) -> dict[str, Any]:
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    client = ollama.Client(
        host=OLLAMA_HOST,
        headers=headers,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )

    # The ollama client expects `content` as a plain string and images as a
    # separate `images` list of base64 strings — not OpenAI-style content parts.
    page_labels = ", ".join(str(page.page_number) for page in pages)
    text = (
        f"Extract all transactions from {file_name}. "
        f"The following {len(pages)} page image(s) are provided in order: {page_labels}."
    )
    images = [page.image_base64 for page in pages]

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text, "images": images},
        ],
        options={"temperature": 0},
    )

    raw_text: str = response["message"]["content"]
    try:
        parsed = _parse_model_json(raw_text)
    except Exception as exc:
        print(
            "[Extraction] Model returned invalid JSON; attempting repair "
            f"for pages {pages[0].page_number}-{pages[-1].page_number}: {exc}",
            flush=True,
        )
        parsed = _repair_model_json(client, raw_text)
    return _normalize_extraction(parsed)


def _repair_model_json(client: ollama.Client, raw_text: str) -> Any:
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. Return only valid JSON. "
                    "Do not add markdown fences, comments, or explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair this malformed bank statement extraction JSON so it "
                    "parses correctly. Preserve all fields and transactions.\n\n"
                    + raw_text
                ),
            },
        ],
        options={"temperature": 0},
    )
    repaired_text: str = response["message"]["content"]
    return _parse_model_json(repaired_text)


def _merge_extractions(extractions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(extractions) == 1:
        return extractions[0]

    fields_by_key: dict[str, dict] = {}
    transactions: list[dict] = []
    notes: list[str] = []
    merged: dict[str, Any] = {
        "institution": None,
        "accountName": None,
        "accountNumber": None,
        "cardName": None,
        "nameOnCard": None,
        "statementPeriod": None,
        "currency": None,
    }

    for extraction in extractions:
        for key in merged:
            if merged[key] is None and extraction.get(key) is not None:
                merged[key] = extraction.get(key)

        for field in extraction.get("fields", []):
            if isinstance(field, dict):
                _add_field(fields_by_key, field)

        txns = extraction.get("transactions", [])
        if isinstance(txns, list):
            transactions.extend(txn for txn in txns if isinstance(txn, dict))

        batch_notes = extraction.get("notes", [])
        if isinstance(batch_notes, list):
            notes.extend(note for note in batch_notes if isinstance(note, str))

    transactions = _fill_row_card_metadata(
        transactions,
        _str_or_none(merged.get("cardName")),
        _str_or_none(merged.get("nameOnCard")),
    )

    for field in _fields_from_rows(transactions):
        _add_field(fields_by_key, field)

    return {
        **merged,
        "fields": list(fields_by_key.values()),
        "transactions": transactions,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# JSON parsing (mirrors parseModelJson in statement.ts)
# ---------------------------------------------------------------------------


def _parse_model_json(content: str) -> Any:
    cleaned = content.strip()
    # strip markdown fences
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE).strip()

    parsed = _loads_json_with_minor_repairs(cleaned)
    if parsed is not None:
        return parsed

    # Try to extract the first JSON object or array
    obj_start = cleaned.find("{")
    arr_start = cleaned.find("[")

    if obj_start == -1 and arr_start == -1:
        raise ValueError("The model did not return parseable JSON.")

    if obj_start == -1:
        first = arr_start
    elif arr_start == -1:
        first = obj_start
    else:
        first = min(obj_start, arr_start)

    last = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if first >= 0 and last > first:
        parsed = _loads_json_with_minor_repairs(cleaned[first : last + 1])
        if parsed is not None:
            return parsed

    raise ValueError("The model did not return parseable JSON.")


def _loads_json_with_minor_repairs(value: str) -> Any | None:
    candidate = _remove_trailing_commas(value)
    for _ in range(12):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            repaired = _repair_missing_comma(candidate, exc)
            if repaired == candidate:
                return None
            candidate = _remove_trailing_commas(repaired)
    return None


def _remove_trailing_commas(value: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", value)


def _repair_missing_comma(value: str, exc: json.JSONDecodeError) -> str:
    if "Expecting ',' delimiter" not in exc.msg:
        return value

    pos = exc.pos
    next_pos = pos
    while next_pos < len(value) and value[next_pos].isspace():
        next_pos += 1

    prev_pos = pos - 1
    while prev_pos >= 0 and value[prev_pos].isspace():
        prev_pos -= 1

    if (
        0 <= prev_pos < next_pos < len(value)
        and value[next_pos] == '"'
        and value[prev_pos] in '"}]0123456789'
    ):
        return value[:next_pos] + "," + value[next_pos:]

    return value


# ---------------------------------------------------------------------------
# Normalisation (mirrors normalizeExtraction + helpers in statement.ts)
# ---------------------------------------------------------------------------


def _normalize_extraction(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        if any(_is_transaction_group(item) for item in data):
            return _normalize_grouped(data)
        return _normalize_flat_array(data)
    return _normalize_object(data)


def _normalize_flat_array(rows: list[Any]) -> dict[str, Any]:
    transactions = [{"values": _row_to_values(r)} for r in rows]
    return {
        "cardName": _first_string_from_rows(transactions, ["cardName", "card", "cardType", "cardProduct", "cardDescription"]),
        "nameOnCard": _first_string_from_rows(transactions, ["nameOnCard", "cardholderName", "cardHolderName", "name"]),
        "fields": _fields_from_rows(transactions),
        "transactions": transactions,
        "notes": [],
    }


def _normalize_grouped(groups: list[Any]) -> dict[str, Any]:
    fields_by_key: dict[str, dict] = {}
    transactions: list[dict] = []
    notes: list[str] = []
    first_card_name: str | None = None
    first_name_on_card: str | None = None

    for group in groups:
        if not isinstance(group, dict):
            continue
        card_name = _first_string(group, ["cardName", "card", "cardType", "cardProduct", "cardDescription"])
        name_on_card = _first_string(group, ["nameOnCard", "cardholderName", "cardHolderName", "name"])
        if first_card_name is None:
            first_card_name = card_name
        if first_name_on_card is None:
            first_name_on_card = name_on_card

        _add_field(fields_by_key, {"key": "cardName", "label": "Card Name"})
        _add_field(fields_by_key, {"key": "nameOnCard", "label": "Name on Card"})

        for f in group.get("fields", []):
            if isinstance(f, dict):
                _add_field(fields_by_key, f)

        notes.extend(n for n in group.get("notes", []) if isinstance(n, str))

        for tx in group.get("transactions", []):
            if isinstance(tx, dict) and "values" in tx:
                values = _row_to_values(tx["values"])
            else:
                values = _row_to_values(tx)
            transactions.append({"values": {"cardName": card_name, "nameOnCard": name_on_card, **values}})

    if transactions:
        for f in _fields_from_rows(transactions):
            _add_field(fields_by_key, f)

    return {
        "cardName": first_card_name,
        "nameOnCard": first_name_on_card,
        "fields": list(fields_by_key.values()),
        "transactions": transactions,
        "notes": notes,
    }


def _normalize_object(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("The model returned JSON, but not an extraction object.")

    raw_txns = data.get("transactions") or data.get("data") or []
    transactions = []
    for row in raw_txns:
        if isinstance(row, dict) and "values" in row and isinstance(row["values"], dict):
            transactions.append({**row, "values": _row_to_values(row["values"])})
        else:
            transactions.append({"values": _row_to_values(row)})

    card_name = _first_string(data, ["cardName", "card", "cardType", "cardProduct", "cardDescription"]) or _first_string_from_rows(
        transactions,
        ["cardName", "card", "cardType", "cardProduct", "cardDescription"],
    )
    name_on_card = _first_string(data, ["nameOnCard", "cardholderName", "cardHolderName", "name"]) or _first_string_from_rows(
        transactions,
        ["nameOnCard", "cardholderName", "cardHolderName", "name"],
    )
    transactions = _fill_row_card_metadata(transactions, card_name, name_on_card)

    fields_by_key: dict[str, dict] = {}
    for field in data.get("fields") or []:
        if isinstance(field, dict):
            _add_field(fields_by_key, field)
    for field in _fields_from_rows(transactions):
        _add_field(fields_by_key, field)

    return {
        "institution": _str_or_none(data.get("institution")),
        "accountName": _str_or_none(data.get("accountName")),
        "accountNumber": _str_or_none(data.get("accountNumber")),
        "cardName": card_name,
        "nameOnCard": name_on_card,
        "statementPeriod": _str_or_none(data.get("statementPeriod")),
        "currency": _str_or_none(data.get("currency")),
        "fields": list(fields_by_key.values()),
        "transactions": transactions,
        "notes": data.get("notes") if isinstance(data.get("notes"), list) else [],
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _is_transaction_group(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("transactions"), list)


def _row_to_values(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {k: _to_cell_value(k, v) for k, v in row.items()}


def _to_cell_value(key: str, value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _clean_amount(value) if _AMOUNT_KEYS_RE.search(key) else value
    return json.dumps(value)


def _clean_amount(value: str) -> str:
    return re.sub(r"\s*(?:CR|DR|CREDIT|DEBIT)\.?$", "", value, flags=re.IGNORECASE).strip()


def _fields_from_rows(rows: list[dict]) -> list[dict]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        values = row.get("values", row)
        if isinstance(values, dict):
            for k in values:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    return [{"key": k, "label": _label_from_key(k)} for k in keys]


def _add_field(fields_by_key: dict, field: dict) -> None:
    key = field.get("key")
    if not isinstance(key, str) or not key.strip():
        return
    label = field.get("label")
    label = label if isinstance(label, str) and label.strip() else _label_from_key(key)
    entry: dict = {"key": key, "label": label}
    if field.get("kind"):
        entry["kind"] = field["kind"]
    fields_by_key[key] = entry


def _fill_row_card_metadata(
    transactions: list[dict],
    card_name: str | None,
    name_on_card: str | None,
) -> list[dict]:
    if not card_name and not name_on_card:
        return transactions

    filled: list[dict] = []
    for transaction in transactions:
        values = transaction.get("values")
        if not isinstance(values, dict):
            filled.append(transaction)
            continue

        next_values = dict(values)
        if card_name and not _str_or_none(next_values.get("cardName")):
            next_values["cardName"] = card_name
        if name_on_card and not _str_or_none(next_values.get("nameOnCard")):
            next_values["nameOnCard"] = name_on_card
        filled.append({**transaction, "values": next_values})

    return filled


def _label_from_key(key: str) -> str:
    label = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    label = re.sub(r"[_-]+", " ", label)
    return label.title()


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _first_string(record: dict, keys: list[str]) -> str | None:
    for k in keys:
        v = _str_or_none(record.get(k))
        if v:
            return v
    return None


def _first_string_from_rows(rows: list[dict], keys: list[str]) -> str | None:
    for row in rows:
        v = _first_string(row.get("values", {}), keys)
        if v:
            return v
    return None
