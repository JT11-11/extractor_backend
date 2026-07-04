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
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    client = ollama.Client(host=OLLAMA_HOST, headers=headers)

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
    parsed = _parse_model_json(raw_text)
    return _normalize_extraction(parsed)


# ---------------------------------------------------------------------------
# JSON parsing (mirrors parseModelJson in statement.ts)
# ---------------------------------------------------------------------------


def _parse_model_json(content: str) -> Any:
    cleaned = content.strip()
    # strip markdown fences
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

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
        return json.loads(cleaned[first : last + 1])

    raise ValueError("The model did not return parseable JSON.")


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

    fields = data.get("fields") or _fields_from_rows(transactions)

    return {
        "institution": _str_or_none(data.get("institution")),
        "accountName": _str_or_none(data.get("accountName")),
        "accountNumber": _str_or_none(data.get("accountNumber")),
        "cardName": _first_string(data, ["cardName", "card", "cardType", "cardProduct", "cardDescription"])
            or _first_string_from_rows(transactions, ["cardName", "card", "cardType", "cardProduct", "cardDescription"]),
        "nameOnCard": _first_string(data, ["nameOnCard", "cardholderName", "cardHolderName", "name"])
            or _first_string_from_rows(transactions, ["nameOnCard", "cardholderName", "cardHolderName", "name"]),
        "statementPeriod": _str_or_none(data.get("statementPeriod")),
        "currency": _str_or_none(data.get("currency")),
        "fields": fields,
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
