"""Pydantic models mirroring src/lib/statement.ts."""

from __future__ import annotations

from typing import Any, Optional, Union
from pydantic import BaseModel, Field


CellValue = Union[str, int, float, bool, None]


class StatementField(BaseModel):
    key: str
    label: str
    kind: Optional[str] = None


class Transaction(BaseModel):
    values: dict[str, CellValue]
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    raw_text: Optional[str] = Field(default=None, alias="rawText")

    model_config = {"populate_by_name": True}


class StatementExtraction(BaseModel):
    institution: Optional[str] = None
    account_name: Optional[str] = Field(default=None, alias="accountName")
    account_number: Optional[str] = Field(default=None, alias="accountNumber")
    card_name: Optional[str] = Field(default=None, alias="cardName")
    name_on_card: Optional[str] = Field(default=None, alias="nameOnCard")
    statement_period: Optional[str] = Field(default=None, alias="statementPeriod")
    currency: Optional[str] = None
    fields: list[StatementField] = []
    transactions: list[Transaction] = []
    notes: list[str] = []

    model_config = {"populate_by_name": True}


class JobCreate(BaseModel):
    pass  # multipart handled directly in the route


class JobStatus(BaseModel):
    id: str
    status: str
    file_name: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    job_id: str
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
