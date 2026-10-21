from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class CreateSessionRequest(BaseModel):
    external_reference: str = Field(..., description="External person reference code (e.g. U123)", example="U123")

class SessionResponse(BaseModel):
    session_id: str
    person_id: str
    external_reference: str
    session_timestamp: datetime
    status: str
    duration: float

    class Config:
        from_attributes = True
