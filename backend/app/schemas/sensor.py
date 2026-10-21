from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class SensorMeasurementRequest(BaseModel):
    timestamp: Optional[datetime] = None
    heart_rate: Optional[float] = Field(None, ge=30.0, le=220.0, description="Heart rate in bpm")
    spo2: Optional[float] = Field(None, ge=70.0, le=100.0, description="SpO2 percentage")
    temperature: Optional[float] = Field(None, ge=30.0, le=45.0, description="Body temperature in Celsius")
    respiratory_rate: Optional[float] = Field(None, ge=5.0, le=60.0, description="Respiratory rate in bpm")
    gsr: Optional[float] = Field(None, ge=0.0, description="Galvanic Skin Response")
    is_simulated: bool = Field(False, description="Explicit flag if payload is a software test fixture")

class SensorMeasurementResponse(BaseModel):
    session_id: str
    status: str
    sensor_quality_score: float
    is_simulated: bool
