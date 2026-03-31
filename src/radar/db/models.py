import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON


class Signal(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str = Field(index=True)
    content: str
    source: str
    url: Optional[str] = None
    date: datetime = Field(default_factory=datetime.now, index=True)


class TacticalAlert(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    domain: str  # AIR, RF, WEATHER, CYBER, GRID
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    data_context: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now, index=True)


class Telemetry(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    temp_f: Optional[float] = None
    aircraft_count: int = 0
    lan_device_count: int = 0
    ssh_failure_count: int = 0
    internet_latency_ms: Optional[float] = None


class RiverLevel(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    station_name: str = Field(index=True)
    value: float
    unit: str  # "ft" or "cfs"


class RFPeak(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    frequency_mhz: float
    power_db: float


class SoftwareInventory(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    manager: str  # "apt", "pip", "uv", "micromamba"
    package_count: int


class Statistic(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    category: str = Field(index=True)  # e.g., "FINANCE", "LOGISTICS", "INFRASTRUCTURE"
    label: str = Field(
        index=True
    )  # e.g., "Gas Price", "Wildfire Acres", "GPU Benchmarks"
    value: float
    unit: Optional[str] = None  # e.g., "USD", "Acres", "t/s"
    description: Optional[str] = None  # Brief context about the stat
    source_signal_id: Optional[uuid.UUID] = Field(default=None)


class ChatSession(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str
    created_at: datetime = Field(default_factory=datetime.now)


class ChatMessage(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="chatsession.id")
    role: str  # user, assistant
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
