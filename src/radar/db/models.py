import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, JSON


class ConnectionType(str, Enum):
    SUPPORTS = "SUPPORTS"
    DRIVES = "DRIVES"
    COMPETES_WITH = "COMPETES"
    PART_OF = "PART_OF"
    MENTIONS = "MENTIONS"


class EntityType(str, Enum):
    COMPANY = "COMPANY"
    TECH = "TECH"
    PERSON = "PERSON"
    MARKET = "MARKET"


class Signal(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    date: datetime = Field(default_factory=datetime.now)
    title: str
    url: Optional[str] = None
    content: str
    raw_text: Optional[str] = None
    source: str = "web"
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(3072)))


class Trend(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str
    velocity: str = "emerging"
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(3072)))
    first_seen: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)


class Entity(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    type: EntityType
    details: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(3072)))
    first_seen: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)


class Connection(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    source_uuid: uuid.UUID = Field(index=True)
    target_uuid: uuid.UUID = Field(index=True)
    type: ConnectionType
    meta_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    first_seen: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)


class ChatSession(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    title: Optional[str] = None


class ChatMessage(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(index=True)
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.now)


class Watchpoint(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    topic: str = Field(unique=True, index=True)
    description: str
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True)


class Alert(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    watchpoint_id: uuid.UUID = Field(index=True)
    signal_id: uuid.UUID = Field(index=True)
    reason: str
    created_at: datetime = Field(default_factory=datetime.now)


class TacticalAlert(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    domain: str  # AIR, RF, WEATHER, CYBER, GRID
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    data_context: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)


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
    label: str = Field(index=True)     # e.g., "Gas Price", "Wildfire Acres", "GPU Benchmarks"
    value: float
    unit: Optional[str] = None         # e.g., "USD", "Acres", "t/s"
    source_signal_id: Optional[uuid.UUID] = Field(default=None, foreign_key="signal.id")
