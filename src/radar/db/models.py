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


class Entity(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    type: EntityType
    details: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(3072)))


class Connection(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    source_uuid: uuid.UUID = Field(index=True)
    target_uuid: uuid.UUID = Field(index=True)
    type: ConnectionType
    meta_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))


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
