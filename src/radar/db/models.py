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
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(768)))


class Trend(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    description: str
    velocity: str = "emerging"
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(768)))


class Entity(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    type: EntityType
    details: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    vector: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(768)))


class Connection(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    source_uuid: uuid.UUID = Field(index=True)
    target_uuid: uuid.UUID = Field(index=True)
    type: ConnectionType
    meta_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
