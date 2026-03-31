from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TacticalSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    temp_f: Optional[float] = None
    aircraft_count: int = 0
    mapped_aircraft_count: int = 0
    lan_device_count: int = 0
    ssh_failure_count: int = 0
    internet_latency_ms: Optional[float] = None
    rf_peaks: List[Dict[str, Any]] = []  # [{'freq': 155.0, 'power': 22.0}]
    rivers: List[Dict[str, Any]] = []  # [{'name': '...', 'value': 7.0, 'unit': 'ft'}]
    software: Dict[str, int] = {}  # {'apt': 3000, ...}
    raw_sitrep: str = ""


class ExtractedEntity(BaseModel):
    name: str = Field(description="The unique name of the entity.")
    type: str = Field(
        description="The type of the entity (COMPANY, TECH, PERSON, MARKET)."
    )
    description: str = Field(
        description="A brief description of the entity based on the context."
    )


class ExtractedConnection(BaseModel):
    source_entity_name: str = Field(description="The name of the source entity.")
    target_entity_name: str = Field(description="The name of the target entity.")
    type: str = Field(description="The type of relationship.")
    description: str = Field(description="Context describing the relationship.")


class ExtractedTrend(BaseModel):
    name: str = Field(description="The name of the emerging trend.")
    description: str = Field(description="Description of the trend and its impact.")
    velocity: str = Field(
        description="The velocity or maturity of the trend (emerging, accelerating, stabilizing)."
    )


class KnowledgeGraphExtraction(BaseModel):
    entities: List[ExtractedEntity] = Field(
        description="List of entities identified in the text."
    )
    connections: List[ExtractedConnection] = Field(
        description="List of relationships identified between entities."
    )
    trends: List[ExtractedTrend] = Field(
        description="List of emerging market trends identified in the text."
    )
