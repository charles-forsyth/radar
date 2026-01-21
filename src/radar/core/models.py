from pydantic import BaseModel, Field
from typing import List
from radar.db.models import EntityType, ConnectionType


class ExtractedEntity(BaseModel):
    name: str = Field(description="The unique name of the entity.")
    type: EntityType = Field(
        description="The type of the entity (COMPANY, TECH, PERSON, MARKET)."
    )
    description: str = Field(
        description="A brief description of the entity based on the context."
    )


class ExtractedConnection(BaseModel):
    source_entity_name: str = Field(description="The name of the source entity.")
    target_entity_name: str = Field(description="The name of the target entity.")
    type: ConnectionType = Field(description="The type of relationship.")
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
