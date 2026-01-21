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


class KnowledgeGraphExtraction(BaseModel):
    entities: List[ExtractedEntity] = Field(
        description="List of entities identified in the text."
    )
    connections: List[ExtractedConnection] = Field(
        description="List of relationships identified between entities."
    )
