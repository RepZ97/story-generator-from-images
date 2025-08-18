from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    type: str
    attributes: Optional[Dict[str, str]] = {}


class FrameMetadata(TypedDict):
    frame_id: str
    timestamp: str
    scene_description: str
    entities: List[Dict[str, Any]]


class Perspectives(BaseModel):
    scene_description: str = Field(
        description="A detailed description of the scene in the image."
    )
    entities: List[Entity] = Field(
        description="A list of entities detected in the scene, each with a name, type and a dictionary of attributes."
    )


class ConsistentEntity(BaseModel):
    entity_id: str
    description: str
    first_seen: str
    entity_type: str
    appearances: List[str] = []


class Event(BaseModel):
    frame_id: str
    timestamp: str
    event: str
