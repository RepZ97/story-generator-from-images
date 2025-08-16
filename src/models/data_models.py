from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    type: str
    attributes: Optional[Dict[str, str]] = {}

class FrameMetadata(TypedDict):
    frame_id: str
    timestamp: str
    scene_description: str
    entities: List[Entity]

class Perspectives(BaseModel):
    scene_description: str
    entities: List[Entity]


