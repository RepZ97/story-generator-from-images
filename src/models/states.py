from typing import TypedDict, List, Dict, Any, Optional
from models.data_models import FrameMetadata


class GraphState(TypedDict):
    image_paths: List[str]
    frame_metadata: List[FrameMetadata]
    consistent_entities: Dict[str, Any]
    final_story: str