import os
import json
from typing import Dict, Any
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from models.states import GraphState
from models.data_models import FrameMetadata, Perspectives
from config.environment import openai_api_key
from utils import get_file_timestamp
from dotenv import load_dotenv

load_dotenv()

def _parse_fallback_response(self, response_content: str) -> Dict[str, Any]:
    """Fallback parser for malformed JSON responses."""

    return {
        "scene_description": "Scene analysis completed",
        "entities": []
    }


def analyze_frames(state: GraphState):

    print("Starting Frame Analysis...")
    frame_metadata_list = []

    openai_model= os.getenv("OPENAI_MODEL")
    if not openai_model:
        raise ValueError("OPENAI_MODEL environment variable is required")

    llm = ChatOpenAI(
            model=openai_model,
            api_key=openai_api_key,
            temperature=0.1
    )

    for i, image_path in enumerate(state["image_paths"]):

        frame_id = f"frame_{i+1:03d}.jpg"
        
        print(f"Analyzing frame {i+1}/{len(state['image_paths'])}: {frame_id}")
        
        # Analyze the frame
        try:
            # Load and encode image to Base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            
            message = HumanMessage(content=[
                {"type": "text", "text": "Please analyze this image and provide detailed information about the scene and entities."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ])
            
            structured_llm = llm.with_structured_output(Perspectives, method="function_calling")

            result: Perspectives = structured_llm.invoke([message])
            
            # Parse the response
            try:
                analysis = result.model_dump()
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                analysis = _parse_fallback_response(result.content)
            
            # Convert entities to dictionaries for consistency
            entities_list = analysis.get("entities", [])
            entities_dicts = []
            for entity in entities_list:
                if hasattr(entity, 'model_dump'):
                    entities_dicts.append(entity.model_dump())
                else:
                    entities_dicts.append(entity)
            
            frame_metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=get_file_timestamp(image_path),
                scene_description=analysis.get("scene_description", "Scene analysis unavailable"),
                entities=entities_dicts
            )
            
            frame_metadata_list.append(frame_metadata)
            
        except Exception as e:
            print(f"Error analyzing frame {frame_id}: {str(e)}")

            return FrameMetadata(
                frame_id=frame_id,
                timestamp=get_file_timestamp(image_path),
                scene_description="Error analyzing frame",
                entities=[]
            )
        
    return {"frame_metadata": frame_metadata_list}