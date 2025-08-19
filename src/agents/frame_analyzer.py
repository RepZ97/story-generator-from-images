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
from config.logging_config import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

def _parse_fallback_response(response_content: str) -> Dict[str, Any]:
    """Fallback parser for malformed JSON responses."""
    logger.warning("Using fallback response parser due to malformed JSON")
    return {"scene_description": "Scene analysis completed", "entities": []}

def analyze_frames(state: GraphState):
    logger.info("Starting Frame Analysis...")
    frame_metadata_list = []

    openai_model = os.getenv("OPENAI_MODEL_FRAME")
    if not openai_model:
        logger.error("OPENAI_MODEL_FRAME environment variable is required")
        raise ValueError("OPENAI_MODEL_FRAME environment variable is required")

    temperature = float(os.getenv("TEMPERATURE_FRAME", "0.1"))

    llm = ChatOpenAI(
        model=openai_model, api_key=openai_api_key, temperature=temperature
    )

    for i, image_path in enumerate(state["image_paths"]):
        frame_id = f"frame_{i+1:03d}.jpg"

        logger.info(f"Analyzing frame {i+1}/{len(state['image_paths'])}: {frame_id}")

        # Analyze the frame
        try:
            # Load and encode image to Base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "You are an expert visual scene analyzer. "
                            "Your task is to examine this single frame and return structured information "
                            "following the provided schema. "
                            "Output MUST be compatible with the Perspectives model.\n\n"
                            "Requirements:\n"
                            "1. Provide a concise but detailed `scene_description` of the entire image.\n"
                            "2. List all visible `entities`. For each entity:\n"
                            "   - `name`: Use a short, consistent identifier (e.g., 'man_1', 'dog_1').\n"
                            "   - `type`: Broad category (person, animal, object, location, etc.).\n"
                            "   - `attributes`: Dictionary with rich details (color, clothing, position in frame, action, size, emotion, etc.).\n"
                            "3. Only include what is clearly visible. Do not speculate.\n"
                            "4. Distinguish between multiple similar entities (e.g., two people, cars).\n"
                            "5. This output will later be linked across frames to build a narrative, so consistency matters."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ]
            )

            structured_llm = llm.with_structured_output(
                Perspectives, method="function_calling"
            )

            result: Perspectives = structured_llm.invoke([message])

            # Parse the response
            try:
                analysis = result.model_dump()
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning(f"JSON decode error for frame {frame_id}, using fallback parser")
                analysis = _parse_fallback_response(result.content)

            # Convert entities to dictionaries for consistency
            entities_list = analysis.get("entities", [])
            entities_dicts = []
            for entity in entities_list:
                if hasattr(entity, "model_dump"):
                    entities_dicts.append(entity.model_dump())
                else:
                    entities_dicts.append(entity)

            frame_metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=get_file_timestamp(image_path),
                scene_description=analysis.get(
                    "scene_description", "Scene analysis unavailable"
                ),
                entities=entities_dicts,
            )

            frame_metadata_list.append(frame_metadata)
            logger.info(f"Successfully analyzed frame {frame_id} with {len(entities_dicts)} entities")

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_id}: {str(e)}")

            frame_metadata = FrameMetadata(
                frame_id=frame_id,
                timestamp=get_file_timestamp(image_path),
                scene_description="Error analyzing frame",
                entities=[],
            )
            frame_metadata_list.append(frame_metadata)

    logger.info(f"Frame analysis completed. Processed {len(frame_metadata_list)} frames")
    return {"frame_metadata": frame_metadata_list}
