import json
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from models.states import GraphState
from models.data_models import FrameMetadata, Entity, ConsistentEntity, Event
from config.environment import openai_api_key
import os
from dotenv import load_dotenv
from utils import clean_json_text

load_dotenv()


def _calculate_similarity(entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two entities based on name, type, and attributes.
    Returns a score between 0 and 1.
    """
    score = 0.0

    # Name similarity (exact match gets high score, partial match gets medium score)
    if entity1["name"].lower() == entity2["name"].lower():
        score += 0.4
    elif (
        entity1["name"].lower() in entity2["name"].lower()
        or entity2["name"].lower() in entity1["name"].lower()
    ):
        score += 0.2

    # Type similarity
    if entity1["type"].lower() == entity2["type"].lower():
        score += 0.3
    elif (
        entity1["type"].lower() in entity2["type"].lower()
        or entity2["type"].lower() in entity1["type"].lower()
    ):
        score += 0.15

    # Attribute similarity
    if entity1.get("attributes") and entity2.get("attributes"):
        common_attrs = set(entity1["attributes"].keys()) & set(
            entity2["attributes"].keys()
        )
        if common_attrs:
            attr_score = 0
            for attr in common_attrs:
                if (
                    entity1["attributes"][attr].lower()
                    == entity2["attributes"][attr].lower()
                ):
                    attr_score += 0.1
            score += min(attr_score, 0.3)

    return score


def _resolve_entities(
    frame_metadata_list: List[FrameMetadata],
) -> Dict[str, ConsistentEntity]:
    """
    Resolve entities across frames and assign consistent IDs.
    """
    consistent_entities = {}
    entity_counter = {}

    for frame_meta in frame_metadata_list:
        for entity in frame_meta["entities"]:
            # Try to find a matching existing entity
            best_match = None
            best_score = 0.7  # Threshold for considering entities the same

            # First, try to find exact name match (case-insensitive)
            for entity_id, consistent_entity in consistent_entities.items():
                if entity["name"].lower() == consistent_entity.description.lower():
                    best_match = entity_id
                    break

            # If no exact match found, try similarity matching
            if not best_match:
                for entity_id, consistent_entity in consistent_entities.items():
                    # Check similarity with last appearance
                    if consistent_entity.appearances:
                        last_frame_id = consistent_entity.appearances[-1]
                        # Find the last frame metadata
                        last_frame = next(
                            (
                                f
                                for f in frame_metadata_list
                                if f["frame_id"] == last_frame_id
                            ),
                            None,
                        )
                        if last_frame:
                            last_entity = next(
                                (
                                    e
                                    for e in last_frame["entities"]
                                    if e["name"].lower() == entity["name"].lower()
                                ),
                                None,
                            )
                            if last_entity:
                                similarity = _calculate_similarity(entity, last_entity)
                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = entity_id

            if best_match:
                # Update existing entity
                if (
                    frame_meta["frame_id"]
                    not in consistent_entities[best_match].appearances
                ):
                    consistent_entities[best_match].appearances.append(
                        frame_meta["frame_id"]
                    )
            else:
                # Create new entity
                entity_type = entity["type"].lower()
                if entity_type not in entity_counter:
                    entity_counter[entity_type] = 0
                entity_counter[entity_type] += 1

                entity_id = f"{entity_type}_{entity_counter[entity_type]}"

                consistent_entities[entity_id] = ConsistentEntity(
                    entity_id=entity_id,
                    description=entity["name"],
                    first_seen=frame_meta["frame_id"],
                    entity_type=entity["type"],
                )
                consistent_entities[entity_id].appearances.append(
                    frame_meta["frame_id"]
                )

    return consistent_entities


def _extract_events(
    frame_metadata_list: List[FrameMetadata],
    consistent_entities: Dict[str, ConsistentEntity],
) -> List[Event]:
    """
    Extract key events from the frame sequence.
    """
    events = []

    for i, frame_meta in enumerate(frame_metadata_list):
        # Create a summary of what's happening in this frame
        frame_entities = [e["name"] for e in frame_meta["entities"]]

        # Determine if this frame introduces new entities or shows interactions
        new_entities = []
        for entity in frame_meta["entities"]:
            for entity_id, consistent_entity in consistent_entities.items():
                if (
                    entity["name"].lower() == consistent_entity.description.lower()
                    and frame_meta["frame_id"] == consistent_entity.first_seen
                ):
                    new_entities.append(entity_id)

        # Generate event description
        if new_entities:
            event_desc = f"{', '.join(new_entities)} enter the scene."
        elif len(frame_entities) > 1:
            event_desc = f"Multiple entities ({', '.join(frame_entities)}) are present in the scene."
        else:
            event_desc = (
                f"{frame_entities[0] if frame_entities else 'Scene'} is visible."
            )

        events.append(
            Event(
                frame_id=frame_meta["frame_id"],
                timestamp=frame_meta["timestamp"],
                event=event_desc,
            )
        )

    return events


def _enhance_with_llm_analysis(
    frame_metadata_list: List[FrameMetadata],
    consistent_entities: Dict[str, ConsistentEntity],
    events: List[Event],
) -> Dict[str, Any]:
    """
    Use LLM to enhance the entity linking and event extraction with more sophisticated analysis.
    """
    openai_model = os.getenv("OPENAI_MODEL_TEMP")
    if not openai_model:
        raise ValueError("OPENAI_MODEL_TEMP environment variable is required")

    temperature = float(os.getenv("TEMPERATURE_TEMP", "0.6"))

    llm = ChatOpenAI(
        model=openai_model, api_key=openai_api_key, temperature=temperature
    )

    # Prepare context for LLM analysis
    context = {
        "frames": [
            {
                "frame_id": meta["frame_id"],
                "timestamp": meta["timestamp"],
                "scene_description": meta["scene_description"],
                "entities": [
                    {"name": e["name"], "type": e["type"]} for e in meta["entities"]
                ],
            }
            for meta in frame_metadata_list
        ],
        "consistent_entities": {
            k: v.model_dump() for k, v in consistent_entities.items()
        },
        "events": [e.model_dump() for e in events],
    }

    prompt = f"""
    You are an expert story analyst. Analyze the following frame sequence and create a detailed narrative with enhanced entity tracking and event extraction.
    
    Frame Data: {json.dumps(context['frames'], indent=2)}
    
    Current Entity Tracking: {json.dumps(context['consistent_entities'], indent=2)}
    Current Events: {json.dumps(context['events'], indent=2)}
    
    Your task is to:
    1. **Enhance Character Descriptions**: Provide detailed characteristics, personality traits, and roles for each entity
    2. **Improve Event Narratives**: Create more engaging and detailed event descriptions that tell a story
    3. **Identify Interactions**: Detect relationships and interactions between entities across frames
    4. **Add Context**: Provide significance and meaning to each event in the overall narrative
    
    Focus on creating a cohesive story that flows naturally from frame to frame.
    
    IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text.
    
    Use this exact JSON structure:
    {{
        "characters": [
            {{
                "entity_id": "animal_1",
                "description": "A black crow with glossy feathers, intelligent eyes, and a distinctive caw",
                "characteristics": ["intelligent", "opportunistic", "resourceful"],
                "role": "protagonist seeking food"
            }}
        ],
        "events": [
            {{
                "frame_id": "frame_001.jpg",
                "timestamp": "2025-08-05T14:00:46Z",
                "event": "The crow takes flight from a tree branch, clutching a piece of bread in its beak",
                "entities_involved": ["animal_1"],
                "significance": "Establishes the crow as the main character and shows its resourcefulness"
            }}
        ]
    }}
    
    Ensure all entity_ids match those from the current entity tracking.
    """

    try:
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])

        # Parse the response
        try:
            # Remove markdown code blocks if present
            content = clean_json_text(response.content)

            enhanced_analysis = json.loads(content)
            print(
                f"LLM enhancement successful: {len(enhanced_analysis.get('characters', []))} characters, {len(enhanced_analysis.get('events', []))} events"
            )
            return enhanced_analysis
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response.content[:500]}...")
            # Fallback to original analysis
            return {
                "characters": [v.model_dump() for v in consistent_entities.values()],
                "events": [e.model_dump() for e in events],
            }
    except Exception as e:
        print(f"Error in LLM enhancement: {str(e)}")
        # Fallback to original analysis
        return {
            "characters": [v.model_dump() for v in consistent_entities.values()],
            "events": [e.model_dump() for e in events],
        }


def link_temporal_entities(state: GraphState) -> Dict[str, Any]:
    """
    Main function for the Temporal Entity Linker node.
    Analyzes frame metadata to establish entity continuity and extract events.
    """
    print("Starting Temporal Entity Linking...")

    frame_metadata_list = state["frame_metadata"]

    if not frame_metadata_list:
        print("No frame metadata available for analysis")
        return {"consistent_entities": {"characters": [], "events": []}}

    # Step 1: Resolve entities across frames
    print("Resolving entities across frames...")
    consistent_entities = _resolve_entities(frame_metadata_list)

    # Step 2: Extract events
    print("Extracting events from frame sequence...")
    events = _extract_events(frame_metadata_list, consistent_entities)

    # Step 3: Enhance with LLM analysis
    print("Enhancing analysis with LLM...")
    enhanced_analysis = _enhance_with_llm_analysis(
        frame_metadata_list, consistent_entities, events
    )

    print(
        f"Temporal Entity Linking completed. Found {len(enhanced_analysis['characters'])} characters and {len(enhanced_analysis['events'])} events."
    )

    return {"consistent_entities": enhanced_analysis}
