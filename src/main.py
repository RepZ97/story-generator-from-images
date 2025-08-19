import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils import read_images_on_folder
from models.states import GraphState
from agents.frame_analyzer import analyze_frames
from agents.temporal_entity_linker import link_temporal_entities
from agents.story_synthesizer import synthesize_story
from config.logging_config import setup_logging, get_logger

# Set up logging
logger = get_logger(__name__)

def main():
    # Set up logging configuration
    setup_logging(log_level="INFO", log_file="logs/story_generator.log")
    
    folder_path = "assests/images/story1"
    if not os.path.isdir(folder_path):
        logger.error(f"The folder at '{folder_path}' does not exist.")
        return

    image_paths = read_images_on_folder(folder_path)
    logger.info(f"Found {len(image_paths)} images in {folder_path}")

    workflow = StateGraph(GraphState)
    workflow.add_node(
        "analyze_frames",
        analyze_frames,
        inputs=["image_paths"],
        outputs=["frame_metadata"],
    )

    workflow.add_node(
        "link_temporal_entities",
        link_temporal_entities,
        inputs=["frame_metadata"],
        outputs=["consistent_entities"],
    )

    workflow.add_node(
        "synthesize_story",
        synthesize_story,
        inputs=["consistent_entities"],
        outputs=["final_story"],
    )

    workflow.add_edge(START, "analyze_frames")
    workflow.add_edge("analyze_frames", "link_temporal_entities")
    workflow.add_edge("link_temporal_entities", "synthesize_story")
    workflow.add_edge("synthesize_story", END)

    workflow = workflow.compile(checkpointer=MemorySaver())

    # Run the workflow
    logger.info("Starting story generation workflow...")
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    result = workflow.invoke({"image_paths": image_paths}, config)

    # Display frame analysis results
    logger.debug("=== Frame Analysis Results ===")
    for metadata in result["frame_metadata"]:
        logger.debug(f"Frame: {metadata['frame_id']}")
        logger.debug(f"Timestamp: {metadata['timestamp']}")
        logger.debug(f"Scene: {metadata['scene_description']}")
        logger.debug(f"Entities found: {len(metadata['entities'])}")

        for entity in metadata["entities"]:
            logger.debug(f"  - {entity['name']}: {entity['type']}")

    # Display temporal entity linking results
    logger.debug("=== Temporal Entity Linking Results ===")
    consistent_entities = result["consistent_entities"]

    logger.debug(f"Characters ({len(consistent_entities['characters'])}):")
    for character in consistent_entities["characters"]:
        logger.debug(f"  - {character['entity_id']}: {character['description']}")
        if "characteristics" in character:
            logger.debug(f"    Characteristics: {', '.join(character['characteristics'])}")
        if "role" in character:
            logger.debug(f"    Role: {character['role']}")

    logger.debug(f"Events ({len(consistent_entities['events'])}):")
    for event in consistent_entities["events"]:
        logger.debug(f"  - Frame {event['frame_id']}: {event['event']}")
        if "entities_involved" in event:
            logger.debug(f"    Entities: {', '.join(event['entities_involved'])}")
        if "significance" in event:
            logger.debug(f"    Significance: {event['significance']}")

    # Display final synthesized story
    logger.info("=== Final Story JSON ===")
    logger.info(result["final_story"])


if __name__ == "__main__":
    main()
