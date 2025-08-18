import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils import read_images_on_folder
from models.states import GraphState
from agents.frame_analyzer import analyze_frames
from agents.temporal_entity_linker import link_temporal_entities
from agents.story_synthesizer import synthesize_story


def main():
    folder_path = "assests/images/story1"
    if not os.path.isdir(folder_path):
        print(f"Error: The folder at '{folder_path}' does not exist.")

    image_paths = read_images_on_folder(folder_path)

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
    print("Starting story generation workflow...")
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    result = workflow.invoke({"image_paths": image_paths}, config)

    # Display frame analysis results
    print("\n=== Frame Analysis Results ===")
    for metadata in result["frame_metadata"]:
        print(f"\nFrame: {metadata['frame_id']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Scene: {metadata['scene_description']}")
        print(f"Entities found: {len(metadata['entities'])}")

        for entity in metadata["entities"]:
            print(f"  - {entity['name']}: {entity['type']}")

    # Display temporal entity linking results
    print("\n=== Temporal Entity Linking Results ===")
    consistent_entities = result["consistent_entities"]

    print(f"\nCharacters ({len(consistent_entities['characters'])}):")
    for character in consistent_entities["characters"]:
        print(f"  - {character['entity_id']}: {character['description']}")
        if "characteristics" in character:
            print(f"    Characteristics: {', '.join(character['characteristics'])}")
        if "role" in character:
            print(f"    Role: {character['role']}")

    print(f"\nEvents ({len(consistent_entities['events'])}):")
    for event in consistent_entities["events"]:
        print(f"  - Frame {event['frame_id']}: {event['event']}")
        if "entities_involved" in event:
            print(f"    Entities: {', '.join(event['entities_involved'])}")
        if "significance" in event:
            print(f"    Significance: {event['significance']}")

    # Display final synthesized story
    print("\n=== Final Story JSON ===")
    print(result["final_story"])


if __name__ == "__main__":
    main()
