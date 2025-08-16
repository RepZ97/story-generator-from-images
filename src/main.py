import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils import read_images_on_folder
from models.states import GraphState
from agents.frame_analyzer import analyze_frames


def main():
    folder_path = "assests/images/story3"
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

    workflow.add_edge(START, "analyze_frames")
    workflow.add_edge("analyze_frames", END)

    workflow = workflow.compile(checkpointer=MemorySaver())

    # Run the workflow
    print("Starting story generation workflow...")
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    result = workflow.invoke({"image_paths": image_paths}, config)

    for metadata in result["frame_metadata"]:
        print(f"\nFrame: {metadata['frame_id']}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Scene: {metadata['scene_description']}")
        print(f"Entities found: {len(metadata['entities'])}")
        
        for entity in metadata['entities']:
            print(f"  - {entity['name']}: {entity['type']}")

if __name__ == "__main__":
    main()
