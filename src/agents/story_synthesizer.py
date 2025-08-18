import json
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from utils import clean_json_text
from config.environment import openai_api_key
from models.states import GraphState


load_dotenv()


def _build_prompt(consistent_entities: Dict[str, Any]) -> str:
    characters = consistent_entities.get("characters", [])
    events = consistent_entities.get("events", [])

    context = {
        "characters": characters,
        "events": events,
    }

    return f"""
You are a skilled story editor. Using the linked characters and chronological events, synthesize a coherent story.

Context (JSON):
{json.dumps(context, indent=2)}

Tasks:
1) Create a short, descriptive Title for the story.
2) Write a concise 2-3 sentence Summary of the full sequence.
3) Produce a simplified list of main characters (2-5 max). Use the character's existing description. Each item must use keys: character_id (entity_id), description.
4) Build the event sequence using each event's frame_id and a clear event_description based on the event text.

Output ONLY a valid JSON object in this exact schema:
{{
  "title": "string",
  "summary": "string (2-3 sentences)",
  "main_characters": [
    {{"character_id": "string", "description": "string"}}
  ],
  "event_sequence": [
    {{"frame_id": "string", "event_description": "string"}}
  ]
}}
"""


def _fallback_synthesis(consistent_entities: Dict[str, Any]) -> Dict[str, Any]:
    characters: List[Dict[str, Any]] = consistent_entities.get("characters", [])
    events: List[Dict[str, Any]] = consistent_entities.get("events", [])

    # Title
    if characters:
        first_desc = characters[0].get("description") or characters[0].get(
            "entity_id", "A Story"
        )
        title = f"Story of {first_desc}"[:80]
    else:
        title = "A Short Story"

    # Main characters
    main_characters = []
    for ch in characters[:5]:
        main_characters.append(
            {
                "character_id": ch.get("entity_id", "entity_1"),
                "description": ch.get("description", "Unknown character"),
            }
        )

    # Event sequence
    event_sequence = []
    for ev in events:
        event_sequence.append(
            {
                "frame_id": ev.get("frame_id", "frame_000.jpg"),
                "event_description": ev.get("event", "An event occurs."),
            }
        )

    # Summary
    if events:
        first_ev = events[0].get("event", "The story begins.")
        last_ev = events[-1].get("event", "It concludes.")
        summary = (
            f"{first_ev} Then, events unfold, leading to the final moment: {last_ev}"[
                :300
            ]
        )
    else:
        summary = "A brief sequence unfolds involving the listed characters."

    return {
        "title": title,
        "summary": summary,
        "main_characters": main_characters,
        "event_sequence": event_sequence,
    }


def synthesize_story(state: GraphState) -> Dict[str, Any]:
    """Final node: synthesize a complete story JSON from consistent entities and events."""
    print("Starting Story Synthesis...")

    consistent_entities = state.get("consistent_entities", {})

    # If nothing to synthesize, return a minimal object
    if not consistent_entities:
        minimal = {
            "title": "A Short Story",
            "summary": "A brief sequence unfolds involving the listed characters.",
            "main_characters": [],
            "event_sequence": [],
        }
        return {"final_story": json.dumps(minimal, ensure_ascii=False)}

    openai_model = os.getenv("OPENAI_MODEL_STORY")
    if not openai_model:
        print("OPENAI_MODEL_STORY not set; using fallback synthesis.")
        synthesized = _fallback_synthesis(consistent_entities)
        return {"final_story": json.dumps(synthesized, ensure_ascii=False)}

    temperature = float(os.getenv("TEMPERATURE_STORY", "0.6"))

    try:
        llm = ChatOpenAI(
            model=openai_model, api_key=openai_api_key, temperature=temperature
        )
        prompt = _build_prompt(consistent_entities)
        response = llm.invoke([HumanMessage(content=prompt)])

        try:
            cleaned = clean_json_text(response.content)
            story = json.loads(cleaned)
        except json.JSONDecodeError:
            print("Story Synthesizer: JSON decode failed. Falling back.")
            story = _fallback_synthesis(consistent_entities)
    except Exception as e:
        print(f"Story Synthesizer error: {str(e)}. Falling back.")
        story = _fallback_synthesis(consistent_entities)

    return {"final_story": json.dumps(story, ensure_ascii=False)}
