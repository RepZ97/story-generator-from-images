# Story Generator from Images

An AI-powered story generation system that analyzes sequences of images and creates cohesive narratives using LangGraph and OpenAI models.

## Overview

This project implements a three-stage AI agent workflow that transforms image sequences into structured stories. It uses advanced vision language models and natural language models together to understand visual content, track entities across frames, and synthesize compelling narratives.

## Features

- **Frame Analysis**: Analyzes individual images to extract scene descriptions and identify entities
- **Temporal Entity Linking**: Tracks characters and objects across multiple frames with consistent identification
- **Story Synthesis**: Generates complete narratives with titles, summaries, and chronological event sequences
- **Structured Output**: Produces well-formatted JSON stories with detailed character and event information
- **LangGraph Workflow**: Uses LangGraph for robust, stateful agent orchestration
- **Comprehensive Logging**: Detailed logging system with file and console output

## Architecture

The system consists of three main agents working in sequence:

1. **Frame Analyzer** (`frame_analyzer.py`)
   - Processes individual images using OpenAI's vision models
   - Extracts scene descriptions and entity information
   - Outputs structured metadata for each frame

2. **Temporal Entity Linker** (`temporal_entity_linker.py`)
   - Links entities across frames using similarity matching
   - Assigns consistent IDs to recurring characters/objects
   - Extracts chronological events and interactions

3. **Story Synthesizer** (`story_synthesizer.py`)
   - Combines linked entities and events into coherent narratives
   - Generates titles, summaries, and detailed event sequences
   - Outputs final JSON story structure

## Prerequisites
- Python 3.8 or higher
- OpenAI API key with access to vision models (e.g., GPT-4V) - GPT-5 is recommended
- Sufficient OpenAI API credits - (Minimal credit requirement)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RepZ97/story-generator-from-images.git
cd story-generator-from-images
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_FRAME=gpt-5
OPENAI_MODEL_TEMP=gpt-5
OPENAI_MODEL_STORY=gpt-5
TEMPERATURE_FRAME=1
TEMPERATURE_TEMP=1
TEMPERATURE_STORY=1
```

### 4. Prepare Image Data

Place your image sequence in one of the folders under `assests/images/`:
- `assests/images/story1/` (currently configured)
- `assests/images/Story2/`
- `assests/images/story3/`

**Image Requirements:**
- Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF
- Images should be in chronological order
- Recommended: 2-10 images per story sequence

## Usage

### Basic Usage

1. **Configure Image Path** (if needed):
   Edit `src/main.py` and update the `folder_path` variable:
   ```python
   folder_path = "assests/images/story1"
   ```

2. **Run the Story Generator**:
   ```bash
   python src/main.py
   ```

### Example Output

The system will generate output in three stages:

```
=== Frame Analysis Results ===
Frame: frame_001.jpg
Timestamp: 2025-01-15T10:30:00Z
Scene: A crow is flying with a piece of bread in its beak
Entities found: 1
  - crow: animal

=== Temporal Entity Linking Results ===
Characters (2):
  - animal_1: A black crow with glossy feathers, intelligent eyes
    Characteristics: intelligent, opportunistic, resourceful
    Role: protagonist seeking food

=== Final Story JSON ===
{
  "title": "The Clever Crow's Quest",
  "summary": "A resourceful crow discovers food and navigates challenges...",
  "main_characters": [
    {"character_id": "animal_1", "description": "A black crow with glossy feathers"}
  ],
  "event_sequence": [
    {"frame_id": "frame_001.jpg", "event_description": "The crow takes flight with bread..."}
  ]
}
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL_FRAME` | Model for frame analysis | `gpt-4o` |
| `OPENAI_MODEL_TEMP` | Model for temporal linking | `gpt-4o` |
| `OPENAI_MODEL_STORY` | Model for story synthesis | `gpt-4o` |
| `TEMPERATURE_FRAME` | Creativity for frame analysis | `0.1` |
| `TEMPERATURE_TEMP` | Creativity for temporal linking | `0.6` |
| `TEMPERATURE_STORY` | Creativity for story synthesis | `0.6` |

### Data Models

The system uses several key data structures:

- **`GraphState`**: Main workflow state containing image paths, metadata, entities, and final story
- **`FrameMetadata`**: Individual frame analysis results
- **`Entity`**: Detected objects/characters with attributes
- **`ConsistentEntity`**: Entities tracked across multiple frames
- **`Event`**: Chronological events in the story

## Customization

### Adding New Image Folders

1. Create a new folder in `assests/images/`
2. Add your image sequence
3. Update the `folder_path` in `src/main.py`

### Modifying Agent Behavior

Each agent can be customized by editing the respective files:
- `src/agents/frame_analyzer.py`: Adjust image analysis prompts
- `src/agents/temporal_entity_linker.py`: Modify entity linking logic
- `src/agents/story_synthesizer.py`: Change story generation style

### Output Format

The final JSON output includes:
- **title**: Descriptive story title
- **summary**: 2-3 sentence overview
- **main_characters**: List of key characters with descriptions
- **event_sequence**: Chronological events with frame references

### Debug Mode

The application uses a comprehensive logging system that provides detailed information about the processing pipeline:

- **Console Output**: Real-time logging to console with timestamps and log levels
- **File Logging**: Detailed logs saved to `logs/story_generator.log`
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Module-specific Loggers**: Each component has its own logger for better debugging

To enable debug logging, modify the logging level in `src/main.py`:
```python
setup_logging(log_level="DEBUG", log_file="logs/story_generator.log")
```

Log entries include:
- Timestamp and module name
- Log level (INFO, WARNING, ERROR, etc.)
- Detailed processing information
- Error messages with context
- Performance metrics and entity tracking
