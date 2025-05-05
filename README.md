# ABSRefined: AudioBookShelf Chapter Marker Refiner

ABSRefined is a tool for improving chapter markers in audiobooks hosted on AudioBookShelf servers. It uses audio transcription and large language models to determine the precise location of chapter transitions, with the help of existing chapter markers.

## Overview

At least in my files, many audiobooks chapter markers fetched from audiable aren't precisely aligned with the actual chapter beginnings. This tool:

1. Connects to your AudioBookShelf server
2. Downloads audio segments around existing chapter markers
3. Transcribes the audio to text
4. Uses an LLM to analyze the transcription and determine if the marker needs adjustment
5. Updates the AudioBookShelf server with refined chapter positions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/absrefined.git
cd absrefined

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with your credentials:

```
ABS_USERNAME=your_audiobookshelf_username
ABS_PASSWORD=your_audiobookshelf_password
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_URL=https://api.openai.com/v1  # or your custom endpoint
OPENAI_MODEL=gpt-4.1  # or other compatible model
```

## Usage

### Basic Usage

```bash
# Process a book using its URL from AudioBookShelf
abs-chapter-refiner --book-url "https://your-abs-server.com/library/item/your-book-id"

# Specify server manually
abs-chapter-refiner --server "https://your-abs-server.com" --book-url "..."
```

### Available Options

| Option | Description |
|--------|-------------|
| `--server` | AudioBookShelf server URL |
| `--book-url` | URL of the book to process (can extract server and item ID automatically) |
| `--llm-api` | OpenAI-compatible API URL |
| `--model` | LLM model to use |
| `--window` | Window size in seconds around each chapter marker (default: 15) |
| `--temp-dir` | Directory for temporary files (default: "temp") |
| `--dry-run` | Don't actually update the server |
| `--just-download` | Only download the audio file without processing |
| `--verbose` | Enable verbose output |

## How It Works

1. **Initialization**: Command line arguments and environment variables are processed
2. **API Connection**: Establishes connection to the AudioBookShelf server
3. **Refinement Process**:
   - Retrieves book and chapter information from the server
   - Downloads audio files from AudioBookShelf
   - Transcribes audio segments around chapter markers
   - Analyzes transcriptions using an LLM
   - Updates refined chapter markers on the AudioBookShelf server

## Project Structure

- `absrefined/main.py`: Main entry point with CLI interface
- `absrefined/client/abs_client.py`: Client for interacting with AudioBookShelf API
- `absrefined/transcriber/audio_transcriber.py`: Handles audio transcription
- `absrefined/refiner/chapter_refiner.py`: Refines chapter markers using LLM
- `absrefined/refinement_tool/chapter_refinement_tool.py`: Orchestrates the refinement process
- `absrefined/utils/timestamp.py`: Utilities for timestamp handling

## Dependencies

- requests: For API communication
- openai: For LLM and transcription API access
- tqdm: For progress bars
- dotenv: For environment variable management
- pytest: For testing

## License

This project is licensed under the terms of the license included in the repository. 