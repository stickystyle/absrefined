# ABSRefined: AudioBookShelf Chapter Marker Refiner

ABSRefined is a tool for improving chapter markers in audiobooks hosted on AudioBookShelf servers. It uses audio transcription and large language models to determine the precise location of chapter transitions, guided by existing chapter markers.

## Overview

Many audiobook chapter markers, especially those imported from external sources, aren't precisely aligned with the actual chapter beginnings (often indicated by spoken chapter titles or distinct pauses). This tool aims to fix that.

ABSRefined offers two interfaces:

1.  **Command-Line Interface (CLI):** For processing books directly from the terminal.
2.  **Graphical User Interface (GUI):** For a more interactive experience, allowing review of changes before applying them.

The core process involves:

1.  Connecting to your AudioBookShelf server.
2.  Downloading audio segments around existing chapter markers.
3.  Transcribing these segments to text using an OpenAI-compatible API.
4.  Using an LLM (via the same API) to analyze the transcription and determine the most likely precise start time of the chapter within the segment.
5.  Presenting the proposed changes (via CLI or GUI).
6.  Updating the AudioBookShelf server with the confirmed refined chapter positions.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/absrefined.git # Replace with actual URL if different
cd absrefined

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install dependencies
# Ensure you have pip and potentially uv installed
# pip install -r requirements.txt
# Or using uv:
# uv pip install -r requirements.txt
# Install the package itself if you want to use the CLI command directly
pip install -e .

# 4. Ensure ffmpeg is installed and in your system PATH
# (Required for audio processing)
# Check installation: ffmpeg -version
# Installation methods vary by OS (e.g., brew install ffmpeg on macOS, apt install ffmpeg on Debian/Ubuntu)
```

## Configuration

Create a `.env` file in the root directory with your credentials and API details:

```dotenv
# Audiobookshelf Credentials
ABS_USERNAME=your_audiobookshelf_username
ABS_PASSWORD=your_audiobookshelf_password

# OpenAI-Compatible API Details
OPENAI_API_KEY=your_openai_api_key_or_equivalent
# Optional: Specify URL if not using standard OpenAI endpoint
# OPENAI_API_URL=https://api.openai.com/v1
# Optional: Specify the model to use for refinement (default: gpt-4o-mini)
# OPENAI_MODEL=gpt-4o-mini
```

## Usage

### Graphical User Interface (GUI)

This is the recommended way for interactive use and reviewing changes.

1.  Ensure your `.env` file is configured.
2.  Run the GUI script from the project root:

```bash
python gui.py
```

**GUI Features:**

*   Enter the Book URL from your Audiobookshelf instance.
*   Adjust the processing window size and LLM model if needed.
*   View processing progress.
*   Review original vs. refined timestamps for each chapter.
*   Use the ▶ button to play a short audio segment starting at the proposed timestamp (requires `simpleaudio` library: `pip install simpleaudio`).
*   Select which chapters to apply changes to using the checkboxes.
*   Push the selected changes back to the Audiobookshelf server (disable "Dry Run" first).
*   Cancel ongoing processing or server updates.

### Command-Line Interface (CLI)

Suitable for scripting or non-interactive use.

1.  Ensure your `.env` file is configured.
2.  Run the tool using the installed command or directly via `python -m absrefined.main`:

```bash
# Process a book using its URL from AudioBookShelf
# You will be prompted to confirm changes unless --dry-run is used.
abs-chapter-refiner --book-url "https://your-abs-server.com/library/item/your-book-id"

# Or run as a module
python -m absrefined.main --book-url "..."

# Example: Specify server manually and use verbose output
abs-chapter-refiner --server "https://your-abs-server.com" --book-url "..." --verbose

# Example: Dry run (show proposed changes but don't update server)
abs-chapter-refiner --book-url "..." --dry-run
```

**CLI Options:**

| Option        | Description                                                            |
|---------------|------------------------------------------------------------------------|
| `--server`    | AudioBookShelf server URL (required if not in `--book-url`)            |
| `--book-url`  | URL of the book item page (extracts server and item ID automatically)  |
| `--llm-api`   | OpenAI-compatible API URL (overrides `OPENAI_API_URL` in `.env`)       |
| `--model`     | LLM model to use (overrides `OPENAI_MODEL` in `.env`)                |
| `--window`    | Window size in seconds around each chapter marker (default: 15)        |
| `--temp-dir`  | Directory for temporary files (default: `temp`)                        |
| `--dry-run`   | Perform analysis but do not prompt or update the server                |
| `--verbose`   | `-v`, Enable detailed DEBUG level logging                              |
| `--debug`     | Enable debug mode (keeps temporary audio chunks and transcripts)       |

*Note: The `--just-download` option was removed as download is integrated into the main workflow.* 

## How It Works (Core Logic)

1.  **Initialization**: Reads configuration (`.env`), command-line arguments (for CLI), or GUI inputs.
2.  **API Connection**: `AudiobookshelfClient` connects and authenticates to the server.
3.  **Refinement Process** (`ChapterRefinementTool`):
    *   Fetches original chapter data.
    *   Downloads the full audio file if needed (cached in `temp_dir`).
    *   For each chapter (except the first):
        *   Extracts a small audio window (`.wav` chunk) around the original marker using `ffmpeg`.
        *   Transcribes the chunk using `AudioTranscriber`.
        *   Passes the transcription and chapter context to `ChapterRefiner` (LLM call).
        *   Stores the original and potentially refined timestamps (floats).
4.  **Results Handling**:
    *   **GUI**: Populates the results table, allowing user review, playback (from chunk), selection, and final update push via `AudiobookshelfClient`.
    *   **CLI**: Compares original and refined times, displays significant changes, prompts for confirmation (if not `--dry-run`), and updates the server via `AudiobookshelfClient` if confirmed.

## Project Structure

- `gui.py`: Graphical User Interface (Tkinter).
- `absrefined/main.py`: Main entry point for the CLI.
- `absrefined/client/abs_client.py`: Client for interacting with AudioBookShelf API.
- `absrefined/transcriber/audio_transcriber.py`: Handles audio transcription.
- `absrefined/refiner/chapter_refiner.py`: Refines chapter markers using LLM.
- `absrefined/refinement_tool/chapter_refinement_tool.py`: Orchestrates the core refinement workflow (used by both GUI and CLI).
- `absrefined/utils/timestamp.py`: Utilities for timestamp handling.

## Dependencies

Key dependencies are listed in `requirements.txt`. Install using `pip install -r requirements.txt` or `uv pip install -r requirements.txt`.

- `requests`: For API communication.
- `openai`: For LLM and transcription API access.
- `python-dotenv`: For environment variable management.
- `simpleaudio` (Optional, for GUI playback): For playing `.wav` audio chunks.

External Dependencies:

- `ffmpeg` and `ffprobe`: Required for audio file processing (downloading segments, getting duration). Must be installed separately and available in the system PATH.

## License

This project is licensed under the terms of the license included in the repository. 