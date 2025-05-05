# AudioBookShelf Chapter Marker Refiner

This tool refines chapter markers in audiobooks hosted on an AudioBookShelf server. It uses mlx-whisper for transcription and an LLM to determine the precise location of chapter markers.

## Features

- Authenticate with your AudioBookShelf server
- Download audio files and extract chapters
- Transcribe audio using mlx-whisper with word-level timestamps
- Use LLM (OpenAI compatible) to determine precise chapter start times
- Update chapter markers on the server

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/abs_chapter_refiner.git
cd abs_chapter_refiner

# Install the package
pip install -e .
```

## Usage

You can use the tool either as a command-line application or import the modules in your own code.

### Command-line usage

```bash
# Using the CLI script
abs-chapter-refiner --server https://your-abs-server.com --book-url "https://your-abs-server.com/item/YOUR-ITEM-ID" --verbose
```

### Configuration

The tool supports configuration via environment variables:

```bash
# Create a .env file
echo "ABS_SERVER_URL=https://your-abs-server.com" > .env
echo "ABS_USERNAME=your_username" >> .env
echo "ABS_PASSWORD=your_password" >> .env
echo "OPENAI_API_URL=https://api.openai.com/v1" >> .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env
echo "OPENAI_MODEL=gpt-3.5-turbo" >> .env
```

### Module Structure

The package has been refactored into modular components:

- `absrefined.client`: Client for interacting with the AudioBookShelf API
- `absrefined.transcriber`: Transcriber for audio files using mlx-whisper
- `absrefined.refiner`: Refiner for chapter markers using LLM analysis
- `absrefined.refinement_tool`: Tool to orchestrate the refinement process
- `absrefined.utils`: Utility functions for timestamp handling, etc.

## Requirements

- Python 3.7 or higher
- requests
- mlx-whisper
- tqdm
- pathlib
- python-dotenv

## License

[MIT](LICENSE)