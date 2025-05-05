#!/usr/bin/env python3
import argparse
import os
import re
from getpass import getpass
from dotenv import load_dotenv
import logging

from absrefined.client import AudiobookshelfClient
from absrefined.transcriber import AudioTranscriber
from absrefined.refiner import ChapterRefiner
from absrefined.refinement_tool import ChapterRefinementTool


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="AudioBookShelf Chapter Marker Refiner"
    )
    parser.add_argument("--server", help="Audiobookshelf server URL")
    parser.add_argument("--llm-api", help="OpenAI-compatible API URL")
    parser.add_argument("--model", help="LLM model to use")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (save transcripts and audio chunks)"
    )
    parser.add_argument("--temp-dir", help="Directory for temporary files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually update ABS server"
    )
    parser.add_argument(
        "--just-download",
        action="store_true",
        help="Just download the audio file without processing",
    )
    parser.add_argument("--book-url", help="URL of the Audiobookshelf book to process")
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Window size in seconds around each chapter marker (default: 15)",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Initialize env var values
    load_dotenv()
    username = os.getenv("ABS_USERNAME")
    password = os.getenv("ABS_PASSWORD")
    llm_api_url = args.llm_api or os.getenv("OPENAI_API_URL")
    llm_api_key = os.getenv("OPENAI_API_KEY", "")
    llm_model = args.model or os.getenv("OPENAI_MODEL", "gpt-4.1")

    # Extract item_id and server_url from book-url if provided
    item_id = None
    server_url = None
    if args.book_url:
        # Try to extract server_url and item_id from URL
        match = re.search(
            r"(https?://[^/]+)(?:/[^/]+)*/item/([a-zA-Z0-9\\-]+)", args.book_url
        )
        if match:
            server_url = match.group(1)
            item_id = match.group(2)
            logger.info(
                f"Extracted server URL: {server_url} and item ID: {item_id} from URL: {args.book_url}"
            )
        else:
            logger.error(
                f"Failed to extract server URL and item ID from URL: {args.book_url}"
            )
            return 1
    else:
        # Fallback: use --server argument
        server_url = args.server

    # Validate required parameters
    if not server_url:
        server_url = input("Enter Audiobookshelf server URL: ")

    # Check if credentials are available in env vars
    if username and password:
        logger.info("Using credentials from environment variables")
    else:
        username = input("Enter username: ")
        password = getpass("Enter password: ")

    # Initialize clients
    client = AudiobookshelfClient(server_url, verbose=args.verbose)
    client.login(username, password)

    # Create temp directory
    temp_dir = args.temp_dir or "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize refiner first (needed by the tool)
    refiner = ChapterRefiner(
        llm_api_url,
        llm_model,
        window_size=args.window,
        verbose=args.verbose,
        llm_api_key=llm_api_key,
    )
    logger.debug(
        f"Initialized LLM refiner with API at {llm_api_url} using model {llm_model}"
    )
    logger.debug(
        f"Using window size of ±{args.window} seconds around each chapter marker"
    )

    # Initialize transcriber (needed by the tool)
    # Pass the OpenAI API key, which is the same as the LLM key
    transcriber = AudioTranscriber(api_key=llm_api_key, verbose=args.verbose, debug=args.debug)
    logger.debug("Initialized Audio Transcriber using OpenAI API.")

    # Create the refinement tool (needs client, transcriber, refiner)
    # Instantiate the tool *before* processing, as it now handles download/transcription
    tool = ChapterRefinementTool(
        client,
        transcriber,
        refiner,
        verbose=args.verbose,
        temp_dir=temp_dir,
        dry_run=args.dry_run,
        debug=args.debug,
        # Pass chunk duration if needed, default is in the tool
        # chunk_duration= # optional override
    )
    logger.debug("Initialized Chapter Refinement Tool.")


    if not item_id:
        logger.error("No book/item ID provided or extracted from URL.") # Made error more specific
        return 1

    # --- Removed Download and Transcription Logic --- #
    # The ChapterRefinementTool now handles downloading and transcription internally
    # within its process_item -> process_chapters -> _get_or_create_transcription methods.

    # Handle --just-download if specified
    if args.just_download:
        # We need to trigger the download part of the tool's logic.
        # We can call the transcription method, which handles download first.
        # It might be slightly wasteful if transcription exists, but ensures download.
        logger.info(f"Ensuring audio for {item_id} is downloaded (if needed)...")
        # Call the internal method, providing a dummy path just to trigger download check if needed.
        # We don't care about the segments here.
        _, audio_path = tool._get_or_create_transcription(item_id)
        if audio_path and os.path.exists(audio_path):
             logger.info(f"Audio file is available at: {audio_path}")
             logger.info("--just-download specified, exiting.")
             return 0
        else:
             logger.error("Failed to ensure audio file was downloaded.")
             return 1
        # --- End --just-download Handling ---


    # --- Process Item using the Tool --- #
    logger.info(f"Starting refinement process for item: {item_id}")
    # process_item now orchestrates everything: fetching chapters, getting transcription, refining, comparing, updating.
    result = tool.process_item(item_id)

    # --- Show Summary --- #
    logger.info("\n--- Refinement Summary ---") # Added separator
    if result.get("error"):
         logger.error(f"Processing failed: {result['error']}")
    else:
        logger.info(f"Item ID: {result.get('item_id')}")
        logger.info(f"Total chapters found: {result.get('total_chapters', 0)}")
        logger.info(f"Chapters refined: {result.get('refined_chapters', 0)}")
        if result.get('refined_chapters', 0) > 0:
             update_status = "Yes" if result.get('updated_on_server', False) else "No (Confirmation declined or failed)"
             if args.dry_run:
                 update_status = "Dry Run (No changes applied)"
             logger.info(f"Server updated: {update_status}")
        else:
             logger.info("Server updated: No significant changes to apply")


    return 0 if not result.get("error") else 1 # Return non-zero on error


if __name__ == "__main__":
    # Consider adding try/except around main() call for very top-level errors
    exit_code = main()
    exit(exit_code)
