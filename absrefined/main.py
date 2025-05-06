#!/usr/bin/env python3
import argparse
import os
import re
from getpass import getpass
from dotenv import load_dotenv
import logging
import shutil
import atexit
from absrefined.utils.timestamp import format_timestamp

from absrefined.client import AudiobookshelfClient
from absrefined.transcriber import AudioTranscriber
from absrefined.refiner import ChapterRefiner
from absrefined.refinement_tool import ChapterRefinementTool

# Global variable to hold temp dir path for cleanup
_main_temp_dir_to_clean = None

def _cleanup_main_temp_dir():
    global _main_temp_dir_to_clean
    if _main_temp_dir_to_clean and os.path.isdir(_main_temp_dir_to_clean):
        logger = logging.getLogger("_cleanup_main_temp_dir")
        logger.info(f"Cleaning up main temporary directory: {_main_temp_dir_to_clean}")
        try:
            shutil.rmtree(_main_temp_dir_to_clean)
            logger.debug(f"Successfully removed directory: {_main_temp_dir_to_clean}")
        except OSError as e:
            logger.warning(f"Error removing directory {_main_temp_dir_to_clean}: {e}")
    _main_temp_dir_to_clean = None # Reset after attempt

# Register the cleanup function to be called on exit
atexit.register(_cleanup_main_temp_dir)

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

    # Create temp directory and register for cleanup
    global _main_temp_dir_to_clean
    temp_dir = args.temp_dir or "temp"
    _main_temp_dir_to_clean = os.path.abspath(temp_dir) # Store absolute path
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Using temporary directory: {_main_temp_dir_to_clean}")

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
    result = tool.process_item(item_id)

    # --- Process CLI Results --- #
    logger.info("\n--- Refinement Results ---")
    updated_on_server = False # Flag to track if update occurred

    if result.get("error"):
        logger.error(f"Processing failed: {result['error']}")
        return 1

    chapter_details = result.get("chapter_details")
    total_chapters = result.get("total_chapters", 0)
    # We re-calculate refined count based on significant changes for CLI
    cli_refined_count = 0
    significant_change_threshold = 0.5 # seconds (can be different from GUI threshold)
    changes_to_confirm = []
    updates_for_server = []

    if not chapter_details:
        logger.warning("Processing finished, but no chapter details were returned.")
    else:
        logger.info(f"Found {len(chapter_details)} chapters.")
        logger.info("Comparing original and refined timestamps...")

        for i, detail in enumerate(chapter_details):
            original_start = detail.get("original_start")
            refined_start = detail.get("refined_start")
            chapter_title = detail.get("title", f"Chapter {i+1}")
            chapter_id = detail.get("id")

            if original_start is None or refined_start is None:
                 # Log if refinement didn't produce a time
                 if refined_start is None and i > 0: # Don't warn for chapter 0 if refinement failed
                      logger.debug(f"Chapter '{chapter_title}': No refined timestamp available.")
                 continue # Skip comparison if times aren't available

            time_diff = refined_start - original_start

            if args.verbose:
                 logger.debug(f"Chapter '{chapter_title}' (ID: {chapter_id}):")
                 logger.debug(f"  Original: {format_timestamp(original_start)}")
                 logger.debug(f"  Refined:  {format_timestamp(refined_start)}")
                 logger.debug(f"  Difference: {time_diff:+.3f}s")

            # Check for significant change (and ignore first chapter)
            if i > 0 and abs(time_diff) > significant_change_threshold:
                cli_refined_count += 1
                change_info = {
                    "index": i,
                    "title": chapter_title,
                    "id": chapter_id,
                    "original": original_start,
                    "refined": refined_start,
                    "diff": time_diff,
                }
                changes_to_confirm.append(change_info)
                updates_for_server.append({"id": chapter_id, "start": max(0.0, refined_start)})

        # --- Confirmation and Update --- #
        if changes_to_confirm:
            logger.info(
                f"\nFound {len(changes_to_confirm)} chapters with changes > {significant_change_threshold}s."
            )
            logger.info("\nPreview of significant changes:")
            for change in changes_to_confirm:
                diff_str = f"({change['diff']:+.3f}s)"
                logger.info(
                    f"  Chapter '{change['title']}': {format_timestamp(change['original'])} -> {format_timestamp(change['refined'])} {diff_str}"
                )

            if not args.dry_run:
                prompt = "\nDo you want to apply these changes to the server? (y/n): "
                try:
                    confirm = input(prompt).strip().lower()
                except EOFError:
                    logger.warning(
                        "Non-interactive environment detected, cancelling update."
                    )
                    confirm = "n"

                if confirm == "y":
                    logger.info("Attempting to update chapters on the server...")
                    try:
                        # Use the new client method for partial updates
                        success = client.update_chapters_start_time(item_id, updates_for_server)
                        if success:
                            logger.info("Server update successful.")
                            updated_on_server = True
                        else:
                            logger.error("Server update failed (check client logs).")
                            # No need to return 1 here, just report failure in summary
                    except Exception as update_err:
                         logger.error(f"Error during server update: {update_err}", exc_info=args.verbose)
                else:
                    logger.info("Update cancelled by user.")
            else:
                 logger.info("\nDRY RUN: Changes detected, but not applying to server.")
        else:
             logger.info("\nNo significant changes found requiring server update.")

    # --- Final Summary --- #
    logger.info("\n--- Final Summary ---")
    logger.info(f"Item ID: {item_id}")
    logger.info(f"Total chapters found: {total_chapters}")
    logger.info(f"Chapters with significant changes detected: {cli_refined_count}")

    update_status = "N/A (No significant changes)"
    if cli_refined_count > 0:
         if args.dry_run:
             update_status = "Dry Run (No changes applied)"
         elif updated_on_server:
             update_status = "Yes (Applied)"
         else:
             update_status = "No (Confirmation declined or failed)"
    logger.info(f"Server updated: {update_status}")

    # Cleanup is handled by atexit

    return 0 # Return 0 unless a critical error occurred earlier


if __name__ == "__main__":
    # Consider adding try/except around main() call for very top-level errors
    exit_code = main()
    exit(exit_code)
