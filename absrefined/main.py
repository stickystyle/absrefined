#!/usr/bin/env python3
import argparse
import logging
import os
import re
import shutil
import tempfile
import atexit
from pathlib import Path

from absrefined.client import AudiobookshelfClient
from absrefined.config import ConfigError, get_config
from absrefined.refinement_tool import ChapterRefinementTool
from absrefined.utils.timestamp import format_timestamp
from absrefined.utils.url_utils import extract_item_id_from_url


# Set up temp directory cleanup on exit
def _cleanup_temp_files(path):
    """Clean up temporary files when the application exits."""
    if not path or not os.path.exists(path):
        return

    logger = logging.getLogger("cleanup")
    try:
        logger.info(f"Cleaning up temporary directory: {path}")
        shutil.rmtree(path)
        logger.debug(f"Successfully removed directory: {path}")
    except OSError as e:
        logger.warning(f"Error removing directory {path}: {e}")


def main():
    """
    Main entry point for the application with command-line interface.

    This function:
    1. Processes command-line arguments
    2. Loads configuration from a TOML file
    3. Initializes the Audiobookshelf client and refinement tool
    4. Runs the refinement process for a specified item
    5. Reports on the changes made to chapter markers

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="ABSRefined: AudioBookShelf Chapter Marker Refinement Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Input Item ---
    parser.add_argument(
        "item_specifier",
        help="The Item ID or the full Audiobookshelf Item URL (e.g., http://host/item/item-id).",
    )
    # --- Configuration ---
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),  # Default to config.toml in current dir
        help="Path to the TOML configuration file (default: config.toml).",
    )
    # --- Overrides ---
    parser.add_argument(
        "--model", help="Override the LLM model specified in the config file."
    )
    parser.add_argument(
        "--window",
        type=int,
        help="Override the search window size (seconds) specified in the config file.",
    )
    parser.add_argument(
        "--download-path",
        type=Path,
        help="Override the temporary download path specified in the config file.",
    )
    # --- Operational Controls ---
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process chapters but do not push any updates to the Audiobookshelf server.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Automatically confirm and push updates to the server if not in dry-run mode.",
    )
    parser.add_argument(
        "--just-download",  # Kept for utility
        action="store_true",
        help="Ensure the audio file is downloaded (to the configured/overridden path) and exit.",
    )
    # --- Logging / Debug ---
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable INFO level logging (overrides config file setting).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG level logging (overrides config file setting). Implies --verbose.",
    )
    # Note: --debug flag here controls LOG level. File preservation is set in config['logging']['debug_files']

    args = parser.parse_args()

    # Set up minimal logging first to catch early errors
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Load Configuration ---
    config = {}
    try:
        config_path = args.config.resolve()  # Get absolute path
        if not config_path.is_file():
            raise ConfigError(f"Config file not found at specified path: {config_path}")
        config = get_config(config_path)
        logging.debug(f"Loaded configuration from: {config_path}")
    except ConfigError as e:
        logging.error(f"Configuration Error: {e}")
        # Attempt to provide guidance on creating config.toml
        example_path = Path(__file__).parent.parent / "config.example.toml"
        if example_path.exists():
            logging.error(f"See example configuration at: {example_path}")
            logging.error("Copy it to config.toml and fill in your details.")
        else:
            logging.error(
                "Ensure a valid config.toml exists or use --config to specify the path."
            )
        return 1  # Exit on config error
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1

    # --- Apply CLI Overrides to Config (in memory) ---
    # This allows components initialized with config to see overrides
    if args.download_path:
        # Ensure 'processing' section exists
        if "processing" not in config:
            config["processing"] = {}
        abs_download_path = str(args.download_path.resolve())  # Use absolute path
        config["processing"]["download_path"] = abs_download_path
        logging.debug(f"Overriding download path with: {abs_download_path}")

    # If download_path not provided in args or config, use system temp directory
    if "processing" not in config:
        config["processing"] = {}

    # Path determined by now (either from config, CLI override, or still not set)
    # The config["processing"]["download_path"] has been potentially set by args.download_path already.
    current_download_path_setting = config.get("processing", {}).get("download_path")

    is_path_auto_generated = (
        False  # Flag to track if we're using an auto-generated path
    )
    if not current_download_path_setting:
        # Path not set by config or CLI, so use system temp
        temp_subdir = os.path.join(
            tempfile.gettempdir(), f"absrefined_cli_{os.getpid()}"
        )
        config["processing"]["download_path"] = (
            temp_subdir  # Update config with the path being used
        )
        logging.debug(f"Using system temp directory for downloads: {temp_subdir}")
        atexit.register(_cleanup_temp_files, temp_subdir)  # Always clean up system temp
        is_path_auto_generated = True
    else:
        # Path was set by config or CLI. Decide if it should be cleaned.
        # debug_files: (Optional, boolean) If true, preserves intermediate files... in the download path
        debug_preserve_path = config.get("logging", {}).get("debug_files", False)

        # DEBUG log to help diagnose the issue
        logging.debug(
            f"Debug files preservation setting from config: {debug_preserve_path}"
        )

        if debug_preserve_path is True:  # Explicitly check for True
            logging.debug(
                f"Preserving user-specified download path due to 'debug_files=true': {current_download_path_setting}"
            )
            # No atexit registration for cleanup, path will be preserved
        else:
            logging.debug(
                f"Registering user-specified download path for cleanup: {current_download_path_setting}"
            )
            atexit.register(
                _cleanup_temp_files, current_download_path_setting
            )  # Clean up user-specified path

    # Ensure the download directory exists, whether it will be cleaned or preserved
    os.makedirs(config["processing"]["download_path"], exist_ok=True)

    if args.model:  # Overrides refiner model
        if "refiner" not in config:
            config["refiner"] = {}
        config["refiner"]["model_name"] = args.model
        logging.debug(f"Overriding refiner model with: {args.model}")
    if args.window:  # Overrides processing window
        if "processing" not in config:
            config["processing"] = {}
        config["processing"]["search_window_seconds"] = args.window
        logging.debug(f"Overriding search window with: {args.window} seconds")

    # --- Setup Logging ---
    log_level_config = config.get("logging", {}).get("level", "INFO").upper()

    log_level = logging.getLevelNamesMapping()[log_level_config]

    # CLI flags override config level
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO

    # Reset logging configuration to apply the determined log level
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        # filename='app.log',
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] %(name)-25s: %(message)s",  # Wider name field
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Re-get logger in case basicConfig was called after initial info log
    logger = logging.getLogger(__name__)
    logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")
    if args.debug:
        logger.debug("Debug logging enabled.")

    # --- Extract Item ID and Initialize Client/Tool ---
    item_id = None
    item_specifier = args.item_specifier

    item_id = extract_item_id_from_url(item_specifier)

    if item_id:
        logger.debug(f"Extracted Item ID '{item_id}' using utility function.")
    else:
        logger.error(
            f"Could not parse Item ID from specifier: '{item_specifier}' using utility function."
        )
        logger.error(
            "Please provide a valid Item ID or the full Audiobookshelf Item URL (e.g., .../item/<item_id> or just <item_id>)."
        )
        return 1

    if not item_id:  # This check might be redundant if the above 'else' handles it, but kept for safety
        logger.error("Failed to determine Item ID.")
        return 1

    try:
        client = AudiobookshelfClient(config=config)
        tool = ChapterRefinementTool(config=config, client=client)
        # Progress callback for CLI could be implemented here if needed (e.g., update tqdm bar)
    except KeyError as e:
        logger.error(
            f"Failed to initialize components due to missing configuration: {e}"
        )
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.exception("Initialization error details:")
        return 1

    # --- Handle --just-download ---
    if args.just_download:
        logger.info(f"Ensuring audio for item {item_id} is downloaded (if needed)...")
        # Use the tool's internal helper which relies on the client and config
        audio_path = tool._ensure_full_audio_downloaded(item_id)
        if audio_path and os.path.exists(audio_path):
            logger.info(f"Audio file is available at: {audio_path}")
            logger.info("Exiting due to --just-download flag.")
            return 0
        else:
            logger.error(f"Failed to download or locate audio file for item {item_id}.")
            return 1

    # --- Determine Parameters for Processing ---
    # Use CLI override if provided, otherwise fall back to config value
    search_window_used = (
        args.window
        if args.window is not None
        else config.get("processing", {}).get("search_window_seconds", 60)
    )
    model_override_used = args.model  # This will be None if not provided via CLI

    logger.info(f"Starting refinement process for item: {item_id}")
    logger.debug(f"Using search window: {search_window_used}s")
    if model_override_used:
        logger.debug(f"Using model override: {model_override_used}")
    else:
        logger.debug(
            f"Using model from config: {tool.refiner.default_model_name}"
        )  # Access default from initialized refiner
    logger.debug(f"Dry run: {args.dry_run}")

    # --- Process Item ---
    result = tool.process_item(
        item_id=item_id,
        search_window_seconds=search_window_used,
        model_name_override=model_override_used,
        dry_run=args.dry_run,  # Pass dry_run status (though tool doesn't update server directly)
    )

    # --- Process CLI Results ---
    logger.info("\n--- Refinement Results ---")
    if result.get("error"):
        logger.error(f"Processing failed: {result['error']}")
        return 1

    chapter_details = result.get("chapter_details")
    if not chapter_details:
        logger.warning("Processing finished, but no chapter details were returned.")
        return 0  # Not necessarily an error, maybe no chapters

    total_chapters = result.get("total_chapters", len(chapter_details))
    logger.info(f"Processed {total_chapters} chapters.")

    changes_to_confirm = []
    updates_for_server = []
    significant_change_threshold = config.get("processing", {}).get(
        "significant_change_threshold", 0.5
    )  # seconds (CLI threshold for reporting/updating)

    print("\n=== Chapter Comparison ===")
    print(
        "Idx | Title                           | Original Time | Refined Time  | Diff (s) | Status"
    )
    print(
        "----|---------------------------------|---------------|---------------|----------|---------"
    )

    cli_refined_count = 0
    for i, detail in enumerate(chapter_details):
        # Skip the first chapter (index 0) as it must always have start time of 0
        if i == 0:
            continue

        original_start = detail.get("original_start")
        refined_start = detail.get("refined_start")
        chapter_title = detail.get("title", f"Chapter {i + 1}")
        chapter_id = detail.get("id")

        # Format for display
        title_disp = (
            (chapter_title[:29] + "...")
            if len(chapter_title) > 32
            else chapter_title.ljust(32)
        )
        orig_disp = (
            format_timestamp(original_start)
            if original_start is not None
            else "--:--:--.---"
        )
        ref_disp = (
            format_timestamp(refined_start)
            if refined_start is not None
            else "--:--:--.---"
        )
        diff_disp = "N/A"
        status_disp = "No Refinement"

        if original_start is not None and refined_start is not None:
            time_diff = refined_start - original_start
            diff_disp = f"{time_diff:+.3f}"
            # Check for significant change (and ignore first chapter for automatic updates)
            if abs(time_diff) > significant_change_threshold:
                status_disp = "CHANGED"
                # Removed check for i > 0 since we now skip i == 0 already
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
                updates_for_server.append(
                    {"id": chapter_id, "start": max(0.0, refined_start)}
                )  # Ensure start time >= 0
            else:
                status_disp = "No Change"
        elif refined_start is None:
            status_disp = "Refine Failed"

        print(
            f"{i:<3d} | {title_disp} | {orig_disp:>13} | {ref_disp:>13} | {diff_disp:>8} | {status_disp}"
        )

    logger.info(
        f"\nFound {cli_refined_count} chapters with significant changes (>{significant_change_threshold}s, excluding first chapter)."
    )

    # --- Update Server (if applicable) ---
    if not args.dry_run and updates_for_server:
        logger.info("\nPreparing to update server...")
        confirm = args.yes  # Auto-confirm if --yes flag is set
        if not confirm:
            try:
                user_input = input(
                    f"Update {len(updates_for_server)} chapter start times on the server {config['audiobookshelf']['host']}? (yes/N): "
                ).lower()
                if user_input == "yes":
                    confirm = True
            except EOFError:  # Handle non-interactive environments
                logger.warning(
                    "Non-interactive environment detected, cannot confirm. Use --yes to force update or --dry-run to skip."
                )
                confirm = False

        if confirm:
            logger.info("Proceeding with server update...")
            try:
                # Use the initialized client (which uses config for auth)
                success = client.update_chapters_start_time(item_id, updates_for_server)
                if success:
                    logger.info(
                        f"Successfully updated {len(updates_for_server)} chapters on the server."
                    )
                else:
                    logger.error(
                        "Server update failed. Check client logs or server API response."
                    )
                    return 1  # Indicate failure
            except Exception as e:
                logger.error(f"Error occurred during server update: {e}")
                logger.exception("Update error details:")
                return 1
        else:
            logger.info(
                "Server update cancelled by user or non-interactive environment."
            )
    elif args.dry_run:
        logger.info("\nDry run enabled. No changes were pushed to the server.")
    elif not updates_for_server:
        logger.info("\nNo significant chapter changes found to update on the server.")

    # --- Display Usage and Cost Information ---
    refinement_usage = result.get("refinement_usage")
    transcription_usage = result.get("transcription_usage")

    # Access cost configuration directly from the loaded 'config' dictionary
    llm_prompt_cost_config = config.get("costs", {}).get(
        "llm_refinement_cost_per_million_prompt_tokens", 0.0
    )
    llm_completion_cost_config = config.get("costs", {}).get(
        "llm_refinement_cost_per_million_completion_tokens", 0.0
    )
    transcription_cost_config = config.get("costs", {}).get(
        "audio_transcription_cost_per_minute", 0.0
    )

    if refinement_usage or transcription_usage:
        logger.info("\n--- Usage & Cost Estimates ---")
        if refinement_usage:
            logger.info("Refinement (LLM):")
            logger.info(
                f"  Prompt Tokens:     {refinement_usage.get('prompt_tokens', 0)}"
            )
            logger.info(
                f"  Completion Tokens: {refinement_usage.get('completion_tokens', 0)}"
            )
            logger.info(
                f"  Total Tokens:      {refinement_usage.get('total_tokens', 0)}"
            )
            if llm_prompt_cost_config > 0 or llm_completion_cost_config > 0:
                logger.info(
                    f"  Estimated Cost:    ${refinement_usage.get('estimated_cost', 0.0):.4f}"
                )
            else:
                logger.info(
                    f"  Estimated Cost:    Not calculated (costs not configured or are zero)"
                )

        if transcription_usage:
            logger.info("Transcription (Audio):")
            logger.info(
                f"  Total Duration:    {transcription_usage.get('duration_seconds', 0.0):.2f}s ({transcription_usage.get('duration_minutes', 0.0):.2f} min)"
            )
            if transcription_cost_config > 0:
                logger.info(
                    f"  Estimated Cost:    ${transcription_usage.get('estimated_cost', 0.0):.4f}"
                )
            else:
                logger.info(
                    f"  Estimated Cost:    Not calculated (cost not configured or is zero)"
                )

    logger.info("Refinement process complete.")
    return 0  # Indicate success


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        logging.exception(f"An unexpected error occurred in main execution: {e}")
        exit(1)
