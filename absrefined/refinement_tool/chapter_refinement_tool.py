import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple

from tqdm import tqdm

from absrefined.client.abs_client import AudiobookshelfClient
from absrefined.refiner.chapter_refiner import ChapterRefiner
from absrefined.transcriber.audio_transcriber import AudioTranscriber
from absrefined.utils.timestamp import format_timestamp, parse_timestamp


class ChapterRefinementTool:
    """Tool to orchestrate the chapter refinement process."""

    def __init__(
        self,
        abs_client: AudiobookshelfClient,
        transcriber: AudioTranscriber,
        refiner: ChapterRefiner,
        verbose: bool = False,
        temp_dir: str = "temp",
        dry_run: bool = False,
    ):
        """
        Initialize the chapter refinement tool.

        Args:
            abs_client (AudiobookshelfClient): Client for interacting with Audiobookshelf
            transcriber (AudioTranscriber): Transcriber for audio segments
            refiner (ChapterRefiner): Refiner for chapter markers
            verbose (bool): Whether to print verbose output
            temp_dir (str): Directory for temporary files
            dry_run (bool): Whether to run in dry-run mode (no updates)
        """
        self.abs_client = abs_client
        self.transcriber = transcriber
        self.refiner = refiner
        self.verbose = verbose
        self.temp_dir = temp_dir
        self.dry_run = dry_run
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        self._check_ffmpeg_ffprobe()

    def _check_ffmpeg_ffprobe(self):
        """Checks if ffmpeg and ffprobe are available in PATH."""
        for cmd in ["ffmpeg", "ffprobe"]:
            if shutil.which(cmd) is None:
                self.logger.error(
                    f"Error: '{cmd}' not found. Please ensure ffmpeg is installed and in your system PATH."
                )
                # Consider raising an exception or setting a failure state
                raise EnvironmentError(f"{cmd} not found. Cannot proceed.")
        self.logger.debug("ffmpeg and ffprobe found.")

    def extract_item_id_from_url(self, url: str) -> str:
        """
        Extract item ID from Audiobookshelf URL.

        Args:
            url (str): URL to extract ID from

        Returns:
            str: Extracted item ID or empty string on failure
        """
        # Extract item ID from URL
        match = re.search(r"/item/([a-zA-Z0-9\-]+)", url)

        if not match:
            self.logger.error(f"Failed to extract item ID from URL: {url}")
            return ""

        item_id = match.group(1)

        self.logger.debug(f"Extracted item ID: {item_id}")

        return item_id

    def _ensure_full_audio_downloaded(self, item_id: str) -> str | None:
        """Downloads the full audio file if it doesn't exist, returns the path."""
        # Try common audio extensions first
        found_path = None
        for ext in ["m4a", "mp3", "ogg", "opus", "flac", "wav"]:
            potential_path = os.path.join(self.temp_dir, f"{item_id}_full.{ext}")
            if os.path.exists(potential_path):
                found_path = potential_path
                self.logger.info(f"Using existing full audio file: {found_path}")
                break

        if found_path:
            return found_path

        # If not found, define the default download path and download
        download_path = os.path.join(self.temp_dir, f"{item_id}_full.m4a")
        self.logger.info(f"Downloading full audio file to {download_path}...")
        downloaded_path = self.abs_client.download_audio_file(item_id, download_path)
        if not downloaded_path:
            self.logger.error(f"Failed to download full audio file for item {item_id}.")
            return None

        self.logger.info(f"Download complete: {downloaded_path}")
        return downloaded_path

    def process_item(self, item_id: str) -> Dict:
        """
        Process an item to refine its chapter markers using small window extraction.

        Args:
            item_id (str): ID of the item to process

        Returns:
            Dict: Result of processing
        """
        result = {
            "item_id": item_id,
            "total_chapters": 0,
            "refined_chapters": 0,
            "updated_on_server": False,
            "error": None,
        }

        if not item_id:
            self.logger.error("No item ID provided for processing.")
            result["error"] = "No item ID provided."
            return result

        try:
            # Get original chapters from the server
            original_chapters = self.abs_client.get_item_chapters(item_id)

            if not original_chapters:
                self.logger.warning(
                    f"No chapters found for item {item_id}. Nothing to process."
                )
                result["error"] = "No chapters found on server."
                return result

            result["total_chapters"] = len(original_chapters)
            self.logger.info(
                f"Found {len(original_chapters)} original chapters for item {item_id}."
            )

            # Ensure full audio is downloaded before processing chapters
            full_audio_path = self._ensure_full_audio_downloaded(item_id)
            if not full_audio_path:
                result["error"] = "Failed to download full audio file."
                return result

            # Get the actual audio duration once
            audio_duration = self._get_audio_duration(full_audio_path)
            if audio_duration is None:
                self.logger.warning(
                    f"Could not determine audio duration for {full_audio_path}. Proceeding without duration check."
                )
                # Continue processing, but end times might be less accurate

            # Process the chapters to refine them
            refined_chapters = self.process_chapters(
                item_id, original_chapters, full_audio_path, audio_duration
            )

            if (
                refined_chapters is None
            ):  # Check for None indicating critical error during chapter processing
                self.logger.error("Chapter processing failed critically.")
                result["error"] = "Chapter processing failed (check logs for details)."
                return result
            elif not refined_chapters:  # Check for empty list
                self.logger.warning(
                    "No refined chapters were generated (potentially no changes needed or errors occurred)."
                )
                # Continue to comparison, which handles this
            else:
                self.logger.info(
                    f"Generated {len(refined_chapters)} potential refined chapters."
                )

            # Compare original and refined chapters
            updated, refined_count = self.compare_and_update(
                item_id, original_chapters, refined_chapters, self.dry_run
            )
            result["refined_chapters"] = refined_count
            result["updated_on_server"] = updated

        except EnvironmentError as e:  # Catch ffmpeg/ffprobe check failure
            self.logger.error(f"Environment error: {e}")
            result["error"] = str(e)
        except Exception as e:
            self.logger.exception(
                f"An unexpected error occurred during processing item {item_id}: {e}"
            )
            result["error"] = f"Unexpected error: {e}"

        return result

    def compare_and_update(
        self,
        item_id: str,
        original_chapters: List[Dict],
        refined_chapters: List[Dict],
        dry_run: bool = False,
    ) -> tuple[bool, int]:  # Return tuple: (updated_on_server, refined_count)
        """
        Compare original and refined chapters, ask for confirmation, and update if changes were made.
        (Largely unchanged from previous version, ensures float handling is robust)

        Args:
            item_id (str): ID of the library item
            original_chapters (List[Dict]): Original chapters
            refined_chapters (List[Dict]): Refined chapters (may contain 'refined' key and float starts)
            dry_run (bool): Whether to perform a dry run (no actual updates)

        Returns:
            tuple[bool, int]: (Whether update was successful/attempted, count of refined chapters)
        """
        changed_chapters_info = []
        significant_change_threshold = 0.5  # seconds
        refined_count = 0

        # Ensure we have comparable lists
        if (
            not original_chapters
            or not refined_chapters
            or len(original_chapters) != len(refined_chapters)
        ):
            self.logger.warning(
                "Chapter lists are missing or mismatched in length. Cannot compare."
            )
            # Return 0 refined count if lists mismatch
            return False, 0

        self.logger.info(
            f"\nComparing {len(original_chapters)} original and refined chapter timestamps..."
        )

        final_chapters_for_update = []

        for i, (orig, refined) in enumerate(zip(original_chapters, refined_chapters)):
            final_chapter = refined.copy()

            try:
                orig_time = parse_timestamp(orig.get("start", 0))
            except ValueError:
                self.logger.warning(
                    f"Could not parse original start time '{orig.get('start')}' for chapter index {i}. Using 0.0."
                )
                orig_time = 0.0

            # Refined time should be float from process_chapters
            refined_start_val = refined.get("start", 0)
            if isinstance(refined_start_val, (float, int)):
                refined_time = float(refined_start_val)
            elif isinstance(
                refined_start_val, str
            ):  # Handle potential string format if passed directly
                try:
                    refined_time = parse_timestamp(refined_start_val)
                except ValueError:
                    self.logger.warning(
                        f"Could not parse refined start time string '{refined_start_val}' for chapter index {i}. Using original time {orig_time:.3f}s."
                    )
                    refined_time = orig_time
            else:
                self.logger.warning(
                    f"Unexpected type for refined start time '{refined_start_val}' (type: {type(refined_start_val)}). Using original time {orig_time:.3f}s."
                )
                refined_time = orig_time

            chapter_title = orig.get("title", f"Chapter {i + 1}")

            # Always ensure the first chapter starts at 0
            if i == 0:
                refined_time = 0.0
                final_chapter["start"] = 0.0  # Update the final dict as well

            is_refined = refined.get("refined", False)
            if is_refined:
                refined_count += 1

            time_diff = refined_time - orig_time

            if self.verbose:
                self.logger.debug(f"Chapter '{chapter_title}' (Index {i}):")
                self.logger.debug(f"  Original: {format_timestamp(orig_time)}")
                self.logger.debug(
                    f"  Refined:  {format_timestamp(refined_time)} ({'Marked as refined' if is_refined else 'Kept original'})"
                )
                self.logger.debug(f"  Difference: {time_diff:+.3f}s")

            if abs(time_diff) > significant_change_threshold:
                self.logger.info(
                    f"  Significant change detected for Chapter '{chapter_title}' (Diff: {time_diff:+.3f}s)"
                )
                changed_chapters_info.append(
                    {
                        "index": i,
                        "title": chapter_title,
                        "original": orig_time,
                        "refined": refined_time,
                        "diff": time_diff,
                    }
                )

            # Remove the internal 'refined' flag before potential update
            if "refined" in final_chapter:
                del final_chapter["refined"]
            # Keep start time as float until boundaries are fixed, then format
            final_chapter["start"] = refined_time  # Keep as float for now
            final_chapters_for_update.append(final_chapter)

        # Update chapters on the server if changes were made
        if changed_chapters_info:
            self.logger.info(
                f"\nFound {len(changed_chapters_info)} chapters with changes > {significant_change_threshold}s."
            )
            self.logger.info("\nPreview of significant changes:")
            for change in changed_chapters_info:
                diff_str = f"({change['diff']:+.3f}s)"
                self.logger.info(
                    f"  Chapter '{change['title']}': {format_timestamp(change['original'])} -> {format_timestamp(change['refined'])} {diff_str}"
                )

            # Fix chapter end times based on the *next* chapter's *refined* start time (which are floats)
            self._fix_chapter_boundaries(final_chapters_for_update)
            self.logger.debug(
                "Applied end time adjustments based on refined start times."
            )

            # NOW format the start times to strings for display/update
            for chapter in final_chapters_for_update:
                if isinstance(chapter["start"], (float, int)):
                    chapter["start"] = format_timestamp(chapter["start"])
                # API expects end times as floats/ints, not strings
                # if "end" in chapter and isinstance(chapter["end"], str):
                #      chapter["end"] = parse_timestamp(chapter["end"])

            if not dry_run:
                prompt = "\nDo you want to apply these changes to the server? (y/n): "
                try:
                    confirm = input(prompt).strip().lower()
                except EOFError:
                    self.logger.warning(
                        "Non-interactive environment detected, cancelling update."
                    )
                    confirm = "n"

                if confirm == "y":
                    self.logger.info("Attempting to update chapters on the server...")
                    # Ensure end times are numbers if they exist, API client expects numbers
                    for chapter in final_chapters_for_update:
                        if "end" in chapter and not isinstance(
                            chapter["end"], (int, float)
                        ):
                            try:
                                chapter["end"] = parse_timestamp(chapter["end"])
                            except ValueError:
                                self.logger.error(
                                    f"Could not parse end time {chapter['end']} before update, setting to None."
                                )
                                chapter["end"] = None  # Or handle differently?

                    success = self.abs_client.update_item_chapters(
                        item_id, final_chapters_for_update
                    )
                    if success:
                        self.logger.info("Server update successful.")
                        return True, refined_count
                    else:
                        self.logger.error("Server update failed.")
                        return False, refined_count
                else:
                    self.logger.info("Update cancelled by user.")
                    return False, refined_count
            else:
                self.logger.info(
                    "\nDRY RUN: Changes detected, but not updating server."
                )
                return True, refined_count  # Indicate update would have happened
        else:
            self.logger.info("\nNo significant changes found requiring server update.")
            return False, refined_count  # No update attempted

    def _fix_chapter_boundaries(self, chapters: List[Dict]) -> None:
        """
        Fix chapter boundaries by setting each chapter's end time to the start time
        of the next chapter. Assumes start times are floats.

        Args:
            chapters (List[Dict]): Chapters to fix boundaries for. Modified in-place.
        """
        if not chapters or len(chapters) < 2:
            return

        # Start times should already be floats from process_chapters
        for i in range(len(chapters) - 1):
            next_start = chapters[i + 1]["start"]
            if isinstance(next_start, (float, int)):
                chapters[i]["end"] = float(next_start)
            else:
                # Handle error: next start wasn't a number as expected
                self.logger.error(
                    f"Cannot fix end time for chapter {i}: next chapter start '{next_start}' is not a number."
                )
                # Decide on fallback: remove end time? Keep original? For now, remove.
                if "end" in chapters[i]:
                    del chapters[i]["end"]

        # Handle last chapter end time? ABS might handle this if missing.
        if "end" not in chapters[-1]:
            self.logger.warning("Last chapter missing 'end' time after boundary fix.")
            # Optionally set based on duration if available, otherwise leave unset

    def _get_audio_duration(self, audio_path: str) -> float | None:
        """Get the duration of an audio file using ffprobe."""
        # Keep this helper method as it's useful for validating extraction ranges
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            self.logger.debug(f"Running ffprobe command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            duration = float(result.stdout.strip())
            self.logger.debug(
                f"Detected duration for {audio_path}: {duration:.2f} seconds"
            )
            return duration
        except FileNotFoundError:
            self.logger.error(
                "Error: ffprobe not found. Please ensure ffmpeg is installed and in your PATH."
            )
            return None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running ffprobe for {audio_path}: {e}")
            self.logger.error(
                f"ffprobe stderr: {e.stderr.strip() if e.stderr else '[No stderr]'}"
            )
            return None
        except ValueError as e:
            self.logger.error(f"Error parsing ffprobe output for {audio_path}: {e}")
            ffprobe_stdout = getattr(result, "stdout", "[ffprobe stdout not captured]")
            self.logger.error(f"ffprobe stdout: {ffprobe_stdout}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error getting audio duration for {audio_path}: {e}"
            )
            return None

    def process_chapters(
        self,
        item_id: str,
        original_chapters: List[Dict],
        full_audio_path: str,
        audio_duration: float | None,
    ) -> List[Dict] | None:
        """
        Process chapters using small window extraction and transcription.

        Args:
            item_id (str): ID of the item to process.
            original_chapters (List[Dict]): The original chapters fetched from the server.
            full_audio_path (str): Path to the downloaded full audio file.
            audio_duration (float | None): Total duration of the audio, if known.

        Returns:
            List[Dict] | None: Refined chapters list with float start times, or None on critical failure.
        """
        updated_chapters = []
        window_half_size = self.refiner.window_size  # Get window size from refiner
        self.logger.info(
            f"Processing chapters using ±{window_half_size}s window extraction..."
        )

        # Use tqdm for progress indication
        chapter_progress = tqdm(
            original_chapters,
            desc="Processing Chapters",
            unit="chapter",
            disable=not self.verbose,
        )

        # Get base name and extension for temp files
        audio_base, audio_ext = os.path.splitext(os.path.basename(full_audio_path))

        for i, chapter in enumerate(chapter_progress):
            temp_chunk_path = None  # Ensure cleanup path is defined
            try:
                chapter_title = chapter.get("title", f"Chapter {i + 1}")
                try:
                    chapter_time = parse_timestamp(chapter.get("start", 0))
                except ValueError:
                    self.logger.warning(
                        f"Could not parse timestamp '{chapter.get('start')}' for chapter '{chapter_title}'. Skipping refinement for this chapter."
                    )
                    refined_chapter = chapter.copy()
                    refined_chapter["start"] = (
                        0.0 if i == 0 else parse_timestamp(chapter.get("start", 0))
                    )  # Keep original as float
                    refined_chapter["refined"] = False
                    updated_chapters.append(refined_chapter)
                    continue  # Move to next chapter

                if self.verbose:
                    chapter_progress.set_description(f"Processing '{chapter_title}'")
                self.logger.debug(
                    f"Processing chapter {i + 1}/{len(original_chapters)}: '{chapter_title}' (Original time: {format_timestamp(chapter_time)})"
                )

                # Define window boundaries
                window_start = max(0.0, chapter_time - window_half_size)
                window_end = chapter_time + window_half_size
                # Clamp window_end to audio duration if known
                if audio_duration is not None:
                    window_end = min(window_end, audio_duration)
                # Ensure window_start is strictly less than window_end
                if window_start >= window_end:
                    self.logger.warning(
                        f"Calculated window for '{chapter_title}' is invalid (start >= end). Skipping segment extraction."
                    )
                    # Decide how to handle - keep original? Assume refinement failed?
                    refined_chapter = chapter.copy()
                    refined_chapter["start"] = chapter_time  # Keep original as float
                    refined_chapter["refined"] = False
                    updated_chapters.append(refined_chapter)
                    continue

                self.logger.debug(
                    f"  Extracting audio window: {window_start:.3f}s - {window_end:.3f}s"
                )

                # --- Extract Audio Segment using ffmpeg --- #
                # Create a temporary file for the chunk securely, force WAV format
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", dir=self.temp_dir, delete=False
                ) as temp_f:
                    temp_chunk_path = temp_f.name

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    full_audio_path,
                    "-ss",
                    str(window_start),
                    "-to",
                    str(window_end),
                    "-vn",  # No video
                    # "-c:a", "pcm_s16le", # Explicitly output WAV
                    "-y",  # Overwrite temporary file if needed
                    temp_chunk_path,
                ]
                self.logger.debug(f"  Running ffmpeg: {' '.join(ffmpeg_cmd)}")
                try:
                    stdout_level = subprocess.DEVNULL if not self.verbose else None
                    stderr_capture = subprocess.PIPE
                    result = subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        stdout=stdout_level,
                        stderr=stderr_capture,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    if result.stderr and (
                        self.verbose or "error" in result.stderr.lower()
                    ):
                        self.logger.debug(f"  ffmpeg stderr:\n{result.stderr.strip()}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(
                        f"  Error extracting audio segment for chapter '{chapter_title}': {e}"
                    )
                    self.logger.error(f"  Command: {' '.join(e.cmd)}")
                    self.logger.error(
                        f"  ffmpeg stderr: {e.stderr.strip() if e.stderr else '[No stderr]'}"
                    )
                    # Keep original chapter if extraction fails
                    refined_chapter = chapter.copy()
                    refined_chapter["start"] = chapter_time  # Keep original as float
                    refined_chapter["refined"] = False
                    updated_chapters.append(refined_chapter)
                    continue  # Proceed to next chapter
                except Exception as e:  # Catch other potential ffmpeg errors
                    self.logger.exception(
                        f"  Unexpected error during ffmpeg extraction for '{chapter_title}': {e}"
                    )
                    refined_chapter = chapter.copy()
                    refined_chapter["start"] = chapter_time  # Keep original as float
                    refined_chapter["refined"] = False
                    updated_chapters.append(refined_chapter)
                    continue

                # --- Transcribe the Small Audio Segment --- #
                self.logger.debug(f"  Transcribing segment: {temp_chunk_path}")
                try:
                    # Transcribe the extracted chunk. Timestamps will be relative to the chunk's start (window_start).
                    # No output file needed.
                    chunk_segments = self.transcriber.transcribe_audio(
                        temp_chunk_path, write_to_file=False
                    )

                    # IMPORTANT: Adjust segment timestamps to be relative to the *original* audio file
                    for seg in chunk_segments:
                        seg["start"] += window_start
                        seg["end"] += window_start
                        if "words" in seg:
                            for word in seg["words"]:
                                word["start"] += window_start
                                word["end"] += window_start

                    if chunk_segments:
                        self.logger.debug(
                            f"  Transcription successful, found {len(chunk_segments)} segments."
                        )
                        first_seg_time = chunk_segments[0]["start"]
                        last_seg_time = chunk_segments[-1]["end"]
                        self.logger.debug(
                            f"    Adjusted time range: {first_seg_time:.3f}s - {last_seg_time:.3f}s"
                        )
                    else:
                        self.logger.warning(
                            f"  Transcription of segment for '{chapter_title}' yielded no segments."
                        )

                except Exception as e:
                    self.logger.error(
                        f"  Error transcribing segment for chapter '{chapter_title}': {e}"
                    )
                    chunk_segments = []  # Treat as failed transcription

                # --- Refine using the Segment's Transcription --- #
                refined_chapter = chapter.copy()  # Start with original chapter info
                refined_chapter["start"] = chapter_time  # Ensure start is float
                refined_chapter["refined"] = False  # Default to not refined

                if chunk_segments:  # Only attempt refinement if transcription succeeded
                    self.logger.debug(
                        "  Detecting chapter start within transcribed segment..."
                    )
                    detection_result = self.refiner.detect_chapter_start(
                        chunk_segments,
                        chapter_title,
                        chapter_time,  # Pass the small, adjusted transcript
                    )
                    if detection_result:
                        new_timestamp = detection_result.get("timestamp")
                        try:
                            new_timestamp_float = max(0.0, float(new_timestamp))
                            # Basic sanity check: ensure refined time is within or very close to the extracted window
                            if not (
                                window_start - 1.0
                                <= new_timestamp_float
                                <= window_end + 1.0
                            ):
                                self.logger.warning(
                                    f"  Refined timestamp {new_timestamp_float:.3f}s is outside the expected window ({window_start:.3f}s - {window_end:.3f}s) for '{chapter_title}'. Keeping original."
                                )
                                new_timestamp_float = (
                                    chapter_time  # Revert to original if way off
                                )
                            else:
                                self.logger.info(
                                    f"  Refined timestamp for '{chapter_title}': {format_timestamp(new_timestamp_float)} (Original: {format_timestamp(chapter_time)})"
                                )
                                refined_chapter["start"] = new_timestamp_float
                                refined_chapter["refined"] = True
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"  Invalid refined timestamp '{new_timestamp}' received. Keeping original."
                            )
                            # Keep chapter_time (already set in refined_chapter)
                    else:
                        self.logger.info(
                            f"  LLM refinement failed for '{chapter_title}'. Keeping original timestamp."
                        )
                else:
                    self.logger.warning(
                        f"  Skipping LLM refinement for '{chapter_title}' due to transcription failure/empty result."
                    )

                updated_chapters.append(refined_chapter)

            finally:
                # --- Clean up temporary audio chunk --- #
                if temp_chunk_path and os.path.exists(temp_chunk_path):
                    try:
                        os.remove(temp_chunk_path)
                        self.logger.debug(
                            f"  Cleaned up temporary file: {temp_chunk_path}"
                        )
                    except OSError as e:
                        self.logger.error(
                            f"  Failed to clean up temp file {temp_chunk_path}: {e}"
                        )

        # Close progress bar
        if isinstance(chapter_progress, tqdm):
            chapter_progress.close()

        self.logger.info("Finished processing all chapters.")
        return updated_chapters
