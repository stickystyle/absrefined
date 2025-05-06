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
        debug: bool = False,
        progress_callback=None,
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
            debug (bool): Whether to preserve audio chunks and transcripts for debugging
            progress_callback (callable, optional): Function to call with progress updates (percent, message).
                                                    Defaults to None.
        """
        self.abs_client = abs_client
        self.transcriber = transcriber
        self.refiner = refiner
        self.verbose = verbose
        self.temp_dir = temp_dir
        self.dry_run = dry_run
        self.debug = debug
        self.progress_callback = progress_callback
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
        Modified for GUI: Returns detailed chapter info and uses progress callback.

        Args:
            item_id (str): ID of the item to process

        Returns:
            Dict: Result dictionary with keys:
                  'error': str | None,
                  'item_id': str,
                  'total_chapters': int,
                  'refined_chapters': int, # Count of chapters with potential changes
                  'audio_file_path': str | None,
                  'chapter_details': List[Dict] | None # List of {'id', 'title', 'original_start', 'refined_start'}
        """
        # Initialize the result structure expected by the GUI
        result = {
            "item_id": item_id,
            "total_chapters": 0,
            "refined_chapters": 0,
            "audio_file_path": None,
            "chapter_details": None,
            "error": None,
        }

        # Helper function for progress updates
        def _update_progress(percent: int, message: str):
            if self.progress_callback:
                try:
                    self.progress_callback(percent, message)
                except Exception as cb_err:
                    self.logger.warning(f"Progress callback failed: {cb_err}")
            self.logger.info(f"Progress: {percent}% - {message}")

        _update_progress(0, "Starting process...")

        if not item_id:
            self.logger.error("No item ID provided for processing.")
            result["error"] = "No item ID provided."
            return result

        try:
            # --- Get Original Chapters --- #
            _update_progress(5, "Fetching chapter information...")
            original_chapters = self.abs_client.get_item_chapters(item_id)

            if not original_chapters:
                self.logger.warning(
                    f"No chapters found for item {item_id}. Nothing to process."
                )
                result["error"] = "No chapters found on server."
                _update_progress(100, "Failed: No chapters found.")
                return result

            result["total_chapters"] = len(original_chapters)
            self.logger.info(
                f"Found {len(original_chapters)} original chapters for item {item_id}."
            )

            # --- Ensure Audio Downloaded --- #
            _update_progress(10, "Checking/Downloading audio file...")
            full_audio_path = self._ensure_full_audio_downloaded(item_id)
            if not full_audio_path:
                result["error"] = "Failed to download full audio file."
                _update_progress(100, "Failed: Audio download error.")
                return result
            result["audio_file_path"] = full_audio_path
            _update_progress(15, "Audio file ready.")

            # --- Get Audio Duration --- #
            _update_progress(17, "Getting audio duration...")
            audio_duration = self._get_audio_duration(full_audio_path)
            if audio_duration is None:
                self.logger.warning(
                    f"Could not determine audio duration for {full_audio_path}. Proceeding without duration check."
                )
            _update_progress(20, "Starting chapter processing...")

            # --- Process Chapters --- #
            # This method now needs to accept the callback and pass it down/use it internally
            processed_chapters_list = self.process_chapters(
                item_id, original_chapters, full_audio_path, audio_duration
            )

            # process_chapters should return None only on critical failure
            if processed_chapters_list is None:
                self.logger.error("Chapter processing failed critically.")
                result["error"] = "Chapter processing failed (check logs for details)."
                _update_progress(100, "Failed: Chapter processing error.")
                return result

            # --- Finalize Results for GUI --- #
            _update_progress(95, "Finalizing results...")
            final_chapter_details = []
            refined_count = 0
            significant_change_threshold = 0.1 # Lower threshold for GUI display

            # Iterate through the results from process_chapters
            for processed_chapter in processed_chapters_list:
                 # Extract times for comparison
                 original_start = processed_chapter.get("original_start")
                 refined_start = processed_chapter.get("refined_start")

                 # Count as refined if time exists and differs significantly
                 if (
                     original_start is not None
                     and refined_start is not None
                     and abs(refined_start - original_start) > significant_change_threshold
                 ):
                      refined_count += 1

                 # Append the *entire* dictionary (including chunk_path, window_start etc.)
                 # to the list that will be sent to the GUI.
                 final_chapter_details.append(processed_chapter)

            result["chapter_details"] = final_chapter_details
            result["refined_chapters"] = refined_count # Store the count of potentially changed chapters
            _update_progress(100, "Processing complete.")

        except EnvironmentError as e:  # Catch ffmpeg/ffprobe check failure
            self.logger.error(f"Environment error: {e}")
            result["error"] = str(e)
            _update_progress(100, f"Failed: {e}")
        except Exception as e:
            self.logger.exception(
                f"An unexpected error occurred during processing item {item_id}: {e}"
            )
            result["error"] = f"Unexpected error: {e}"
            _update_progress(100, "Failed: Unexpected error.")

        return result

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
        Modified for GUI: Returns detailed results including original and refined start times (as floats).
                      Uses progress callback for updates.

        Args:
            item_id (str): ID of the item to process.
            original_chapters (List[Dict]): The original chapters fetched from the server.
            full_audio_path (str): Path to the downloaded full audio file.
            audio_duration (float | None): Total duration of the audio, if known.

        Returns:
            List[Dict] | None: A list of chapter detail dictionaries, each containing:
                              {'id': str, 'title': str, 'original_start': float, 'refined_start': float | None}
                              Returns None on critical failure.
        """
        chapter_details_list = []
        window_half_size = self.refiner.window_size  # Get window size from refiner
        total_chapters = len(original_chapters)
        base_progress = 20 # Start progress reporting after initial steps
        progress_range = 95 - base_progress # Progress percentage allocated to this loop

        self.logger.info(
            f"Processing {total_chapters} chapters using ±{window_half_size}s window extraction..."
        )

        # Helper for progress within this method
        def _update_chapter_progress(index: int, message: str):
            if self.progress_callback:
                percent = base_progress + int(((index + 1) / total_chapters) * progress_range)
                try:
                    self.progress_callback(percent, f"Ch.{index + 1}/{total_chapters}: {message}")
                except Exception as cb_err:
                    self.logger.warning(f"Progress callback failed: {cb_err}")


        # Use tqdm for console progress if verbose, but rely on callback for GUI
        # chapter_progress = tqdm(..., disable=not self.verbose)

        # Get base name and extension for temp files
        # audio_base, audio_ext = os.path.splitext(os.path.basename(full_audio_path))

        for i, chapter in enumerate(original_chapters):
            temp_chunk_path = None  # Ensure cleanup path is defined
            _update_chapter_progress(i, "Starting...")
            try:
                chapter_title = chapter.get("title", f"Chapter {i + 1}")
                chapter_id = chapter.get("id")
                if not chapter_id:
                    self.logger.error(f"Chapter {i+1} ('{chapter_title}') missing ID. Cannot process.")
                    # How to handle? Skip? Create dummy result?
                    # Let's create a dummy result indicating failure for this chapter
                    chapter_details_list.append({
                        "id": f"missing_id_{i}",
                        "title": chapter_title,
                        "original_start": parse_timestamp(chapter.get("start", 0)), # Best effort parse
                        "refined_start": None
                    })
                    continue

                try:
                    original_start_time = parse_timestamp(chapter.get("start", 0))
                except ValueError:
                    self.logger.warning(
                        f"Could not parse timestamp '{chapter.get('start')}' for chapter '{chapter_title}'. Using 0.0 for original time."
                    )
                    original_start_time = 0.0

                # Initialize result for this chapter
                current_chapter_detail = {
                    "id": chapter_id,
                    "title": chapter_title,
                    "original_start": original_start_time,
                    "refined_start": None, # Default to None
                    "chunk_path": None, # Will be populated after extraction
                    "window_start": None, # Will be populated after calculation
                }

                # Skip refinement for the very first chapter (always 0)
                if i == 0:
                     self.logger.info("Skipping refinement for first chapter (start time is always 0.0).")
                     # Ensure original_start is 0.0 for the first chapter detail
                     current_chapter_detail["original_start"] = 0.0
                     chapter_details_list.append(current_chapter_detail)
                     continue

                self.logger.debug(
                    f"Processing chapter {i + 1}/{total_chapters}: '{chapter_title}' (Original time: {format_timestamp(original_start_time)})"
                )

                # --- Define Window --- #
                window_start = max(0.0, original_start_time - window_half_size)
                window_end = original_start_time + window_half_size
                if audio_duration is not None:
                    window_end = min(window_end, audio_duration)

                # Store window_start for playback reference
                current_chapter_detail["window_start"] = window_start

                if window_start >= window_end:
                    self.logger.warning(
                        f"Calculated window for '{chapter_title}' is invalid ({window_start:.3f}s >= {window_end:.3f}s). Skipping refinement."
                    )
                    chapter_details_list.append(current_chapter_detail)
                    continue

                # --- Extract Audio Segment --- #
                _update_chapter_progress(i, f"Extracting audio {window_start:.1f}s-{window_end:.1f}s...")
                self.logger.debug(
                    f"  Extracting audio window: {window_start:.3f}s - {window_end:.3f}s"
                )

                if self.debug: # In debug mode, create predictable filenames
                    safe_title = re.sub(r'[^\w\-_]', '_', chapter_title)
                    chunk_filename = f"{item_id}_chapter_{i+1}_{safe_title}_{window_start:.1f}s-{window_end:.1f}s.wav"
                    temp_chunk_path = os.path.join(self.temp_dir, chunk_filename)
                    # No need to add predictable names to cleanup, handled by directory cleanup
                    self.logger.info(f"Debug mode: Saving audio chunk to {temp_chunk_path}")
                else:
                    # Use tempfile for non-debug to guarantee uniqueness, but don't delete automatically
                    with tempfile.NamedTemporaryFile(
                        prefix=f"{item_id}_ch{i+1}_", suffix=".wav", dir=self.temp_dir, delete=False
                    ) as temp_f:
                        temp_chunk_path = temp_f.name
                        # Do NOT add temp_chunk_path to _temp_files_to_delete here, GUI cleans dir

                # Store chunk path for playback
                current_chapter_detail["chunk_path"] = temp_chunk_path

                # Use WAV for transcription compatibility with some engines
                # Also helps simpleaudio playback in GUI if we use the same chunk
                # Consider adding `-ac 1` (mono) and `-ar 16000` (sample rate) if needed by transcriber/refiner
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    full_audio_path,
                    "-ss",
                    str(window_start),
                    "-to",
                    str(window_end),
                    "-vn",
                    "-acodec", "pcm_s16le", # Output WAV format
                    "-y",
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
                    # Only log stderr if verbose or error indication
                    if result.stderr and (
                        self.verbose or "error" in result.stderr.lower() or "warning" in result.stderr.lower()
                    ):
                        self.logger.debug(f"  ffmpeg stderr:\n{result.stderr.strip()}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(
                        f"  Error extracting audio segment for chapter '{chapter_title}': {e}"
                    )
                    self.logger.error(f"  Command: {' '.join(e.cmd)}")
                    self.logger.error(f"  ffmpeg stderr: {e.stderr.strip() if e.stderr else '[No stderr]'}")
                    chapter_details_list.append(current_chapter_detail) # Keep original
                    continue
                except Exception as e:
                    self.logger.exception(
                        f"  Unexpected error during ffmpeg extraction for '{chapter_title}': {e}"
                    )
                    chapter_details_list.append(current_chapter_detail) # Keep original
                    continue

                # --- Transcribe Segment --- #
                _update_chapter_progress(i, "Transcribing audio segment...")
                self.logger.debug(f"  Transcribing segment: {temp_chunk_path}")
                chunk_segments = []
                try:
                    # Timestamps will be relative to the chunk's start (window_start).
                    chunk_segments = self.transcriber.transcribe_audio(
                        temp_chunk_path, write_to_file=False
                    )

                    # Adjust segment timestamps to be relative to the *original* audio file
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
                        # Log adjusted time range for debugging
                        if self.verbose:
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
                    # chunk_segments remains empty

                # --- Refine using Transcription --- #
                if chunk_segments:
                    _update_chapter_progress(i, "Refining timestamp with LLM...")
                    self.logger.debug(
                        "  Detecting chapter start within transcribed segment..."
                    )
                    detection_result = self.refiner.detect_chapter_start(
                        chunk_segments, # Pass the small, adjusted transcript
                        chapter_title,
                        original_start_time,
                    )
                    if detection_result:
                        new_timestamp = detection_result.get("timestamp")
                        try:
                            new_timestamp_float = max(0.0, float(new_timestamp))
                            # Basic sanity check: ensure refined time is within or very close to the extracted window
                            # Allow slightly more leeway than the window half-size
                            if not (
                                window_start - (window_half_size * 0.5)
                                <= new_timestamp_float
                                <= window_end + (window_half_size * 0.5)
                            ):
                                self.logger.warning(
                                    f"  Refined timestamp {new_timestamp_float:.3f}s is significantly outside the expected window ({window_start:.3f}s - {window_end:.3f}s) for '{chapter_title}'. Keeping original."
                                )
                                # Keep original (refined_start is still None)
                            else:
                                self.logger.info(
                                    f"  Refined timestamp for '{chapter_title}': {format_timestamp(new_timestamp_float)} (Original: {format_timestamp(original_start_time)})"
                                )
                                current_chapter_detail["refined_start"] = new_timestamp_float

                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"  Invalid refined timestamp '{new_timestamp}' received. Keeping original."
                            )
                    else:
                        self.logger.info(
                            f"  LLM refinement failed for '{chapter_title}'. Keeping original timestamp."
                        )
                else:
                     _update_chapter_progress(i, "Skipping refinement (transcription failed).")
                     self.logger.warning(
                        f"  Skipping LLM refinement for '{chapter_title}' due to transcription failure/empty result."
                     )

                chapter_details_list.append(current_chapter_detail)
                _update_chapter_progress(i, "Done.")

            finally:
                # --- Clean up temporary audio chunk --- #
                # REMOVED: Cleanup is handled by GUI atexit handler cleaning the temp dir
                # if temp_chunk_path and os.path.exists(temp_chunk_path) and not self.debug:
                #     try:
                #         os.remove(temp_chunk_path)
                #         self.logger.debug(f"  Cleaned up temp chunk: {temp_chunk_path}")
                #     except OSError as e:
                #         self.logger.warning(
                #             f"  Could not remove temporary chunk file {temp_chunk_path}: {e}"
                #         )
                pass # Keep the finally block structure just in case

        self.logger.info("Finished processing all chapters.")
        return chapter_details_list
