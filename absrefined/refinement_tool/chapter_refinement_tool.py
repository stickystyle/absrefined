import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from absrefined.client.abs_client import AudiobookshelfClient
from absrefined.refiner.chapter_refiner import ChapterRefiner
from absrefined.transcriber.audio_transcriber import AudioTranscriber
from absrefined.utils.timestamp import format_timestamp, parse_timestamp


class ChapterRefinementTool:
    """Tool to orchestrate the chapter refinement process."""

    def __init__(
        self,
        config: Dict[str, Any],
        client: AudiobookshelfClient,
        progress_callback: Callable[[int, str], None] | None = None,
    ):
        """
        Initialize the chapter refinement tool.

        Args:
            config (Dict[str, Any]): The main configuration dictionary.
            client (AudiobookshelfClient): An initialized AudiobookshelfClient instance.
            progress_callback (callable, optional): Function to call with progress updates (percent, message).
        """
        self.config = config
        self.abs_client = client
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            self.transcriber = AudioTranscriber(config=self.config)
            self.refiner = ChapterRefiner(config=self.config)
        except KeyError as e:
            self.logger.error(
                f"Failed to initialize components due to missing config key: {e}"
            )
            raise

        processing_config = self.config.get("processing", {})
        logging_config = self.config.get("logging", {})

        self.base_temp_dir = processing_config.get(
            "download_path", "absrefined_temp_audio"
        )
        self.debug_preserve_files = logging_config.get("debug_files", False)

        costs_config = self.config.get("costs", {})
        self.llm_cost_per_mil_prompt_tokens = costs_config.get(
            "llm_refinement_cost_per_million_prompt_tokens", 0.0
        )
        self.llm_cost_per_mil_completion_tokens = costs_config.get(
            "llm_refinement_cost_per_million_completion_tokens", 0.0
        )
        self.transcription_cost_per_minute = costs_config.get(
            "audio_transcription_cost_per_minute", 0.0
        )
        self.logger.info(
            f"Costs loaded: LLM Prompt CPMT: ${self.llm_cost_per_mil_prompt_tokens:.4f}, LLM Completion CPMT: ${self.llm_cost_per_mil_completion_tokens:.4f}, Transcription CPM: ${self.transcription_cost_per_minute:.4f}"
        )

        self.logger.info(
            f"ChapterRefinementTool initialized. Base temp dir: {self.base_temp_dir}, Debug preserve: {self.debug_preserve_files}"
        )

        os.makedirs(self.base_temp_dir, exist_ok=True)
        self._check_ffmpeg_ffprobe()

    def _check_ffmpeg_ffprobe(self):
        """Checks if ffmpeg and ffprobe are available in PATH."""
        for cmd in ["ffmpeg", "ffprobe"]:
            if shutil.which(cmd) is None:
                self.logger.error(
                    f"Error: '{cmd}' not found. Please ensure ffmpeg is installed and in your system PATH."
                )
                raise EnvironmentError(f"{cmd} not found. Cannot proceed.")
        self.logger.debug("ffmpeg and ffprobe found.")

    def _ensure_full_audio_downloaded(self, item_id: str) -> str | None:
        """Downloads the full audio file if it doesn't exist, returns the path."""
        prospective_filename = f"{item_id}_full_audio.m4a"
        full_audio_path_to_check = os.path.join(
            self.base_temp_dir, prospective_filename
        )

        if not os.path.exists(full_audio_path_to_check):
            self.logger.debug(
                f"Default path {full_audio_path_to_check} not found. Checking other extensions."
            )
            for ext in ["mp3", "ogg", "opus", "flac", "wav", "m4b"]:
                path_with_other_ext = os.path.join(
                    self.base_temp_dir, f"{item_id}_full_audio.{ext}"
                )
                if os.path.exists(path_with_other_ext):
                    full_audio_path_to_check = path_with_other_ext
                    self.logger.info(
                        f"Found existing full audio file: {full_audio_path_to_check}"
                    )
                    break

        if os.path.exists(full_audio_path_to_check):
            self.logger.info(
                f"Using existing full audio file: {full_audio_path_to_check}"
            )
            return full_audio_path_to_check

        self.logger.info(
            f"Downloading full audio file to {full_audio_path_to_check} (or similar)..."
        )

        downloaded_path = self.abs_client.download_audio_file(
            item_id, full_audio_path_to_check, debug_preserve_files=self.debug_preserve_files
        )

        if not downloaded_path or not os.path.exists(downloaded_path):
            self.logger.error(
                f"Failed to download or locate full audio file for item {item_id} at {downloaded_path or full_audio_path_to_check}."
            )
            return None

        self.logger.info(f"Download complete: {downloaded_path}")
        
        # If the downloaded path is different from the expected path, but the file exists,
        # ensure we return the actual path where the file is located
        if downloaded_path != full_audio_path_to_check and os.path.exists(downloaded_path):
            self.logger.info(f"Note: Downloaded file path ({downloaded_path}) differs from expected path ({full_audio_path_to_check})")
            return downloaded_path
        
        return downloaded_path

    def process_item(
        self,
        item_id: str,
        search_window_seconds: int,
        model_name_override: str | None,
        dry_run: bool = False,
    ) -> Dict:
        """
        Process an item to refine its chapter markers.
        Args:
            item_id (str): ID of the item to process.
            search_window_seconds (int): Transcription window around chapter marks.
            model_name_override (str | None): Specific LLM model name to use, overrides config if provided.
            dry_run (bool): If True, no changes will be made to the server (currently handled by caller).
        Returns:
            Dict: Result dictionary (same structure as before).
        """
        result = {
            "item_id": item_id,
            "total_chapters": 0,
            "refined_chapters": 0,
            "audio_file_path": None,
            "chapter_details": None,
            "error": None,
        }

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

            _update_progress(10, "Checking/Downloading audio file...")
            full_audio_path = self._ensure_full_audio_downloaded(item_id)
            if not full_audio_path:
                result["error"] = "Failed to download full audio file."
                _update_progress(100, "Failed: Audio download error.")
                return result
            result["audio_file_path"] = full_audio_path
            _update_progress(15, "Audio file ready.")
            _update_progress(17, "Getting audio duration...")
            audio_duration = self._get_audio_duration(full_audio_path)
            if audio_duration is None:
                self.logger.warning(
                    f"Could not determine audio duration for {full_audio_path}. Proceeding without duration check."
                )
            _update_progress(20, "Starting chapter processing...")

            processed_chapters_list = self.process_chapters(
                item_id=item_id,
                original_chapters=original_chapters,
                full_audio_path=full_audio_path,
                audio_duration=audio_duration,
                search_window_seconds=search_window_seconds,
                model_name_override=model_name_override,
            )

            if processed_chapters_list is None:
                self.logger.error("Chapter processing failed critically.")
                result["error"] = "Chapter processing failed (check logs for details)."
                _update_progress(100, "Failed: Chapter processing error.")
                return result

            _update_progress(95, "Finalizing results...")
            final_chapter_details = []
            refined_count = 0
            significant_change_threshold = 0.1

            # Accumulators for usage and cost calculation
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_refined_tokens = 0
            total_transcribed_duration_seconds = 0.0

            for i, processed_chapter in enumerate(processed_chapters_list):
                original_start = processed_chapter.get("original_start")
                refined_start = processed_chapter.get("refined_start")

                # Skip counting the first chapter as refined, even if its start time is different
                if (
                    i > 0  # Skip the first chapter
                    and original_start is not None
                    and refined_start is not None
                    and abs(refined_start - original_start)
                    > significant_change_threshold
                ):
                    refined_count += 1

                # Append details for this chapter, including the ABSOLUTE refined time
                final_chapter_details.append(
                    {
                        "id": processed_chapter.get("id"),
                        "title": processed_chapter.get(
                            "title", f"Chapter {processed_chapter.get('index', 'N/A')}"
                        ),
                        "original_start": original_start,  # Absolute
                        "refined_start": refined_start,  # Absolute (or None)
                        "chunk_path": processed_chapter.get("chunk_path"),
                        "window_start_time": processed_chapter.get(
                            "window_start_time"
                        ),  # Absolute start of chunk
                    }
                )

                # Calculate usage and cost for this chapter
                if processed_chapter.get("usage_data"):
                    usage_data = processed_chapter["usage_data"]
                    total_prompt_tokens += usage_data.get("prompt_tokens", 0)
                    total_completion_tokens += usage_data.get("completion_tokens", 0)
                    total_refined_tokens += usage_data.get("total_tokens", 0)
                    self.logger.debug(
                        f"Chapter '{processed_chapter['title']}' LLM usage: P:{usage_data.get('prompt_tokens', 0)}, C:{usage_data.get('completion_tokens', 0)}, T:{usage_data.get('total_tokens', 0)}"
                    )

                # Calculate transcription duration for this chapter
                if processed_chapter.get("chunk_duration_seconds"):
                    chunk_duration_seconds = processed_chapter["chunk_duration_seconds"]
                    total_transcribed_duration_seconds += chunk_duration_seconds

            result["chapter_details"] = final_chapter_details
            result["refined_chapters"] = refined_count

            # Calculate costs
            refinement_cost = 0.0
            if total_refined_tokens > 0:  # Only calculate if there were tokens
                prompt_cost = 0.0
                if self.llm_cost_per_mil_prompt_tokens > 0 and total_prompt_tokens > 0:
                    prompt_cost = (
                        total_prompt_tokens / 1_000_000.0
                    ) * self.llm_cost_per_mil_prompt_tokens

                completion_cost = 0.0
                if (
                    self.llm_cost_per_mil_completion_tokens > 0
                    and total_completion_tokens > 0
                ):
                    completion_cost = (
                        total_completion_tokens / 1_000_000.0
                    ) * self.llm_cost_per_mil_completion_tokens

                refinement_cost = prompt_cost + completion_cost

            transcription_cost = 0.0
            if (
                self.transcription_cost_per_minute > 0
                and total_transcribed_duration_seconds > 0
            ):
                transcription_cost = (
                    total_transcribed_duration_seconds / 60.0
                ) * self.transcription_cost_per_minute

            result["refinement_usage"] = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_refined_tokens,
                "estimated_cost": refinement_cost,
            }
            result["transcription_usage"] = {
                "duration_seconds": total_transcribed_duration_seconds,
                "duration_minutes": total_transcribed_duration_seconds / 60.0,
                "estimated_cost": transcription_cost,
            }

            self.logger.info(
                f"Refinement token usage: Prompt={total_prompt_tokens}, Completion={total_completion_tokens}, Total={total_refined_tokens}. Estimated Cost: ${refinement_cost:.4f}"
            )
            self.logger.info(
                f"Transcription usage: Duration={total_transcribed_duration_seconds:.2f}s ({total_transcribed_duration_seconds / 60.0:.2f}m). Estimated Cost: ${transcription_cost:.4f}"
            )

            _update_progress(100, "Processing complete.")

            # The first chapter should always start at 0, regardless of LLM output
            # Add the 'refined_start_abs' to each original chapter for the update
            chapters_for_update = []
            id_to_refined_start_map = {}
            for i, chapter in enumerate(original_chapters):
                id_to_refined_start_map[chapter["id"]] = final_chapter_details[i][
                    "refined_start"
                ]
                if final_chapter_details[i]["refined_start"] is not None:
                    chapters_for_update.append(i)

            # Cap at 99 before potential push
            chapters_for_update = chapters_for_update[:99]

            result["chapters_for_update"] = chapters_for_update
            result["usage_and_cost"] = {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_refined_tokens": total_refined_tokens,
                "estimated_llm_cost": refinement_cost,
                "estimated_transcription_cost": transcription_cost,
                "total_estimated_cost": refinement_cost + transcription_cost,
            }

            if dry_run:
                _update_progress(100, "Changes not pushed to server.")

            self.logger.info(f"Item processing completed for {item_id}.")
            cost_info = result.get("usage_and_cost", {})
            self.logger.info(
                f"Tokens: Prompt={cost_info.get('total_prompt_tokens', 0)}, Completion={cost_info.get('total_completion_tokens', 0)}. Transcribed: {cost_info.get('total_transcribed_duration_seconds', 0):.2f}s"
            )
            self.logger.info(
                f"Est. Costs: LLM=${cost_info.get('estimated_llm_cost', 0.0):.4f}, Transcription=${cost_info.get('estimated_transcription_cost', 0.0):.4f}, Total=${cost_info.get('total_estimated_cost', 0.0):.4f}"
            )

        except EnvironmentError as e:
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
        search_window_seconds: int,
        model_name_override: str | None,
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
            search_window_seconds (int): Transcription window around chapter marks.
            model_name_override (str | None): Specific LLM model name to use, overrides config if provided.

        Returns:
            List[Dict] | None: A list of chapter detail dictionaries, each containing:
                              {'id': str, 'title': str, 'original_start': float, 'refined_start': float | None}
                              Returns None on critical failure.
        """
        all_chapter_details = []
        total_chapters = len(original_chapters)
        base_progress = 20  # Start progress reporting after initial steps
        progress_range = (
            95 - base_progress
        )  # Progress percentage allocated to this loop

        self.logger.info(
            f"Processing {total_chapters} chapters using ±{search_window_seconds}s window extraction..."
        )

        # Helper for progress within this method
        def _update_chapter_progress(index: int, message: str):
            if self.progress_callback:
                percent = base_progress + int(
                    ((index + 1) / total_chapters) * progress_range
                )
                try:
                    self.progress_callback(
                        percent, f"Ch.{index + 1}/{total_chapters}: {message}"
                    )
                except Exception as cb_err:
                    self.logger.warning(f"Progress callback failed: {cb_err}")

        # Get base name and extension for temp files
        audio_base, audio_ext = os.path.splitext(os.path.basename(full_audio_path))

        for i, chapter in enumerate(
            tqdm(original_chapters, desc="Processing Chapters")
        ):
            _update_chapter_progress(i, f"Starting '{chapter.get('title', 'Unknown')}'")

            chapter_id = chapter.get("id")
            chapter_title = chapter.get("title", f"Chapter {i + 1}")
            original_start_time = chapter.get("start")

            if original_start_time is None:
                self.logger.warning(
                    f"Chapter '{chapter_title}' missing start time. Using 0.0 for original time."
                )
                original_start_time = 0.0

            # Skip processing for first chapter (index 0) - it should always start at time 0
            if i == 0:
                self.logger.info(
                    f"Skipping refinement for first chapter '{chapter_title}' - setting start time to 0.0"
                )
                all_chapter_details.append(
                    {
                        "id": chapter_id,
                        "title": chapter_title,
                        "original_start": original_start_time,
                        "refined_start": 0.0,  # Always set to 0.0
                        "chunk_path": None,
                        "window_start_time": None,
                        "usage_data": None,
                        "chunk_duration_seconds": 0.0,
                    }
                )
                continue

            # Define window ABSOLUTE boundaries
            window_start = max(0.0, original_start_time - search_window_seconds / 2)
            window_end = original_start_time + search_window_seconds / 2
            if audio_duration is not None:  # Cap window_end at audio duration
                window_end = min(window_end, audio_duration)

            # Calculate actual chunk duration for the refiner prompt
            actual_chunk_duration = window_end - window_start

            # Path for this chapter's audio chunk
            chunk_audio_path = None
            if self.debug_preserve_files:
                safe_title = re.sub(r"[^\w\-_]", "_", chapter_title)
                chunk_audio_path = os.path.join(
                    self.base_temp_dir,
                    f"{item_id}_chapter_{i + 1}_{safe_title}_{window_start:.1f}s-{window_end:.1f}s.wav",
                )
                self.logger.info(
                    f"Debug mode: Saving audio chunk to {chunk_audio_path}"
                )
            else:
                # Use tempfile for non-debug to guarantee uniqueness, but don't delete automatically
                with tempfile.NamedTemporaryFile(
                    prefix=f"{item_id}_ch{i + 1}_",
                    suffix=".wav",
                    dir=self.base_temp_dir,
                    delete=False,
                ) as temp_f:
                    chunk_audio_path = temp_f.name
                    # Do NOT add chunk_audio_path to _temp_files_to_delete here, GUI cleans dir

            # Path for this chapter's transcript (only used if debug)
            chunk_transcript_path = None
            if self.debug_preserve_files:
                chunk_transcript_path = os.path.join(
                    self.base_temp_dir,
                    f"{item_id}_chapter_{i + 1}_{safe_title}_transcript.jsonl",
                )
                self.logger.info(
                    f"Debug mode: Saving transcript to {chunk_transcript_path}"
                )

            _update_chapter_progress(i, "Extracting audio segment...")
            segment_extracted = self._extract_audio_segment(
                full_audio_path, window_start, window_end, chunk_audio_path
            )

            refined_start_time_abs = None  # Initialize
            transcript_segments_absolute = None  # Store absolute transcript temporarily

            if segment_extracted and os.path.exists(chunk_audio_path):
                _update_chapter_progress(i, "Transcribing segment...")
                # Transcriber returns segments with ABSOLUTE timestamps relative to full audio
                transcript_segments_absolute = self.transcriber.transcribe_audio(
                    chunk_audio_path,
                    # Output file is only for the user-specified JSONL, separate from debug
                    # We don't need to write the final adjusted transcript here by default
                    output_file=chunk_transcript_path
                    if self.debug_preserve_files
                    else None,
                    segment_start_time=window_start,  # Tell transcriber the absolute start time
                    write_to_file=self.debug_preserve_files,  # Only write if debugging
                )

                if transcript_segments_absolute:
                    # --- NORMALIZATION STEP ---
                    # Convert absolute timestamps in transcript to be relative (0-based) for the refiner
                    normalized_transcript_for_refiner = []
                    for seg_abs in transcript_segments_absolute:
                        # Basic check for expected keys before proceeding
                        if not all(k in seg_abs for k in ("start", "end", "text")):
                            self.logger.warning(
                                f"Skipping segment due to missing keys: {seg_abs}"
                            )
                            continue

                        norm_seg = {
                            "start": seg_abs["start"] - window_start,
                            "end": seg_abs["end"] - window_start,
                            "text": seg_abs["text"],
                            "words": [],
                        }
                        # Normalize word timestamps if they exist
                        if "words" in seg_abs and seg_abs["words"]:
                            for word_abs in seg_abs["words"]:
                                if not all(
                                    k in word_abs for k in ("start", "end", "word")
                                ):
                                    self.logger.warning(
                                        f"Skipping word due to missing keys: {word_abs}"
                                    )
                                    continue
                                norm_seg["words"].append(
                                    {
                                        "word": word_abs["word"],
                                        "start": word_abs["start"] - window_start,
                                        "end": word_abs["end"] - window_start,
                                        "probability": word_abs.get("probability"),
                                    }
                                )
                        # Clamp relative times to be within [0, actual_chunk_duration]
                        norm_seg["start"] = max(
                            0.0, min(norm_seg["start"], actual_chunk_duration)
                        )
                        norm_seg["end"] = max(
                            0.0, min(norm_seg["end"], actual_chunk_duration)
                        )
                        if norm_seg["words"]:
                            for word in norm_seg["words"]:
                                word["start"] = max(
                                    0.0, min(word["start"], actual_chunk_duration)
                                )
                                word["end"] = max(
                                    0.0, min(word["end"], actual_chunk_duration)
                                )

                        normalized_transcript_for_refiner.append(norm_seg)
                    # --------------------------

                    _update_chapter_progress(i, "Refining timestamp with LLM...")
                    # Calculate target time relative to chunk start for the LLM
                    target_time_relative_to_chunk = original_start_time - window_start

                    # Call refiner with NORMALIZED transcript and RELATIVE target time
                    refined_offset_in_chunk, usage_data = (
                        self.refiner.refine_chapter_start_time(
                            transcript_segments=normalized_transcript_for_refiner,  # Use normalized
                            chapter_title=chapter_title,
                            target_time_seconds=target_time_relative_to_chunk,  # Use relative target
                            search_window_seconds=actual_chunk_duration,  # Pass actual duration
                            model_name_override=model_name_override,
                        )
                    )

                    if refined_offset_in_chunk is not None:
                        # --- CONVERT BACK TO ABSOLUTE ---
                        # Convert the relative offset back to an absolute timestamp
                        refined_start_time_abs = window_start + refined_offset_in_chunk
                        # ------------------------------
                        self.logger.info(
                            f"Chapter '{chapter_title}': Original: {format_timestamp(original_start_time)}, Refined: {format_timestamp(refined_start_time_abs)}"
                        )
                    else:
                        self.logger.info(
                            f"LLM refinement failed for '{chapter_title}'. Keeping original timestamp."
                        )
                else:
                    self.logger.warning(
                        f"Skipping LLM refinement for '{chapter_title}' due to transcription failure/empty result."
                    )
            else:
                self.logger.warning(
                    f"Skipping segment extraction for '{chapter_title}' due to transcription failure/empty result."
                )

            # Debugging values for playback keys
            final_chunk_path_to_store = None
            final_window_start_to_store = None

            log_msg_prefix = f"Chapter '{chapter_title}' (ID: {chapter_id if chapter_id else 'N/A'}) details for GUI:"
            self.logger.debug(
                f"{log_msg_prefix} Current segment_extracted={segment_extracted}"
            )
            self.logger.debug(
                f"{log_msg_prefix} Current original chunk_audio_path='{chunk_audio_path}'"
            )

            if segment_extracted:
                final_window_start_to_store = (
                    window_start  # window_start is from the loop, absolute
                )
                if chunk_audio_path and os.path.exists(chunk_audio_path):
                    final_chunk_path_to_store = chunk_audio_path
                    self.logger.debug(
                        f"{log_msg_prefix} File check os.path.exists('{chunk_audio_path}') is True. Storing path."
                    )
                elif chunk_audio_path:  # Path string exists but file doesn't
                    self.logger.warning(
                        f"{log_msg_prefix} File check os.path.exists('{chunk_audio_path}') is False. Storing None for chunk_path."
                    )
                else:  # chunk_audio_path is None (e.g. tempfile creation failed, though unlikely)
                    self.logger.warning(
                        f"{log_msg_prefix} Original chunk_audio_path is None. Storing None for chunk_path."
                    )
            else:  # segment_extracted is False
                self.logger.warning(
                    f"{log_msg_prefix} segment_extracted is False. Storing None for chunk_path and window_start_time."
                )

            self.logger.debug(
                f"{log_msg_prefix} Storing chunk_path='{final_chunk_path_to_store}'"
            )
            self.logger.debug(
                f"{log_msg_prefix} Storing window_start_time={final_window_start_to_store}"
            )

            # Append details for this chapter, including the ABSOLUTE refined time
            all_chapter_details.append(
                {
                    "id": chapter_id,
                    "title": chapter_title,
                    "original_start": original_start_time,  # Absolute
                    "refined_start": refined_start_time_abs,  # Absolute (or None)
                    "chunk_path": final_chunk_path_to_store,
                    "window_start_time": final_window_start_to_store,  # Absolute start of chunk
                    "usage_data": usage_data,
                    "chunk_duration_seconds": actual_chunk_duration,
                }
            )

            # Cleanup individual chunk and transcript (if not debugging)
            # The transcript path was only created if debug_preserve_files was true,
            # so no explicit cleanup needed here for it if not debugging.

            _update_chapter_progress(i, "Done.")

        self.logger.info("Finished processing all chapters.")
        return all_chapter_details

    def _extract_audio_segment(
        self,
        full_audio_path: str,
        window_start: float,
        window_end: float,
        chunk_audio_path: str | None,
    ) -> bool:
        """Extracts an audio segment from the full audio file and saves it to the specified path."""
        if chunk_audio_path is None:
            return False

        try:
            cmd = [
                "ffmpeg",
                "-i",
                full_audio_path,
                "-ss",
                str(window_start),
                "-to",
                str(window_end),
                "-vn",
                "-acodec",
                "pcm_s16le",  # Output WAV format
                "-y",
                chunk_audio_path,
            ]
            self.logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            # Only log stderr if verbose or error indication
            if result.stderr and (
                self.logger.isEnabledFor(logging.DEBUG)
                or "error" in result.stderr.lower()
                or "warning" in result.stderr.lower()
            ):
                self.logger.debug(f"ffmpeg stderr:\n{result.stderr.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error extracting audio segment: {e}")
            self.logger.error(f"Command: {' '.join(e.cmd)}")
            self.logger.error(
                f"ffmpeg stderr: {e.stderr.strip() if e.stderr else '[No stderr]'}"
            )
            return False
        except Exception as e:
            self.logger.exception(
                f"Unexpected error during audio segment extraction: {e}"
            )
            return False
