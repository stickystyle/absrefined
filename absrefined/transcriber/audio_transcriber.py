import os
import json
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError


class AudioTranscriber:
    """Class for transcribing audio segments using an OpenAI compatible API."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transcriber using a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Raises:
            KeyError: If required configuration keys for OpenAI API are missing.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Extract OpenAI client settings from config (typically under 'refiner' or a dedicated 'openai' section)
        # Using 'refiner' as per previous config structure discussion
        openai_config = self.config.get("refiner", {})
        self.api_key = openai_config.get("openai_api_key")
        self.base_url = openai_config.get(
            "openai_api_url"
        )  # For custom OpenAI-compatible endpoints

        if not self.api_key:
            raise KeyError(
                "OpenAI API key (openai_api_key) not found in [refiner] config section."
            )
        # base_url can be None if targeting official OpenAI API, client handles it.

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.info(
                f"AudioTranscriber initialized. Target API URL: {self.base_url or 'Official OpenAI'}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise  # Re-raise critical initialization error

        # Debug behavior for preserving transcripts
        logging_config = self.config.get("logging", {})
        self.debug_preserve_files = logging_config.get("debug_files", False)

        # Transcription model (could also be from config['transcription'] if more granularity needed)
        self.whisper_model = openai_config.get(
            "whisper_model_name", "whisper-1"
        )  # Default to whisper-1
        self.logger.info(f"Using Whisper model for transcription: {self.whisper_model}")

    def transcribe_audio(
        self,
        audio_file: str,
        output_file: str | None = None,
        segment_start_time: float = 0,
        write_to_file: bool = True,
    ) -> List[Dict]:
        """
        Transcribe audio using an OpenAI compatible API with word-level timestamps.
        Timestamps are adjusted based on the segment_start_time.

        Args:
            audio_file (str): Path to the audio file (chunk).
            output_file (str | None): Path to save the adjusted transcription for this chunk as JSONL.
                                        Used if write_to_file is True.
            segment_start_time (float): The absolute start time of this audio_file
                                        segment in the original large file (in seconds).
            write_to_file (bool): Whether to write the adjusted transcription to output_file.
                                  If self.debug_preserve_files is True, a debug transcript will always be saved
                                  to the audio_file's directory, regardless of this flag.

        Returns:
            List[Dict]: Adjusted transcription segments for the chunk.
        """

        # Determine path for debug transcript if debug_preserve_files is True
        # This file is *always* created in debug mode for raw output inspection.
        debug_transcript_path = None
        if self.debug_preserve_files:
            audio_filename_base = os.path.splitext(os.path.basename(audio_file))[0]
            debug_dir = os.path.dirname(audio_file)  # Save next to the audio chunk
            os.makedirs(debug_dir, exist_ok=True)  # Ensure dir exists
            debug_transcript_path = os.path.join(
                debug_dir, f"{audio_filename_base}_transcript_DEBUG.jsonl"
            )
            self.logger.info(
                f"Debug mode: Raw OpenAI response segments will be saved to {debug_transcript_path}"
            )

        if write_to_file and output_file is None:
            # If the user explicitly wants to write to a file, output_file must be given.
            # This is distinct from the debug_transcript_path.
            raise ValueError(
                "output_file must be provided if write_to_file is True and not just for debugging."
            )

        try:
            self.logger.info(
                f"Transcribing {audio_file} using model '{self.whisper_model}' via {self.base_url or 'OpenAI'}."
            )
            self.logger.debug(f"Applying time offset: {segment_start_time:.2f}s")
            if write_to_file and output_file:
                self.logger.debug(
                    f"User-specified output transcript will be saved to {output_file}"
                )

            with open(audio_file, "rb") as audio_data:
                response = self.client.audio.transcriptions.create(
                    model=self.whisper_model,
                    file=audio_data,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )

            # The response object itself is what we might want to save for debugging
            # before any adjustments.
            if self.debug_preserve_files and debug_transcript_path:
                try:
                    with open(debug_transcript_path, "w", encoding="utf-8") as dbg_f:
                        # Save the raw segments and words from response for debugging
                        # The response object is not directly serializable easily.
                        # We can save response.segments and response.words if they exist.
                        raw_data_to_save = {}
                        if hasattr(response, "segments"):
                            raw_data_to_save["segments"] = [
                                seg.model_dump()
                                for seg in response.segments  # Use model_dump for Pydantic objects
                            ]
                        if hasattr(response, "words"):
                            raw_data_to_save["words"] = [
                                word.model_dump() for word in response.words
                            ]
                        if hasattr(response, "text"):
                            raw_data_to_save["text"] = response.text

                        json.dump(raw_data_to_save, dbg_f, indent=2)
                    self.logger.info(
                        f"Raw OpenAI response data saved to {debug_transcript_path}"
                    )
                except Exception as e_dbg:
                    self.logger.warning(
                        f"Could not save debug transcript to {debug_transcript_path}: {e_dbg}"
                    )

            # --- Process the response (segments and words) ---
            openai_segments_raw = (
                response.segments if hasattr(response, "segments") else []
            )
            openai_words_raw = response.words if hasattr(response, "words") else []

            if not openai_segments_raw and not openai_words_raw:
                self.logger.warning(
                    f"API returned no segments or words for {audio_file}"
                )
                return []

            adjusted_segments = []  # This will store our final segment list with adjusted times

            if openai_segments_raw:
                self.logger.debug(
                    f"Processing {len(openai_segments_raw)} segments from API response."
                )
                for segment_raw in openai_segments_raw:
                    # segment_raw is a Segment object from OpenAI library
                    adj_seg = {
                        "start": segment_raw.start + segment_start_time,
                        "end": segment_raw.end + segment_start_time,
                        "text": segment_raw.text.strip(),
                        "words": [],  # Initialize words list for this segment
                    }
                    # OpenAI's verbose_json for whisper-1 might provide words within each segment
                    # or a separate top-level 'words' list. Check segment_raw.words first.
                    segment_words_raw = (
                        segment_raw.words
                        if hasattr(segment_raw, "words") and segment_raw.words
                        else []
                    )

                    if not segment_words_raw and openai_words_raw:
                        # If segment has no words, but top-level words exist, try to map them
                        # This is complex; for now, we assume words are within segments if timestamp_granularities=["word"] is used.
                        # Or, we use the top-level list if segment_raw.words is empty.
                        # Let's use a simpler approach: if segment_raw.words is empty, iterate all openai_words_raw
                        # and assign those whose time falls within this segment_raw's original time.
                        # This is a fallback, ideally words are nested.
                        current_segment_orig_start = segment_raw.start
                        current_segment_orig_end = segment_raw.end

                        # Filter top-level words that belong to the current segment
                        # This assumes openai_words_raw contains words for the *entire* request (chunk)
                        relevant_top_level_words = [
                            w
                            for w in openai_words_raw
                            if hasattr(w, "start")
                            and hasattr(w, "end")
                            and w.start >= current_segment_orig_start
                            and w.end <= current_segment_orig_end
                        ]
                        if relevant_top_level_words:
                            segment_words_raw = (
                                relevant_top_level_words  # Use these instead
                            )
                            # self.logger.debug(f"Found {len(segment_words_raw)} words from top-level list for current segment.")

                    for word_raw in segment_words_raw:  # word_raw is a Word object
                        if hasattr(word_raw, "start") and hasattr(word_raw, "end"):
                            adj_seg["words"].append(
                                {
                                    "word": word_raw.word.strip(),  # Strip whitespace from word itself
                                    "start": word_raw.start + segment_start_time,
                                    "end": word_raw.end + segment_start_time,
                                    "probability": getattr(
                                        word_raw, "probability", None
                                    ),  # v2.0.0 has this as `probability`
                                }
                            )
                        else:
                            self.logger.warning(
                                f"Word object missing start/end time: {word_raw}"
                            )
                    adjusted_segments.append(adj_seg)

            elif openai_words_raw:  # Fallback: No segments, but words are present. Create one large segment.
                self.logger.warning(
                    f"API returned words but no segments for {audio_file}. Creating one synthetic segment."
                )
                all_adj_words_for_synthetic_segment = []
                min_s = float("inf")
                max_e = float("-inf")
                full_text_list = []
                for word_raw in openai_words_raw:
                    if hasattr(word_raw, "start") and hasattr(word_raw, "end"):
                        adj_s = word_raw.start + segment_start_time
                        adj_e = word_raw.end + segment_start_time
                        min_s = min(min_s, adj_s)
                        max_e = max(max_e, adj_e)
                        full_text_list.append(word_raw.word.strip())
                        all_adj_words_for_synthetic_segment.append(
                            {
                                "word": word_raw.word.strip(),
                                "start": adj_s,
                                "end": adj_e,
                                "probability": getattr(word_raw, "probability", None),
                            }
                        )
                    else:
                        self.logger.warning(
                            f"Word object missing start/end time in top-level list: {word_raw}"
                        )

                if all_adj_words_for_synthetic_segment:
                    adjusted_segments.append(
                        {
                            "start": min_s,
                            "end": max_e,
                            "text": " ".join(full_text_list),
                            "words": all_adj_words_for_synthetic_segment,
                        }
                    )
                    self.logger.debug(
                        f"Created synthetic segment from words: {min_s:.2f}s - {max_e:.2f}s"
                    )
                else:
                    self.logger.warning(
                        f"Could not create synthetic segment for {audio_file}, no valid words found."
                    )
                    return []

            if not adjusted_segments:
                self.logger.error(
                    f"Failed to produce any adjusted segments for {audio_file} from API response."
                )
                return []

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"First adjusted segment for '{os.path.basename(audio_file)}': "
                    f"{adjusted_segments[0]['start']:.2f}s - {adjusted_segments[0]['end']:.2f}s: "
                    f"'{adjusted_segments[0]['text'][:50]}...'"
                )

            # --- File Writing for user-specified output_file (JSONL) ---
            if write_to_file and output_file:
                # Ensure output_file path is absolute and directory exists
                final_output_path = os.path.abspath(output_file)
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                try:
                    with open(final_output_path, "w", encoding="utf-8") as f:
                        # Determine if tqdm should be disabled (e.g. if not verbose/interactive)
                        # For now, let's tie it to logger level or a specific config setting.
                        # Using a simple check: disable if logger is WARNING or above.
                        disable_tqdm = (
                            self.logger.getEffectiveLevel() >= logging.WARNING
                        )

                        for segment_to_write in tqdm(
                            adjusted_segments,
                            desc=f"Writing to {os.path.basename(final_output_path)}",
                            disable=disable_tqdm,
                        ):
                            json.dump(segment_to_write, f)
                            f.write("\n")
                    self.logger.info(
                        f"Adjusted transcription segments saved to {final_output_path}"
                    )
                except Exception as e_write:
                    self.logger.error(
                        f"Failed to write transcription to {final_output_path}: {e_write}"
                    )
                    # Do not re-raise here, as transcription itself succeeded.

            return adjusted_segments

        # Specific OpenAI errors handled by the SDK and re-raised if not caught by these:
        except AuthenticationError as e:
            self.logger.error(
                f"OpenAI API Authentication Error: {e}. Check your API key and organization if applicable."
            )
            raise  # Re-raise to be caught by caller
        except RateLimitError as e:
            self.logger.error(
                f"OpenAI API Rate Limit Error: {e}. Check your usage and limits."
            )
            raise
        except APIConnectionError as e:
            self.logger.error(
                f"OpenAI API Connection Error: {e}. Check network or API endpoint URL."
            )
            raise
        except (
            Exception
        ) as e:  # Catch-all for other errors during transcription process
            self.logger.exception(
                f"Unexpected error during transcription for {audio_file}: {e}"
            )
            raise  # Re-raise to be caught by caller
