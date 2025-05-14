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
            KeyError: If required configuration keys for API are missing.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # First check for dedicated transcription config
        transcription_config = self.config.get("transcription", {})
        use_local = transcription_config.get("use_local", False)

        # Debug what config values we're actually seeing
        self.logger.debug(f"Transcription config: {transcription_config}")
        self.logger.debug(
            f"use_local value detected: {use_local} (type: {type(use_local)})"
        )

        # Store OpenAI credentials for potential fallback
        self.openai_api_key = self.config.get("refiner", {}).get("openai_api_key")
        self.openai_api_url = self.config.get("refiner", {}).get("openai_api_url")

        # Track whether we're using local server for fallback logic
        self.use_local = use_local == True

        # Check if fallback is enabled
        self.enable_fallback = (
            transcription_config.get("enable_fallback", False) == True
        )

        # Use dedicated transcription API settings if available, otherwise fall back to refiner
        if self.use_local:
            self.api_key = transcription_config.get("api_key", "local-key")
            self.base_url = transcription_config.get(
                "api_url", "http://localhost:8000/v1"
            )
            self.logger.debug(
                "Local transcription server enabled via use_local=true setting"
            )
            self.logger.debug(f"Using local transcription server at {self.base_url}")

            # Check if we have OpenAI fallback credentials available
            if self.enable_fallback and self.openai_api_key:
                self.logger.debug(
                    "Fallback to OpenAI API is ENABLED if local server fails"
                )
            elif self.enable_fallback and not self.openai_api_key:
                self.logger.warning(
                    "Fallback is enabled but no OpenAI API credentials available"
                )
            else:
                self.logger.debug("Fallback to OpenAI API is DISABLED")
        elif "api_url" in transcription_config:
            self.api_key = transcription_config.get("api_key")
            self.base_url = transcription_config.get("api_url")
            self.logger.debug(f"Using custom transcription API at {self.base_url}")
        else:
            # Fall back to refiner config for backward compatibility
            self.api_key = self.openai_api_key
            self.base_url = self.openai_api_url
            self.logger.debug(f"Using OpenAI API for transcription")

        if not self.api_key:
            raise KeyError("API key for transcription not found in config")

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.debug(
                f"AudioTranscriber initialized. Target API URL: {self.base_url or 'Official OpenAI'}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise  # Re-raise critical initialization error

        # Debug behavior for preserving transcripts
        logging_config = self.config.get("logging", {})
        self.debug_preserve_files = logging_config.get("debug_files", False)

        # Get transcription model from transcription config first, fall back to refiner
        self.whisper_model = transcription_config.get(
            "whisper_model_name"
        ) or self.config.get("refiner", {}).get("whisper_model_name", "whisper-1")
        self.logger.debug(
            f"Using Whisper model for transcription: {self.whisper_model}"
        )

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
        Falls back to OpenAI API if local server fails.

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
            os.makedirs(debug_dir, exist_ok=True)
            debug_transcript_path = os.path.join(
                debug_dir, f"{audio_filename_base}_transcript_DEBUG.jsonl"
            )
            self.logger.debug(
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

            # Try with configured API first
            response = None
            try:
                with open(audio_file, "rb") as audio_data:
                    response = self.client.audio.transcriptions.create(
                        model=self.whisper_model,
                        file=audio_data,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                    )
                self.logger.info(
                    f"Successfully transcribed using {self.base_url or 'OpenAI'}"
                )
            except Exception as local_err:
                # Only try fallback if explicitly enabled
                if self.use_local and self.enable_fallback and self.openai_api_key:
                    self.logger.warning(
                        f"Local transcription server failed: {local_err}. Falling back to OpenAI API."
                    )

                    # Create temporary client for OpenAI API
                    fallback_client = OpenAI(
                        api_key=self.openai_api_key, base_url=self.openai_api_url
                    )

                    with open(audio_file, "rb") as audio_data:
                        self.logger.info(
                            f"Attempting fallback transcription with OpenAI model: whisper-1"
                        )
                        response = fallback_client.audio.transcriptions.create(
                            model="whisper-1",  # Always use stable OpenAI model for fallback
                            file=audio_data,
                            response_format="verbose_json",
                            timestamp_granularities=["word"],
                        )
                    self.logger.info(
                        "Successfully transcribed using OpenAI API fallback"
                    )
                else:
                    # Either not using local server, fallback disabled, or no credentials
                    self.logger.error(f"Transcription failed: {local_err}")
                    if self.use_local and not self.enable_fallback:
                        self.logger.error(
                            "OpenAI API fallback is disabled. Enable with 'enable_fallback = true' in config.toml to use fallback"
                        )
                    raise

            # The response object itself is what we might want to save for debugging
            # before any adjustments.
            if self.debug_preserve_files and debug_transcript_path:
                try:
                    with open(debug_transcript_path, "w", encoding="utf-8") as dbg_f:
                        # Save the raw data in a structured way that works for various response types
                        raw_data_to_save = {
                            "text": getattr(response, "text", str(response))
                            if response
                            else "No response"
                        }

                        # Try to extract segments if they exist
                        if hasattr(response, "segments") and response.segments:
                            try:
                                raw_data_to_save["segments"] = [
                                    seg.model_dump()
                                    if hasattr(seg, "model_dump")
                                    else vars(seg)
                                    for seg in response.segments
                                ]
                            except Exception as seg_err:
                                raw_data_to_save["segments_error"] = (
                                    f"Error extracting segments: {seg_err}"
                                )

                        # Try to extract words if they exist
                        if hasattr(response, "words") and response.words:
                            try:
                                raw_data_to_save["words"] = [
                                    word.model_dump()
                                    if hasattr(word, "model_dump")
                                    else vars(word)
                                    for word in response.words
                                ]
                            except Exception as word_err:
                                raw_data_to_save["words_error"] = (
                                    f"Error extracting words: {word_err}"
                                )

                        # For fallback, include additional metadata
                        raw_data_to_save["api_source"] = (
                            "OpenAI API Fallback"
                            if self.use_local and self.base_url != self.openai_api_url
                            else self.base_url or "OpenAI API"
                        )

                        json.dump(raw_data_to_save, dbg_f, indent=2)
                    self.logger.info(
                        f"Raw response data saved to {debug_transcript_path}"
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
                # Create a synthetic segment with the full text
                if hasattr(response, "text") and response.text:
                    self.logger.info("Creating synthetic segment from full text")
                    full_text = response.text
                    # Create a single segment with the full text
                    openai_segments_raw = [
                        {
                            "id": 0,
                            "text": full_text,
                            "start": 0.0,
                            "end": 30.0,  # Default to 30 seconds
                        }
                    ]
                else:
                    return []

            adjusted_segments = []

            if openai_segments_raw:
                self.logger.debug(
                    f"Processing {len(openai_segments_raw)} segments from API response."
                )
                for segment_raw in openai_segments_raw:
                    # Extract the start and end times, defaulting to 0 if not found
                    segment_start = (
                        getattr(segment_raw, "start", 0)
                        if hasattr(segment_raw, "start")
                        else segment_raw.get("start", 0)
                        if isinstance(segment_raw, dict)
                        else 0
                    )
                    segment_end = (
                        getattr(segment_raw, "end", 0)
                        if hasattr(segment_raw, "end")
                        else segment_raw.get("end", 0)
                        if isinstance(segment_raw, dict)
                        else 0
                    )
                    segment_text = (
                        getattr(segment_raw, "text", "")
                        if hasattr(segment_raw, "text")
                        else segment_raw.get("text", "")
                        if isinstance(segment_raw, dict)
                        else ""
                    )

                    adj_seg = {
                        "start": segment_start + segment_start_time,
                        "end": segment_end + segment_start_time,
                        "text": segment_text.strip(),
                        "words": [],
                    }

                    # Get words from segment if available
                    segment_words_raw = []
                    if hasattr(segment_raw, "words") and segment_raw.words:
                        segment_words_raw = segment_raw.words
                    elif (
                        isinstance(segment_raw, dict)
                        and "words" in segment_raw
                        and segment_raw["words"]
                    ):
                        segment_words_raw = segment_raw["words"]

                    if not segment_words_raw and openai_words_raw:
                        current_segment_orig_start = segment_start
                        current_segment_orig_end = segment_end

                        relevant_top_level_words = [
                            w
                            for w in openai_words_raw
                            if (
                                (
                                    hasattr(w, "start")
                                    and hasattr(w, "end")
                                    and w.start >= current_segment_orig_start
                                    and w.end <= current_segment_orig_end
                                )
                                or (
                                    isinstance(w, dict)
                                    and "start" in w
                                    and "end" in w
                                    and w["start"] >= current_segment_orig_start
                                    and w["end"] <= current_segment_orig_end
                                )
                            )
                        ]
                        if relevant_top_level_words:
                            segment_words_raw = relevant_top_level_words

                    # If we still don't have words, try to generate approximate timestamps
                    if not segment_words_raw and adj_seg["text"]:
                        self.logger.debug(
                            f"Generating approximate word timestamps for segment: '{adj_seg['text'][:30]}...'"
                        )
                        segment_words = adj_seg["text"].split()
                        segment_duration = adj_seg["end"] - adj_seg["start"]
                        if segment_words and segment_duration > 0:
                            avg_word_duration = segment_duration / len(segment_words)
                            for i, word in enumerate(segment_words):
                                word_start = adj_seg["start"] + (i * avg_word_duration)
                                word_end = word_start + avg_word_duration

                                adj_seg["words"].append(
                                    {
                                        "word": word,
                                        "start": word_start,
                                        "end": word_end,
                                        "probability": 0.5,  # Default probability for generated timestamps
                                    }
                                )
                            self.logger.debug(
                                f"Generated {len(segment_words)} approximate word timestamps"
                            )

                    # Process any available word timestamps
                    for word_raw in segment_words_raw:
                        if (
                            isinstance(word_raw, dict)
                            and "start" in word_raw
                            and "end" in word_raw
                        ):
                            # Dictionary format
                            adj_seg["words"].append(
                                {
                                    "word": word_raw.get(
                                        "word", word_raw.get("text", "")
                                    ),
                                    "start": word_raw["start"] + segment_start_time,
                                    "end": word_raw["end"] + segment_start_time,
                                    "probability": word_raw.get(
                                        "probability", word_raw.get("confidence", 1.0)
                                    ),
                                }
                            )
                        elif hasattr(word_raw, "start") and hasattr(word_raw, "end"):
                            # Object format
                            adj_seg["words"].append(
                                {
                                    "word": getattr(
                                        word_raw, "word", getattr(word_raw, "text", "")
                                    ),
                                    "start": word_raw.start + segment_start_time,
                                    "end": word_raw.end + segment_start_time,
                                    "probability": getattr(
                                        word_raw,
                                        "probability",
                                        getattr(word_raw, "confidence", 1.0),
                                    ),
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

    def transcribe_file(self, audio_path: str, output_file: str | None = None) -> Dict:
        """
        Transcribe an entire audio file and return the result as a dictionary.

        Args:
            audio_path (str): Path to the audio file.
            output_file (str | None): Path to save the transcription output.

        Returns:
            Dict: Transcription result containing 'text' and 'segments'.
        """
        # Using 'refiner' as per previous config structure discussion
        openai_config = self.config.get("refiner", {})
        # Fallback if whisper model name not found in config
        model_name = openai_config.get("whisper_model_name", "whisper-1")
        # Default name for the transcript file if not provided
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        # Main file path for the transcription output (non-debug)
        default_output_path = os.path.join(
            os.path.dirname(audio_path), f"{base_name}_transcript.jsonl"
        )

        segments = self.transcribe_audio(
            audio_path,
            output_file=default_output_path,
            segment_start_time=0,  # Assuming the file is the whole segment for this simpler method
            write_to_file=True,  # Always write for this public method unless overridden by caller
        )

        full_text = " ".join([seg["text"] for seg in segments if seg.get("text")])
        self.logger.info(
            f"Transcription for {audio_path} completed. Text length: {len(full_text)}"
        )

        # Basic error handling for API connection or authentication issues
        # More specific errors are caught within transcribe_audio
        return {"text": full_text, "segments": segments}
