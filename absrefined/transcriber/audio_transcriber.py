import os
import json
from typing import List, Dict
import logging
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError


class AudioTranscriber:
    """Class for transcribing audio segments using the OpenAI API."""

    def __init__(self, api_key: str, verbose: bool = False, debug: bool = False):
        """
        Initialize the transcriber.

        Args:
            api_key (str): The OpenAI API key.
            verbose (bool): Whether to print verbose output
            debug (bool): Whether to save all transcripts for debugging
        """
        self.api_key = api_key
        self.verbose = verbose
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = OpenAI(api_key=self.api_key)
        self.logger.debug("AudioTranscriber initialized with OpenAI API.")

    def transcribe_audio(
        self,
        audio_file: str,
        output_file: str | None = None,
        segment_start_time: float = 0,
        write_to_file: bool = True,
    ) -> List[Dict]:
        """
        Transcribe audio using the OpenAI API with word-level timestamps.
        Timestamps are adjusted based on the segment_start_time.

        Args:
            audio_file (str): Path to the audio file (chunk).
            output_file (str | None): Path to save the transcription for this chunk. Required if write_to_file is True.
            segment_start_time (float): The absolute start time of this audio_file
                                        segment in the original large file (in seconds).
                                        Defaults to 0.
            write_to_file (bool): Whether to write the adjusted transcription to output_file. Defaults to True.


        Returns:
            List[Dict]: Adjusted transcription segments for the chunk.
        """
        if write_to_file and output_file is None:
            raise ValueError("output_file must be provided if write_to_file is True")

        # Create debug output filename if in debug mode but no output file specified
        debug_output_file = None
        if self.debug and not write_to_file:
            audio_filename = os.path.basename(audio_file)
            base_name = os.path.splitext(audio_filename)[0]
            debug_dir = os.path.dirname(audio_file)
            debug_output_file = os.path.join(debug_dir, f"{base_name}_transcript.jsonl")
            self.logger.info(f"Debug mode: Will save transcript to {debug_output_file}")
            write_to_file = True
            output_file = debug_output_file

        try:
            self.logger.debug(f"Transcribing {audio_file} with OpenAI API (whisper-1)...")
            self.logger.debug(f"Applying time offset: {segment_start_time:.2f}s")
            if write_to_file:
                self.logger.debug(f"Output will be saved to {output_file}")

            with open(audio_file, "rb") as audio_data:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_data,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            # --- Process the response --- 
            openai_segments = response.segments
            openai_words = response.words

            if not openai_segments and not openai_words:
                 self.logger.warning(f"OpenAI API returned no segments or words for {audio_file}")
                 return []

            adjusted_segments = []

            if openai_segments: # Prioritize segments if available
                self.logger.debug(
                    f"Transcription complete. Found {len(openai_segments)} segments."
                )
                for segment in openai_segments:
                    adjusted_segment = {
                        "start": segment.start + segment_start_time,
                        "end": segment.end + segment_start_time,
                        "text": segment.text.strip(),
                    }
                    if hasattr(segment, 'words') and segment.words:
                        adjusted_words = []
                        for word in segment.words:
                            if hasattr(word, 'start') and hasattr(word, 'end'):
                                adjusted_word = {
                                    "word": word.word,
                                    "start": word.start + segment_start_time,
                                    "end": word.end + segment_start_time,
                                    "probability": getattr(word, 'probability', None) # Include probability if available
                                }
                                adjusted_words.append(adjusted_word)
                            else:
                                self.logger.warning(f"Word missing start/end time in segment: {word}")
                        adjusted_segment["words"] = adjusted_words
                    else:
                        adjusted_segment["words"] = []
                    adjusted_segments.append(adjusted_segment)

            elif openai_words: # Fallback: create a synthetic segment from words if segments are missing
                 self.logger.warning(f"OpenAI API returned words but no segments for {audio_file}. Creating synthetic segment.")
                 all_adjusted_words = []
                 min_start = float('inf')
                 max_end = float('-inf')
                 full_text = []

                 for word in openai_words:
                     if hasattr(word, 'start') and hasattr(word, 'end'):
                         adj_start = word.start + segment_start_time
                         adj_end = word.end + segment_start_time
                         min_start = min(min_start, adj_start)
                         max_end = max(max_end, adj_end)
                         full_text.append(word.word)
                         adjusted_word = {
                                "word": word.word,
                                "start": adj_start,
                                "end": adj_end,
                                "probability": getattr(word, 'probability', None)
                            }
                         all_adjusted_words.append(adjusted_word)
                     else:
                         self.logger.warning(f"Word missing start/end time in word list: {word}")
                 
                 if all_adjusted_words: # Only create segment if words were valid
                     synthetic_segment = {
                         "start": min_start,
                         "end": max_end,
                         "text": " ".join(full_text).strip(),
                         "words": all_adjusted_words
                     }
                     adjusted_segments.append(synthetic_segment)
                     self.logger.debug(f"Created synthetic segment covering {min_start:.2f}s - {max_end:.2f}s")
                 else:
                     self.logger.warning(f"Could not create synthetic segment for {audio_file} as no valid words were found.")
                     return [] # Return empty if no valid words to make segment from
            
            # If we reach here and adjusted_segments is still empty, something unexpected happened.
            if not adjusted_segments:
                 self.logger.error(f"Failed to process transcription response for {audio_file} despite receiving data.")
                 return []

            # Logging first segment time range (works for both real and synthetic)
            self.logger.debug(
                f"First adjusted segment time range: {adjusted_segments[0]['start']:.2f}s - {adjusted_segments[0]['end']:.2f}s"
            )

            # --- File Writing Logic ---
            if write_to_file:
                abs_output_file = os.path.abspath(output_file)
                os.makedirs(os.path.dirname(abs_output_file), exist_ok=True)

                with open(abs_output_file, "w", encoding="utf-8") as f:
                    desc = f"Writing adjusted segments for {os.path.basename(audio_file)}"
                    for segment in tqdm(adjusted_segments, desc=desc, disable=not self.verbose):
                        json.dump(segment, f)
                        f.write("\n")
                self.logger.debug(f"Adjusted transcription saved to {output_file}")

            return adjusted_segments
        except ImportError as e:
             self.logger.error(f"Error: Required library not installed. {e}")
             raise
        except AuthenticationError:
            self.logger.error("OpenAI API Error: Authentication failed. Check your API key.")
            raise
        except RateLimitError:
            self.logger.error("OpenAI API Error: Rate limit exceeded. Please check your plan and usage.")
            raise
        except APIConnectionError as e:
            self.logger.error(f"OpenAI API Error: Could not connect to API. {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during OpenAI transcription or adjustment for {audio_file}: {e}")
            raise

    def read_transcription(self, transcription_file: str) -> List[Dict]:
        """
        Read transcription from a JSONL file.

        Args:
            transcription_file (str): Path to the transcription file

        Returns:
            List[Dict]: Transcription segments
        """
        if not os.path.exists(transcription_file):
            self.logger.debug(f"Transcription file not found: {transcription_file}")
            return []

        segments = []

        try:
            self.logger.debug(f"Reading transcription from {transcription_file}")

            with open(transcription_file, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    if line.strip():
                        segment = json.loads(line)
                        segments.append(segment)

            self.logger.debug(
                f"Read {len(segments)} segments from {transcription_file}"
            )
            if segments:
                self.logger.debug(
                    f"First segment time range read: {segments[0]['start']:.2f}s - {segments[0]['end']:.2f}s"
                )

            return segments
        except Exception as e:
            self.logger.error(f"Error reading transcription from {transcription_file}: {e}")
            return []
