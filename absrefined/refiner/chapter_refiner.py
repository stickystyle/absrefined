from typing import Dict, List, Optional, Any
import logging
from openai import OpenAI, APIError


class ChapterRefiner:
    """Class for refining chapter markers using LLM analysis via an OpenAI-compatible API."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chapter refiner using a configuration dictionary.

        Args:
            config (Dict[str, Any]): The main configuration dictionary.

        Raises:
            KeyError: If required configuration keys for the LLM API are missing.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        refiner_config = self.config.get("refiner", {})
        self.api_base_url = refiner_config.get("openai_api_url")
        self.api_key = refiner_config.get("openai_api_key")
        self.default_model_name = refiner_config.get("model_name", "gpt-4o-mini")

        if not self.api_key:
            self.logger.error(
                "OpenAI API key (openai_api_key) not found in [refiner] config section."
            )
            raise KeyError("Missing openai_api_key in refiner configuration")
        if not self.api_base_url:
            # Some users might use official OpenAI, where base_url is not strictly needed for the client if it defaults.
            # However, for consistency and supporting local LLMs, we expect it.
            # If OpenAI library handles default base_url=None correctly, this check could be softened to a warning.
            # For now, enforce it as per previous gui.py and config.example.toml structure.
            self.logger.error(
                "OpenAI API URL (openai_api_url) not found in [refiner] config section."
            )
            raise KeyError("Missing openai_api_url in refiner configuration")

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
            self.logger.info(
                f"ChapterRefiner initialized. Target API URL: {self.api_base_url}, Default Model: {self.default_model_name}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to initialize OpenAI client for ChapterRefiner: {e}"
            )
            raise

        # self.window_size is removed; context window info (search_window_seconds) is passed to methods.

    # verify_api_key method can be removed or kept if more detailed key validation is needed beyond client init.
    # For now, successful client initialization is the primary check.

    def refine_chapter_start_time(
        self,
        transcript_segments: List[
            Dict
        ],  # Segments from the relevant chunk, 0-based timestamps for this chunk
        chapter_title: str,
        target_time_seconds: float,  # Original chapter start, relative to the chunk start (0-based)
        search_window_seconds: float,  # The full duration of the audio chunk/window being analyzed
        model_name_override: Optional[str] = None,
    ) -> Optional[float]:  # Returns refined time offset *within the chunk*, or None
        """
        Detects the precise start of a chapter within the provided transcript chunk.

        Args:
            transcript_segments (List[Dict]): Transcript segments of the audio chunk.
                                             Timestamps are relative to the CHUNK start (0).
            chapter_title (str): Name of the chapter.
            target_time_seconds (float): Original expected start time of the chapter, relative to chunk start.
            search_window_seconds (float): Total duration of the provided transcript_segments/audio chunk.
            model_name_override (str, optional): Specific LLM model name to use, overrides default from config.

        Returns:
            Tuple[Optional[float], Optional[Dict[str, int]]]: Detected chapter start time (offset from chunk start)
                                                               in seconds, and a dictionary containing token usage
                                                               (prompt_tokens, completion_tokens, total_tokens), or (None, None).
        """
        if not transcript_segments:
            self.logger.warning(
                f"No transcript segments provided for chapter '{chapter_title}'. Cannot refine."
            )
            return None, None

        model_to_use = (
            model_name_override if model_name_override else self.default_model_name
        )

        system_prompt = (
            "You are an expert audio timestamp detector for audiobooks. Your task is to determine the exact moment a new chapter starts, using the provided transcript segments and their detailed word-level timings.\n"
            f"The transcript segments cover a window of approximately {search_window_seconds:.0f} seconds. "
            "All timestamps within the segments and words are relative to the start of this window (0.0s).\n"
            "Structure of transcript data:\n"
            "- start: segment start (relative to window start)\n"
            "- end: segment end (relative to window start)\n"
            "- text: transcribed text\n"
            "- words: array of {word, start, end, probability} (timestamps relative to window start)\n\n"
            "Analyze for chapter transition markers (explicit announcements, epigraphs, significant pauses, narrative breaks).\n"
            "IMPORTANT: When you identify the start of the chapter (e.g., the word 'Chapter', first word of title/epigraph, or first word after a pause), use the `start` timestamp of **that specific word** as the precise chapter start time within this window.\n"
            f"The original target timestamp within this window is {target_time_seconds:.2f}s. The actual start should be near this time.\n\n"
            "**CRITICAL: Your response MUST be ONLY the final timestamp in seconds (e.g., 123.45) relative to the start of this window. Do NOT include ANY other text, explanations, reasoning, labels, or formatting.**"
        )

        user_prompt = (
            f"Find the precise relative timestamp (within this window) where Chapter '{chapter_title}' starts.\n"
            f"The original target relative timestamp is {target_time_seconds:.2f}s.\n"
            f"Transcript window (total duration: {search_window_seconds:.2f}s):\n\n"
        )
        for segment in transcript_segments:
            user_prompt += f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n"
            if "words" in segment and segment["words"]:
                user_prompt += "  Words:\n"
                for word_info in segment["words"]:
                    user_prompt += f"    - {word_info.get('word', '?')} [{word_info.get('start', -1.0):.3f}s - {word_info.get('end', -1.0):.3f}s]\n"
            user_prompt += "\n"

        user_prompt += "Return ONLY the relative timestamp in seconds."

        llm_response_content, usage_data = self.query_llm(
            system_prompt, user_prompt, model_to_use, max_tokens=20
        )

        if not llm_response_content:
            self.logger.warning(
                f"LLM query failed or returned empty for chapter '{chapter_title}'."
            )
            return None, None

        try:
            parsed_timestamp = float(llm_response_content.strip())
            self.logger.info(
                f"LLM proposed timestamp for '{chapter_title}': {parsed_timestamp:.3f}s (relative to chunk start). Original target: {target_time_seconds:.3f}s."
            )

            # Sanity check: timestamp should be within the chunk boundaries (0 to search_window_seconds)
            # Add a small tolerance (e.g., 0.5 seconds outside search_window_seconds) for LLM responses that might be slightly off
            # if the LLM is confused about the exact end. More importantly, it should not be negative.
            if -0.5 <= parsed_timestamp <= (search_window_seconds + 0.5):
                # Ensure it's not negative after tolerance (clamp to 0 if slightly negative due to tolerance)
                return max(0.0, parsed_timestamp), usage_data
            else:
                self.logger.warning(
                    f"LLM proposed timestamp {parsed_timestamp:.3f}s is outside the plausible chunk window [0, {search_window_seconds:.2f}s] for '{chapter_title}'. Discarding."
                )
                self.logger.debug(
                    f"Out-of-bounds LLM response for '{chapter_title}': '{llm_response_content}'"
                )
                return None, None
        except ValueError:
            self.logger.warning(
                f"Could not parse timestamp from LLM response for '{chapter_title}'. Response: '{llm_response_content}'"
            )
            return None, None

    def query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        max_tokens: int = 50,
    ) -> tuple[Optional[str], Optional[Dict[str, int]]]:
        """
        Query the LLM API using the initialized OpenAI client.
        Args: model_name is the specific model to use.
        Returns: A tuple containing:
                 - Generated text content from the LLM (str, or None on failure).
                 - Token usage dictionary (prompt_tokens, completion_tokens, total_tokens), or None on failure/if not available.
        """
        try:
            self.logger.info(
                f"Sending query to LLM (Model: {model_name}). Max tokens: {max_tokens}."
            )
            # self.logger.debug(f"System Prompt: {system_prompt}") # Can be very verbose
            # self.logger.debug(f"User Prompt (first 500 chars): {user_prompt[:500]}...")

            chat_completion_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            response = self.client.chat.completions.create(**chat_completion_params)

            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                self.logger.debug(f"LLM Raw Response: '{content}'")

                usage_info = None
                if response.usage:
                    usage_info = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    self.logger.debug(f"LLM Usage: {usage_info}")
                else:
                    self.logger.warning("LLM response missing usage data.")

                return content.strip() if content else None, usage_info
            else:
                self.logger.warning("LLM response missing choices or message content.")
                return None, None

        except APIError as e:
            self.logger.error(
                f"OpenAI API Error during LLM query (Model: {model_name}, BaseURL: {self.api_base_url}): {e}"
            )
            if e.status_code == 401:
                self.logger.error("Authentication error: Please check your API key.")
            elif e.status_code == 429:
                self.logger.error(
                    "Rate limit error: You have exceeded your quota or rate limit."
                )
            elif e.status_code == 404:
                self.logger.error(
                    f"Model not found error: The model '{model_name}' may not be available at {self.api_base_url}. Check model name and API endpoint."
                )
            # Other status codes will just log the generic APIError message.
            return None, None
        except Exception as e:
            self.logger.exception(
                f"Unexpected error during LLM query (Model: {model_name}): {e}"
            )
            return None, None
