import re
from typing import Dict, List, Optional
import logging
from openai import OpenAI

from absrefined.utils.timestamp import format_timestamp


class ChapterRefiner:
    """Class for refining chapter markers using LLM analysis.

    IMPORTANT: This class treats ALL endpoints as OpenAI-compatible API endpoints.
    It should not have special handling for different providers (like OpenRouter, Anthropic, etc).
    All endpoints are expected to follow the OpenAI API format, and authentication should
    be handled the same way regardless of the endpoint domain.

    DO NOT add endpoint-specific logic or conditional handling based on the API URL.
    """

    def __init__(
        self,
        api_base: str,
        model: str = "gpt-4o",
        window_size: int = 15,
        verbose: bool = False,
        llm_api_key: str = None,
    ):
        """
        Initialize the chapter refiner.

        Args:
            api_base (str): Base URL for the OpenAI-compatible API
            model (str): Model to use
            window_size (int): Window size in seconds around chapter markers
            verbose (bool): Whether to print verbose output
            llm_api_key (str): API key for the LLM service
        """
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.window_size = window_size
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use the API key passed to the constructor
        self.api_key = llm_api_key

        if not self.api_key:
            self.logger.warning(f"No API key provided for {self.api_base}")
            self.logger.warning(
                "Please provide an API key when initializing ChapterRefiner"
            )
        else:
            self.verify_api_key()

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def verify_api_key(self):
        """Verify that the API key is properly set and formatted."""
        if not self.api_key:
            self.logger.error("No API key is set - LLM queries will fail")
            return False

        # Check if the API key looks valid (basic format check)
        if len(self.api_key) < 8:  # Most API keys are longer than this
            self.logger.warning(
                f"WARNING: API key seems too short ({len(self.api_key)} chars) - it may be invalid"
            )
            return False

        self.logger.debug(
            f"API key verification: API key is set (length: {len(self.api_key)})"
        )

        return True

    def detect_chapter_start(
        self, transcript: List[Dict], chapter_name: str, orig_timestamp: float
    ) -> Optional[Dict]:
        """
        Detect the precise start of a chapter in a transcript.

        Args:
            transcript (List[Dict]): Transcript segments
            chapter_name (str): Name of the chapter
            orig_timestamp (float): Original timestamp for reference

        Returns:
            Optional[Dict]: Detected chapter start info with timestamp
        """
        if not transcript:
            self.logger.debug(
                f"No transcript segments provided for chapter '{chapter_name}'"
            )
            return None

        # Filter transcript segments around the expected timestamp
        # Use the configured window size around the expected timestamp
        window_start = max(0, orig_timestamp - self.window_size)
        window_end = orig_timestamp + self.window_size

        # Filter segments in the window
        relevant_segments = [
            s
            for s in transcript
            if (s["start"] >= window_start and s["start"] <= window_end)
            or (s["end"] >= window_start and s["end"] <= window_end)
            or (s["start"] <= window_start and s["end"] >= window_end)
        ]

        if not relevant_segments:
            self.logger.debug(
                f"No transcript segments found around timestamp {format_timestamp(orig_timestamp)}"
            )
            return None

        # Prepare system prompt
        system_prompt = (
            "You are an expert audio timestamp detector for audiobooks. Your task is to determine the exact moment a new chapter starts, using the provided transcript segments and their detailed word-level timings.\n"
            "The transcript segments have this structure:\n"
            "- start: timestamp in seconds when segment begins\n"
            "- end: timestamp in seconds when segment ends\n"
            "- text: transcribed text content\n"
            "- words: array of {word, start, end, probability} objects for precise word-level timing\n\n"
            "Analyze the transcript, paying close attention to the `words` data, for these chapter transition markers:\n"
            "1. Explicit announcements: 'Chapter X', the chapter title itself, or section markers like 'Part X', 'Section Y'.\n"
            "2. Epigraphs: Quoted passages preceding the main chapter content. Often follow a pattern of 'quotation → attribution → narrative start'.\n"
            "3. Significant pauses: Look for noticeable time gaps between the `end` timestamp of the last word of one section and the `start` timestamp of the first word of the new section.\n"
            "4. Clear narrative breaks: Identify where the previous narrative concludes and a distinct new section begins, even if not explicitly announced.\n\n"
            "IMPORTANT: When you identify the start of the chapter (e.g., the word 'Chapter', the first word of the title, the first word of the epigraph, or the first word after a definitive pause), **use the `start` timestamp of that specific word** as the precise chapter start time.\n\n"
            f"The original timestamp is approximate. The actual chapter start should be within {self.window_size} seconds of this time.\n\n"
            "**CRITICAL: Your response MUST be ONLY the final timestamp in seconds (e.g., 123.45). Do NOT include ANY other text, explanations, reasoning, labels, or formatting.**"
        )

        # Build user prompt with the transcript and chapter info
        user_prompt = (
            f"Find the precise timestamp where Chapter '{chapter_name}' starts in this audio transcript.\n\n"
            f"The original approximate timestamp is {orig_timestamp}.\n\n"
            f"Here is the transcript of the audio around this timestamp (±{self.window_size} seconds):\n\n"
        )

        # Add segments with their timestamps AND words for better context
        for segment in relevant_segments:
            user_prompt += f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n"
            # Include words if they exist
            if 'words' in segment and segment['words']:
                user_prompt += "  Words:\n"
                for word_info in segment['words']:
                    word_text = word_info.get('word', '?')
                    word_start = word_info.get('start', -1.0)
                    word_end = word_info.get('end', -1.0)
                    user_prompt += f"    - {word_text} [{word_start:.3f}s - {word_end:.3f}s]\n"
            user_prompt += "\n" # Add a blank line between segments for readability

        user_prompt += (
            "\nReturn ONLY the timestamp in seconds where the chapter starts."
        )

        # Call the LLM to analyze the transcript
        response = self.query_llm(system_prompt, user_prompt, max_tokens=50)
        if not response:
            self.logger.debug("Failed to get response from LLM")
            return None

        # Parse the timestamp from the response
        # First check if the response is already a number
        if re.match(r"^\d+\.?\d*$", response):
            timestamp = float(response)
        else:
            # Try to extract a number from the response
            timestamp_match = re.search(r"(\d+\.\d+|\d+)", response)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
            else:
                self.logger.debug(
                    f"Failed to extract timestamp from LLM response: {response}"
                )
                return None

        # Create the result with just the timestamp
        result = {"timestamp": timestamp}

        self.logger.debug(f"Detected chapter start for '{chapter_name}':")
        self.logger.debug(f"  Original timestamp: {format_timestamp(orig_timestamp)}")
        self.logger.debug(f"  Detected timestamp: {format_timestamp(timestamp)}")

        return result

    def query_llm(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 50
    ) -> str:
        """
        Query the LLM API using OpenAI client.

        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: Generated text
        """
        # Verify API key before proceeding
        if not self.verify_api_key():
            self.logger.error("Skipping LLM query due to missing or invalid API key")
            return ""

        try:
            self.logger.debug("Sending query to OpenAI API using client")
            self.logger.debug(f"Model: {self.model}")

            # Prepare the base parameters that work with all models
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            # Check if the model is a reasoning model that doesn't support temperature/max_tokens
            is_reasoning_model = any(
                model_name in self.model for model_name in ["o3-mini", "o3"]
            )

            # Only add temperature and max_tokens for models that support them
            if not is_reasoning_model:
                params["temperature"] = (
                    0.1  # Keep temperature low for consistent results
                )
                params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**params)

            # Extract content from the response
            content = response.choices[0].message.content

            # Save raw content for debugging
            raw_content = content

            # Clean up the response - strip any thinking sections and special commands
            clean_content = content

            # Handle <think> tags that don't have a closing tag
            if clean_content.startswith("<think>"):
                # Try to extract any number from the thinking section
                # This is useful for timestamp extraction when the response is incomplete
                numbers = re.findall(r"(\d+\.\d+|\d+)", clean_content)
                if numbers:
                    # For timestamp extraction, just return the first number found
                    self.logger.debug(
                        f"Extracted number from incomplete thinking: {numbers[0]}"
                    )
                    return numbers[0]
                elif "</think>" not in clean_content:
                    # Just remove the entire think block since it's incomplete
                    clean_content = ""

            # Remove complete think blocks
            thinking_pattern = r"<think>.*?</think>"
            clean_content = re.sub(thinking_pattern, "", clean_content, flags=re.DOTALL)

            # Remove any special commands
            for cmd in ["/no_think", "/think"]:
                clean_content = clean_content.replace(cmd, "")

            # Final cleanup - focus on extracting just numbers if that's all we need
            clean_content = clean_content.strip()

            # If we just need a timestamp, try to extract any number
            if not clean_content or not re.match(r"^\d+\.?\d*$", clean_content):
                # Try to extract any number from the entire response
                numbers = re.findall(r"(\d+\.\d+|\d+)", raw_content)
                if numbers:
                    # Prioritize numbers that appear after "Timestamp:" if present
                    timestamp_numbers = re.findall(
                        r"Timestamp:\s*(\d+\.?\d*)", raw_content, re.IGNORECASE
                    )
                    if timestamp_numbers:
                        self.logger.debug(
                            f"Extracted timestamp number from response: {timestamp_numbers[0]}"
                        )
                        return f"Timestamp: {timestamp_numbers[0]}"
                    else:
                        self.logger.debug(
                            f"Clean content not numeric, extracted number from raw response: {numbers[0]}"
                        )
                        # Return in the expected format to help downstream processing
                        return f"Timestamp: {numbers[0]}"

            self.logger.debug(f"Received response from LLM: '{clean_content}'")
            if not clean_content and raw_content:
                self.logger.debug(f"Raw response before cleaning: '{raw_content}'")

            return clean_content

        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            return ""
