import os
import requests
from typing import Dict, List
from tqdm import tqdm
import logging

# Will be imported from utils
from absrefined.utils.timestamp import format_timestamp, parse_timestamp


class AudiobookshelfClient:
    """Client for interacting with the Audiobookshelf API."""

    def __init__(self, server_url: str, verbose: bool = False):
        """
        Initialize the Audiobookshelf client.

        Args:
            server_url (str): URL of the Audiobookshelf server
            verbose (bool): Whether to print verbose output
        """
        self.server_url = server_url.rstrip("/")
        self.verbose = verbose
        self.token = None
        self.user_id = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def login(self, username: str, password: str) -> bool:
        """
        Login to the Audiobookshelf server.

        Args:
            username (str): Username for authentication
            password (str): Password for authentication

        Returns:
            bool: Whether login was successful
        """
        login_url = f"{self.server_url}/login"
        payload = {"username": username, "password": password}

        try:
            self.logger.debug(f"Attempting to login to {login_url}")
            response = requests.post(login_url, json=payload)
            self.logger.debug(f"Login response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("user", {}).get("token")
                self.user_id = data.get("user", {}).get("id")
                self.logger.debug(f"Login successful for user {username}")
                self.logger.debug(f"Token received: {self.token is not None}")
                if self.token:
                    self.logger.debug(f"Token length: {len(self.token)}")
                return True
            else:
                self.logger.debug(
                    f"Login failed with status code {response.status_code}: {response.text}"
                )
                return False
        except Exception as e:
            self.logger.debug(f"Error during login: {e}")
            return False

    def get_item_details(self, item_id: str) -> Dict:
        """
        Get details for a library item.

        Args:
            item_id (str): ID of the library item

        Returns:
            Dict: Library item details
        """
        if not self.token:
            raise ValueError("Not authenticated. Call login() first.")

        details_url = f"{self.server_url}/api/items/{item_id}"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            self.logger.debug(f"Fetching details for item {item_id}")
            response = requests.get(details_url, headers=headers)

            if response.status_code == 200:
                details = response.json()

                self.logger.debug("Item details structure:")
                self.logger.debug(f"Keys at root level: {list(details.keys())}")
                if "media" in details:
                    self.logger.debug(
                        f"Media section keys: {list(details['media'].keys())}"
                    )
                else:
                    self.logger.debug("No 'media' key found in response")

                # Let's inspect other potentially relevant sections
                if "mediaType" in details:
                    self.logger.debug(f"Media type: {details['mediaType']}")
                if "path" in details:
                    self.logger.debug(f"Path: {details['path']}")
                if "audioFiles" in details:
                    self.logger.debug(
                        f"Audio files count: {len(details.get('audioFiles', []))}"
                    )

                return details
            else:
                self.logger.debug(
                    f"Failed to get item details: {response.status_code}: {response.text}"
                )
                return {}

        except Exception as e:
            self.logger.debug(f"Error getting item details: {e}")
            return {}

    def get_item_chapters(self, item_id: str) -> List[Dict]:
        """
        Get chapters for a library item.

        Args:
            item_id (str): ID of the library item

        Returns:
            List[Dict]: List of chapters
        """
        item_details = self.get_item_details(item_id)

        if not item_details:
            return []

        # Extract chapters from the response
        return item_details.get("media", {}).get("chapters", [])

    def download_audio_file(self, item_id: str, output_path: str) -> str:
        """
        Download the complete audio file from a library item.

        Args:
            item_id (str): ID of the library item
            output_path (str): Path to save the audio file

        Returns:
            str: Path to the saved audio file
        """
        if not self.token:
            raise ValueError("Not authenticated. Call login() first.")

        # First, get the item details to find audio files
        details = self.get_item_details(item_id)

        if not details:
            self.logger.debug(f"Failed to get item details for {item_id}")
            return ""

        # Extract audio files information
        audio_files = []
        if "media" in details and "audioFiles" in details["media"]:
            audio_files = details["media"]["audioFiles"]

        if not audio_files:
            self.logger.debug(f"No audio tracks found for item {item_id}")
            return ""

        self.logger.debug(f"Found {len(audio_files)} audio files:")
        for i, af in enumerate(audio_files):
            self.logger.debug(
                f"  {i + 1}. Index: {af.get('index')}, Path: {af.get('relPath')}, Duration: {af.get('duration')}"
            )

        # Use the first audio file
        audio_file = audio_files[0]
        file_id = audio_file.get("ino")

        if not file_id:
            self.logger.debug("Audio file has no ID (ino)")
            return ""

        # Construct the stream request URL
        file_url = f"{self.server_url}/api/items/{item_id}/file/{file_id}"
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            self.logger.debug(f"Downloading complete audio file from {file_url}")
            self.logger.debug("This may take a while depending on the file size...")

            response = requests.get(file_url, headers=headers, stream=True)

            if response.status_code == 200:
                content_length = response.headers.get("Content-Length", "unknown")
                self.logger.debug(f"Response status: {response.status_code}")
                self.logger.debug(f"Content-Length: {content_length} bytes")
                if content_length != "unknown" and content_length.isdigit():
                    self.logger.debug(
                        f"Downloading approximately {int(content_length) / 1024 / 1024:.2f} MB"
                    )

                # Ensure output directory exists
                os.makedirs(
                    os.path.dirname(os.path.abspath(output_path)), exist_ok=True
                )

                # Use tqdm to show download progress if verbose
                if content_length != "unknown" and content_length.isdigit():
                    total_size = int(content_length)
                    with open(output_path, "wb") as f:
                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            desc="Downloading",
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                else:
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                self.logger.debug(
                    f"Successfully downloaded audio file to {output_path}"
                )
                file_size = os.path.getsize(output_path)
                self.logger.debug(
                    f"File size on disk: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)"
                )

                return output_path
            else:
                self.logger.debug(
                    f"Failed to download audio: {response.status_code}: {response.text}"
                )
                return ""

        except Exception as e:
            self.logger.debug(f"Error downloading audio: {e}")
            return ""

    def stream_audio_segment(
        self, item_id: str, start_time: float, end_time: float, output_path: str
    ) -> str:
        """
        Stream a segment of audio from a library item.
        This method downloads the full file if it doesn't exist yet.

        Args:
            item_id (str): ID of the library item
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            output_path (str): Path to save the audio segment

        Returns:
            str: Path to the full audio file
        """
        # Use a cached file path based on the item_id
        temp_dir = os.path.dirname(output_path)
        full_file_path = os.path.join(temp_dir, f"{item_id}_full.m4a")

        # Download the full file if it doesn't exist
        if not os.path.exists(full_file_path):
            self.logger.debug(
                f"Full audio file not found. Downloading to {full_file_path}"
            )

            download_result = self.download_audio_file(item_id, full_file_path)
            if not download_result:
                self.logger.debug("Failed to download full audio file")
                return ""
        else:
            self.logger.debug(f"Using cached full audio file: {full_file_path}")

        # Return the full file path instead of extracting a segment
        return full_file_path

    def update_item_chapters(self, item_id: str, chapters: List[Dict]) -> bool:
        """
        Update chapters for a library item.

        Args:
            item_id (str): ID of the library item
            chapters (List[Dict]): List of chapters to update

        Returns:
            bool: Whether update was successful
        """
        if not self.token:
            raise ValueError("Not authenticated. Call login() first.")

        # Ensure chapters have the correct format
        processed_chapters = []
        for i, chapter in enumerate(chapters):
            # Create a copy to avoid modifying the original
            ch = chapter.copy()

            # Ensure 'start' is a float, not a string
            if "start" in ch and isinstance(ch["start"], str):
                try:
                    ch["start"] = float(ch["start"].replace(":", ".").replace(",", "."))
                except ValueError:
                    # Try to convert from timestamp format like "01:23:45.678"
                    from absrefined.utils.timestamp import parse_timestamp

                    ch["start"] = parse_timestamp(ch["start"])

            # Ensure 'end' is a float, not a string
            if "end" in ch and isinstance(ch["end"], str):
                try:
                    ch["end"] = float(ch["end"].replace(":", ".").replace(",", "."))
                except ValueError:
                    # Try to convert from timestamp format
                    from absrefined.utils.timestamp import parse_timestamp

                    ch["end"] = parse_timestamp(ch["end"])

            # Force first chapter to start at 0
            if i == 0:
                ch["start"] = 0.0

            # Remove any custom fields that shouldn't be sent to the server
            if "refined" in ch:
                del ch["refined"]

            # Add to processed chapters
            processed_chapters.append(ch)

        # Fix chapter boundaries to ensure end times match the start times of next chapters
        self._fix_chapter_boundaries(processed_chapters)

        # Try the standard endpoint first
        chapters_url = f"{self.server_url}/api/items/{item_id}/chapters"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {"chapters": processed_chapters}

        try:
            self.logger.debug(f"Updating chapters for item {item_id}")
            self.logger.debug(f"POST URL: {chapters_url}")
            self.logger.debug(f"POST Headers: {headers}")
            self.logger.debug(f"POST Payload: {payload}")
            self.logger.debug(
                f"Number of chapters in payload: {len(processed_chapters)}"
            )
            if processed_chapters:
                self.logger.debug(f"First chapter structure: {processed_chapters[0]}")
                self.logger.debug(f"Last chapter structure: {processed_chapters[-1]}")
                self.logger.debug(f"Chapter keys: {list(processed_chapters[0].keys())}")

            response = requests.post(chapters_url, json=payload, headers=headers)

            if response.status_code == 200:
                self.logger.debug(f"Successfully updated chapters for item {item_id}")
                return True
            else:
                self.logger.debug(
                    f"Failed to update chapters: {response.status_code}: {response.text}"
                )
                return False
        except Exception as e:
            self.logger.debug(f"Error updating chapters: {e}")
            return False

    def _fix_chapter_boundaries(self, chapters: List[Dict]) -> None:
        """
        Fix chapter boundaries by setting each chapter's end time to the start time of the next chapter.
        For the last chapter, we keep its existing end time as it should match the end of the book.

        Args:
            chapters (List[Dict]): Chapters to fix boundaries for
        """
        if not chapters or len(chapters) < 2:
            return

        # Fix the end times for all chapters except the last one
        for i in range(len(chapters) - 1):
            next_start = chapters[i + 1]["start"]
            chapters[i]["end"] = next_start

        # The last chapter's end time should already be set to the end of the book

    def process_chapter(
        self,
        item_id: str,
        chapter: Dict,
        index: int,
        total: int,
        full_audio_path: str,
        transcriber=None,
    ) -> Dict:
        """
        Process a chapter and refine its timestamp if needed.

        Args:
            item_id (str): ID of the library item
            chapter (Dict): Chapter data
            index (int): Index of the chapter in the list (0-based)
            total (int): Total number of chapters
            full_audio_path (str): Path to the full audio file if downloaded
            transcriber (AudioTranscriber, optional): Instance of AudioTranscriber for transcription

        Returns:
            Dict: Updated chapter data
        """
        chapter_title = chapter.get("title", f"Chapter {index + 1}")
        chapter_time = parse_timestamp(chapter.get("start", "0"))

        self.logger.debug(f"Processing chapter {index + 1}/{total}: '{chapter_title}'")
        self.logger.debug(f"  Original timestamp: {format_timestamp(chapter_time)}")

        # Set transcription file path for the full audio
        transcript_filename = f"{item_id}_full_audio.jsonl"
        transcript_path = os.path.join(
            os.path.dirname(full_audio_path), transcript_filename
        )

        # Transcribe the full audio if transcriber is available and transcription doesn't exist
        if transcriber and full_audio_path and os.path.exists(full_audio_path):
            # Check if transcription already exists
            if os.path.exists(transcript_path):
                self.logger.debug(
                    f"  Reading existing transcription from {transcript_path}"
                )
                segments = transcriber.read_transcription(transcript_path)
            else:
                self.logger.debug(f"  Transcribing full audio file: {full_audio_path}")
                segments = transcriber.transcribe_audio(
                    full_audio_path, transcript_path
                )

            if not segments:
                self.logger.debug(
                    f"  No transcription segments found for chapter '{chapter_title}', keeping original timestamp"
                )
                return chapter

            # If refiner is attached to this client, use it to refine the timestamp
            if hasattr(self, "refiner") and self.refiner:
                self.logger.debug(
                    f"  Using LLM to refine chapter timestamp for '{chapter_title}'"
                )

                # Call the refiner to detect chapter start
                detection_result = self.refiner.detect_chapter_start(
                    segments, chapter_title, chapter_time
                )

                if detection_result:
                    new_timestamp = detection_result.get("timestamp")

                    self.logger.debug(
                        f"  Detected timestamp: {format_timestamp(new_timestamp)}"
                    )

                    # Create new chapter dict with updated timestamp
                    updated_chapter = chapter.copy()
                    updated_chapter["start"] = format_timestamp(new_timestamp)
                    updated_chapter["refined"] = True
                    return updated_chapter

        # If we couldn't refine, return the original chapter
        return chapter
