import os
import requests
from typing import Dict, List, Any
from tqdm import tqdm
import logging


class AudiobookshelfClient:
    """Client for interacting with the Audiobookshelf API."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Audiobookshelf client using a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing
                                     audiobookshelf host, api_key, and optional timeout.

        Raises:
            KeyError: If required configuration keys ('host', 'api_key') are missing.
        """
        abs_config = config.get("audiobookshelf", {})

        self.server_url = abs_config.get("host")
        if not self.server_url:
            raise KeyError(
                "Audiobookshelf host not found in configuration ('audiobookshelf.host')"
            )
        self.server_url = self.server_url.rstrip("/")

        self.api_key = abs_config.get("api_key")
        if not self.api_key:
            raise KeyError(
                "Audiobookshelf API key not found in configuration ('audiobookshelf.api_key')"
            )

        self.request_timeout = abs_config.get("timeout", 30)  # Default to 30 seconds

        # self.user_id = None # Removed unused instance variable
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"AudiobookshelfClient initialized for host: {self.server_url}"
        )

    def _get_auth_headers(self) -> Dict[str, str]:
        """Returns the authorization headers for API requests."""
        if not self.api_key:
            # This should ideally be caught during __init__
            self.logger.error("API key is not set. Cannot make authenticated requests.")
            raise ValueError("API key not configured. Client not properly initialized.")
        return {"Authorization": f"Bearer {self.api_key}"}

    def get_item_details(self, item_id: str) -> Dict:
        """
        Get details for a library item.
        """
        details_url = f"{self.server_url}/api/items/{item_id}"
        headers = self._get_auth_headers()

        try:
            self.logger.debug(f"Fetching details for item {item_id} from {details_url}")
            response = requests.get(
                details_url, headers=headers, timeout=self.request_timeout
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

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
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting item details for {item_id}: {e}")
            return {}  # Or raise a custom exception
        except Exception as e:  # Catch other unexpected errors
            self.logger.error(
                f"Unexpected error getting item details for {item_id}: {e}"
            )
            return {}

    def get_item_chapters(self, item_id: str) -> List[Dict]:
        """
        Get chapters for a library item.
        """
        item_details = self.get_item_details(item_id)
        if not item_details:
            return []
        return item_details.get("media", {}).get("chapters", [])

    def download_audio_file(self, item_id: str, output_path: str) -> str:
        """
        Download the complete audio file from a library item.
        """
        details = self.get_item_details(item_id)
        if not details:
            self.logger.error(
                f"Failed to get item details for {item_id}, cannot download."
            )
            return ""

        audio_files_info = details.get("media", {}).get("audioFiles", [])
        if not audio_files_info:
            self.logger.warning(
                f"No audio tracks (audioFiles) found in media for item {item_id}"
            )
            # Removed commented-out legacy 'tracks' lookup
            return ""

        self.logger.debug(f"Found {len(audio_files_info)} audio file entries.")
        # Assuming the first audio file is the one to download, or that they are concatenated.
        # The API structure seems to point to 'ino' for the download URL part.

        # Prefer audioFiles structure if available
        if audio_files_info and "ino" in audio_files_info[0]:
            file_ino = audio_files_info[0].get("ino")
        else:
            self.logger.error(
                f"Could not determine 'ino' for audio download from item {item_id}"
            )
            return ""

        if not file_ino:
            self.logger.error(f"Audio file 'ino' is missing for item {item_id}")
            return ""

        file_url = f"{self.server_url}/api/items/{item_id}/file/{file_ino}"
        headers = self._get_auth_headers()

        try:
            self.logger.info(
                f"Downloading complete audio file from {file_url} to {output_path}"
            )
            response = requests.get(
                file_url, headers=headers, stream=True, timeout=self.request_timeout
            )
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            if content_length and content_length.isdigit():
                total_size = int(content_length)
                self.logger.info(
                    f"Downloading approximately {total_size / (1024 * 1024):.2f} MB"
                )
                with (
                    open(output_path, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {os.path.basename(output_path)}",
                    ) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                self.logger.info("Downloading audio (size unknown)...")
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            self.logger.info(f"Successfully downloaded audio file to {output_path}")
            return output_path
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading audio for {item_id}: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"Unexpected error downloading audio for {item_id}: {e}")
            return ""

    # Removed unused _fix_chapter_boundaries method

    def update_chapters_start_time(
        self, item_id: str, chapter_updates: List[Dict]
    ) -> bool:
        """
        Update start times for a list of chapters by their IDs.
        This fetches all existing chapters, updates the relevant ones, and then PUTs all chapters back.
        Audiobookshelf's chapter update API typically requires the full set of chapters.

        Args:
            item_id (str): ID of the library item.
            chapter_updates (List[Dict]): A list of dictionaries, each with "id" and new "start" time.
                                         Example: [{'id': 'chapter_id_1', 'start': 123.45}, ...]
        Returns:
            bool: True if successful, False otherwise.
        """
        self.logger.info(
            f"Attempting to update start times for {len(chapter_updates)} chapters in item {item_id}."
        )

        current_chapters = self.get_item_chapters(item_id)
        if not current_chapters:
            self.logger.error(
                f"Could not retrieve current chapters for item {item_id}. Cannot update start times."
            )
            return False

        self.logger.debug(f"Retrieved {len(current_chapters)} current chapters.")

        updated_count = 0
        chapters_map = {chap["id"]: chap for chap in current_chapters}

        for update in chapter_updates:
            chap_id = update.get("id")
            new_start_time = update.get("start")

            if not chap_id or new_start_time is None:
                self.logger.warning(
                    f"Skipping update due to missing id or start time: {update}"
                )
                continue

            if chap_id in chapters_map:
                try:
                    new_start_float = float(new_start_time)
                    if chapters_map[chap_id]["start"] != new_start_float:
                        self.logger.debug(
                            f"Updating chapter ID {chap_id}: old_start={chapters_map[chap_id]['start']}, new_start={new_start_float}"
                        )
                        chapters_map[chap_id]["start"] = new_start_float
                        updated_count += 1
                    else:
                        self.logger.debug(
                            f"Chapter ID {chap_id}: start time {new_start_float} is already set. No change."
                        )
                except ValueError:
                    self.logger.warning(
                        f"Invalid start time format for chapter ID {chap_id}: {new_start_time}. Skipping."
                    )
            else:
                self.logger.warning(
                    f"Chapter ID {chap_id} not found in current chapters. Skipping update for this chapter."
                )

        if updated_count == 0:
            self.logger.info("No actual changes to chapter start times were needed.")
            # Depending on desired behavior, this could be True (no failure) or False (no update made)
            # Let's return True as the operation didn't fail, even if no data changed.
            return True

        # Convert map back to list for updating
        final_chapters_list = sorted(
            list(chapters_map.values()), key=lambda x: x["start"]
        )

        # The ABS API typically requires the 'end' field.
        # We need to recalculate end times based on the new start times.
        # The last chapter's end time should ideally be the media duration.
        # This is a complex part if media duration isn't easily available or if overlaps are an issue.

        # Simple fix for end times: next chapter's start. Last chapter's end remains.
        # This might not be perfect if the server doesn't auto-adjust the last chapter's end.
        for i in range(len(final_chapters_list) - 1):
            final_chapters_list[i]["end"] = final_chapters_list[i + 1]["start"]

        # If the last chapter's 'end' needs to be the total duration, that info is missing here.
        # The server might handle this. If not, the last chapter's 'end' might be incorrect.
        # Removed commented-out debug log

        update_url = f"{self.server_url}/api/items/{item_id}/chapters"  # Removed redundant comment

        self.logger.info(
            f"Sending {len(final_chapters_list)} chapter start time updates for item {item_id} to {update_url}"
        )
        payload = {"chapters": final_chapters_list}
        self.logger.debug(f"Bulk update payload: {payload}")

        headers = self._get_auth_headers()  # Removed redundant comment
        headers["Content-Type"] = "application/json"  # Add Content-Type

        self.logger.debug(
            f"CONFIRMING: About to call POST on URL: {update_url} with payload: {payload}"
        )

        try:
            response = requests.post(
                update_url, headers=headers, json=payload, timeout=self.request_timeout
            )
            response.raise_for_status()

            self.logger.info(f"Chapters updated successfully for item {item_id}.")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error updating chapters for item {item_id}: {e}")
            if hasattr(e, "response") and e.response is not None:
                self.logger.error(f"Response content: {e.response.text}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error updating chapters for {item_id}: {e}")
            return False
