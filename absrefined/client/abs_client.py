import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Any, Dict, List

import requests
from tqdm import tqdm


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
        except Exception as e:
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

    def download_audio_file(
        self, item_id: str, output_path: str, debug_preserve_files: bool = False
    ) -> str:
        """
        Download the audio for a library item.
        For ZIP files, the contents are extracted and processed.
        For single-file audiobooks, the file is copied directly to the output path.
        For multi-file audiobooks, the audio files are concatenated using ffmpeg.
        """

        # Get item details to determine audio file info
        item_details = self.get_item_details(item_id)
        if not item_details:
            self.logger.error(f"Could not retrieve details for item {item_id}")
            return ""

        audio_files = item_details.get("media", {}).get("audioFiles", [])
        if not audio_files:
            self.logger.error(f"No audio files found for item {item_id}")
            return ""

        if len(audio_files) == 0:
            self.logger.error(f"Empty audio files array for item {item_id}")
            return ""

        # Get the first audio file's info
        audio_file_info = audio_files[0]
        if "ino" not in audio_file_info:
            self.logger.error(
                f"Missing ino field in audio file info for item {item_id}"
            )
            return ""

        audio_ino = audio_file_info["ino"]
        file_url = f"{self.server_url}/api/items/{item_id}/download"

        # Ensure parent directory for output_path exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        zip_processing_dir = None

        try:
            self.logger.info(f"Requesting audio data from {file_url}")
            response = requests.get(
                file_url,
                stream=True,
                timeout=self.request_timeout,
                params={"token": self.api_key},
            )
            response.raise_for_status()

            # Check content type to determine if it's a ZIP file
            content_type = response.headers.get("Content-Type", "")
            is_zip_file = "zip" in content_type.lower()

            self.logger.info(
                f"Content-Type: {content_type}, {'ZIP detected' if is_zip_file else 'non-ZIP file detected'}"
            )

            if not is_zip_file:
                # It's a direct audio file, write it directly to the output path
                self.logger.info(f"Downloading audio file directly to {output_path}...")
                content_length = response.headers.get("Content-Length")

                if content_length and content_length.isdigit():
                    total_size = int(content_length)
                    self.logger.info(f"File size: {total_size / (1024 * 1024):.2f} MB")
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
                    self.logger.info("Downloading audio file (size unknown)...")
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                self.logger.info("Audio file download complete.")
                return output_path

            # ZIP file handling
            temp_dir_parent = os.path.dirname(os.path.abspath(output_path))
            zip_processing_dir = tempfile.mkdtemp(
                prefix=f"abs_zip_{item_id}_", dir=temp_dir_parent
            )
            self.logger.debug(
                f"Using temporary directory for ZIP processing: {zip_processing_dir}"
            )

            downloaded_zip_path = os.path.join(
                zip_processing_dir, f"{item_id}_source.zip"
            )
            extracted_files_dir = os.path.join(zip_processing_dir, "extracted")
            os.makedirs(extracted_files_dir, exist_ok=True)

            # Download ZIP file
            self.logger.info(f"Downloading ZIP file to {downloaded_zip_path}...")
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                total_size = int(content_length)
                self.logger.info(f"ZIP size: {total_size / (1024 * 1024):.2f} MB")
                with (
                    open(downloaded_zip_path, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading ZIP {os.path.basename(downloaded_zip_path)}",
                    ) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                self.logger.info("Downloading ZIP (size unknown)...")
                with open(downloaded_zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            self.logger.info("ZIP file download complete.")

            # Extract ZIP file
            self.logger.info(f"Extracting ZIP file to {extracted_files_dir}...")
            with zipfile.ZipFile(downloaded_zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_files_dir)
            self.logger.info("ZIP extraction complete.")

            # Find audio files in extracted directory
            audio_file_paths = []
            for f_name in sorted(os.listdir(extracted_files_dir)):
                full_f_path = os.path.join(extracted_files_dir, f_name)
                if os.path.isfile(full_f_path) and f_name.lower().endswith(
                    (".mp3", ".m4a", ".ogg", ".wav", ".flac", ".opus", ".m4b")
                ):
                    audio_file_paths.append(os.path.abspath(full_f_path))

            if not audio_file_paths:
                self.logger.error("No audio files found in the extracted ZIP.")
                return ""

            # Update output path to match the extension of the actual audio file
            # This ensures the file extension is correct regardless of what was requested
            if audio_file_paths:
                first_audio_file = audio_file_paths[0]
                _, first_file_ext = os.path.splitext(first_audio_file)
                output_base, output_ext = os.path.splitext(output_path)

                if first_file_ext and first_file_ext != output_ext:
                    new_output_path = f"{output_base}{first_file_ext}"
                    self.logger.info(
                        f"Adjusting output path extension from {output_ext} to {first_file_ext}: {new_output_path}"
                    )
                    output_path = new_output_path

            if len(audio_file_paths) == 1:
                # Optimization for single-file case - just copy the file directly
                self.logger.info(
                    f"Single audio file found. Copying directly to {output_path}"
                )
                shutil.copy(audio_file_paths[0], output_path)
                self.logger.info(
                    f"Successfully copied single audio file to {output_path}"
                )
                return output_path
            else:
                # Multiple audio files need concatenation
                self.logger.info(
                    f"Found {len(audio_file_paths)} audio files for concatenation."
                )

                # Create ffmpeg concat list
                ffmpeg_list_file_path = os.path.join(
                    zip_processing_dir, "ffmpeg_concat_list.txt"
                )
                with open(ffmpeg_list_file_path, "w", encoding="utf-8") as f_list:
                    for audio_path_entry in audio_file_paths:
                        line_content = f"file '{audio_path_entry}'"
                        f_list.write(line_content + "\n")

                # Concatenate audio files using ffmpeg
                self.logger.info(
                    f"Concatenating audio files to {output_path} using ffmpeg..."
                )
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    ffmpeg_list_file_path,
                    "-map",
                    "0a",
                    "-c:a",
                    "acc",
                    "-ac",
                    "2",
                    "-b:a",
                    "128k",
                    output_path,
                ]
                self.logger.debug(f"Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")
                process = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    encoding="utf-8",
                )

                if process.returncode != 0:
                    self.logger.error(
                        f"ffmpeg concatenation failed. Return code: {process.returncode}"
                    )
                    self.logger.error(
                        f"ffmpeg stdout: {process.stdout.strip() if process.stdout else '[No stdout]'}"
                    )
                    self.logger.error(
                        f"ffmpeg stderr: {process.stderr.strip() if process.stderr else '[No stderr]'}"
                    )
                    return ""

                self.logger.info("ffmpeg concatenation successful.")
                return output_path

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Error during audio download/processing for {item_id}: {e}"
            )
            return ""
        except zipfile.BadZipFile as e:
            self.logger.error(f"Error extracting ZIP file for {item_id}: {e}")
            return ""
        except Exception as e:
            self.logger.error(
                f"Unexpected error during download/processing for {item_id}: {e}"
            )
            self.logger.exception("Details of unexpected error:")
            return ""
        finally:
            # Cleanup of temporary directory for ZIP processing
            if zip_processing_dir and os.path.exists(zip_processing_dir):
                if debug_preserve_files:
                    self.logger.info(
                        f"Debug mode: Preserving temporary ZIP processing directory: {zip_processing_dir}"
                    )
                else:
                    self.logger.debug(
                        f"Cleaning up temporary ZIP processing directory: {zip_processing_dir}"
                    )
                    try:
                        shutil.rmtree(zip_processing_dir)
                    except Exception as e_clean:
                        self.logger.error(
                            f"Failed to clean up temporary directory {zip_processing_dir}: {e_clean}"
                        )

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

        update_url = f"{self.server_url}/api/items/{item_id}/chapters"
        self.logger.info(
            f"Sending {len(final_chapters_list)} chapter start time updates for item {item_id} to {update_url}"
        )
        payload = {"chapters": final_chapters_list}
        self.logger.debug(f"Bulk update payload: {payload}")

        headers = self._get_auth_headers()
        headers["Content-Type"] = "application/json"

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
