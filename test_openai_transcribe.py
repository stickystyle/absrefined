import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

def test_transcription(audio_file_path: str):
    """Transcribes a single audio file using OpenAI API mimicking AudioTranscriber."""
    
    print(f"Attempting to transcribe: {audio_file_path}")

    if not os.path.exists(audio_file_path):
        print(f"Error: File not found at {audio_file_path}")
        return

    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        return

    try:
        # Initialize client
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized.")

        # Open file and make API call
        with open(audio_file_path, "rb") as audio_data:
            print("Audio file opened in binary read mode.")
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            print("API call successful.")

        # Print the raw response (or specific parts)
        print("\n--- API Response Object ---")
        # print(response) # Print the full response object
        print("-------------------------")
        # print(f"Text: {response.text}")
        print(f"Segments: {response.words}")
        # Add a check before accessing len()
        # if response.segments is not None:
        #     print(f"Segments found: {len(response.segments)}")
        #     if response.segments:
        #         print("First segment words:", response.segments[0].words)
        #     else:
        #         print("Segments list is empty.")
        # else:
        #      print("response.segments is None.") # Explicitly state if it's None

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_openai_transcribe.py <path_to_audio_file.wav>")
    else:
        test_transcription(sys.argv[1]) 