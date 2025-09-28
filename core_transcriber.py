import tempfile
import os
import subprocess
import whisper
import torch
from pathlib import Path
import requests
import glob

_whisper_model = None

def load_whisper_model():
    """
    Loads the Whisper model, caching it.
    """
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...") # Add print for visibility outside Streamlit
        _whisper_model = whisper.load_model("base")
        print("Whisper model loaded.") # Add print for visibility outside Streamlit
    return _whisper_model

def extract_audio_ffmpeg(video_path, output_wav_path):
    """
    Extract audio from video using ffmpeg, convert to mono 16kHz WAV.
    Raises RuntimeError if ffmpeg fails.
    """
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn", output_wav_path
    ]
    print(f"Running ffmpeg command: {' '.join(command)}") # Add print for visibility
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ffmpeg stdout: {result.stdout.decode()}") # Add print for debugging
        print(f"ffmpeg stderr: {result.stderr.decode()}") # Add print for debugging
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
    print("ffmpeg extraction successful.") # Add print for visibility


def is_youtube_url(url):
    """
    Checks if a URL is a YouTube URL.
    """
    return any(domain in url for domain in ["youtube.com", "youtu.be"])

def download_video(url):
    video_file_path = None
    try:
        print(f"Attempting to download video from: {url}") # Add print for visibility
        if is_youtube_url(url):
            with tempfile.TemporaryDirectory() as temp_dir:
                yt_dlp_cmd = [
                    "yt-dlp", "-f", "best[ext=mp4]/best",
                    "-o", os.path.join(temp_dir, "%(title)s.%(ext)s"),
                    url
                ]
                result = subprocess.run(yt_dlp_cmd, cwd=temp_dir, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"yt-dlp stdout: {result.stdout}") # Add print for debugging
                    print(f"yt-dlp stderr: {result.stderr}") # Add print for debugging
                    raise RuntimeError(f"yt-dlp error: {result.stderr}")

                files = glob.glob(os.path.join(temp_dir, "*"))
                if not files:
                    raise RuntimeError("yt-dlp did not produce any output file.")
                downloaded_file_path = files[0]

                # Copy to a new temp file to persist after exiting the context
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(downloaded_file_path).suffix) as dst:
                    with open(downloaded_file_path, "rb") as src:
                        dst.write(src.read())
                    video_file_path = dst.name
        else:
            response = requests.get(url, stream=True, timeout=60) # Increased timeout
            response.raise_for_status()
            ext = Path(url).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_video:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)
                video_file_path = temp_video.name

        if not video_file_path or not os.path.exists(video_file_path) or os.path.getsize(video_file_path) == 0:
             raise RuntimeError("Downloaded video file is missing or empty.")

        print(f"Video downloaded successfully to: {video_file_path}") # Add print for visibility
        return video_file_path

    except Exception as e:
        # Clean up potentially created temp file if download failed midway
        if video_file_path and os.path.exists(video_file_path):
             os.remove(video_file_path)
        raise RuntimeError(f"Failed to download video: {e}")


def transcribe_video(video_source):
    video_file_path = None
    temp_wav_path = None
    transcription = None

    try:
        if os.path.exists(video_source):
            video_file_path = video_source
            print(f"Using local video file: {video_file_path}") # Add print for visibility
        elif video_source.startswith("http://") or video_source.startswith("https://"):
            video_file_path = download_video(video_source)
        else:
            raise FileNotFoundError(f"Video source not found or is not a valid URL: {video_source}")

        # Check file exists and is not empty before processing
        if not os.path.exists(video_file_path) or os.path.getsize(video_file_path) == 0:
            raise FileNotFoundError(f"Video file is missing or empty: {video_file_path}")

        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name

        print("Extracting audio...") # Add print for visibility
        extract_audio_ffmpeg(video_file_path, temp_wav_path)

        # Check if audio extraction resulted in a non-empty file
        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
             raise RuntimeError("No audio was extracted from the video or the audio file is empty. Please ensure the video has an audio track.")
        print(f"Audio extracted to: {temp_wav_path}") # Add print for visibility

        # Load the Whisper model (will be cached after the first call)
        model = load_whisper_model()

        print("Transcribing audio with Whisper...") # Add print for visibility
        result = model.transcribe(temp_wav_path)
        transcription = result["text"].strip()
        print("Transcription complete.") # Add print for visibility

        return transcription

    except Exception as e:
        # Re-raise the exception after printing for debugging
        print(f"An error occurred during transcription: {e}")
        raise RuntimeError(f"Transcription failed: {e}") from e

    finally:
        # Clean up temporary files
        if video_file_path and os.path.exists(video_file_path) and (video_source.startswith("http://") or video_source.startswith("https://")):
            # Only remove if the video was downloaded
            try:
                os.remove(video_file_path)
                print(f"Cleaned up video file: {video_file_path}")
            except Exception as e:
                print(f"Error cleaning up video file {video_file_path}: {e}")
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                print(f"Cleaned up audio file: {temp_wav_path}")
            except Exception as e:
                print(f"Error cleaning up audio file {temp_wav_path}: {e}")


if __name__ == '__main__':
    # Example usage (replace with a valid video file path or URL)
    # Note: Running this requires ffmpeg and potentially yt-dlp to be installed
    # and accessible in your system's PATH.
    # For testing with a local file:
    # video_path = "path/to/your/video.mp4"
    # For testing with a URL:
    video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" # Replace with a real YouTube video ID or a direct video URL

    try:
        # transcript = transcribe_video(video_path) # Uncomment to test with local file
        transcript = transcribe_video(video_url) # Uncomment to test with URL
        print("\\n--- Transcription ---")
        print(transcript)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")