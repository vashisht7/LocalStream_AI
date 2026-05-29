import os
import logging
import subprocess
from faster_whisper import WhisperModel
from localstream.config import settings

logger = logging.getLogger("localstream.transcribe")

# List of common video file extensions to detect
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".m4v", ".mpg", ".mpeg", ".3gp"}

def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def extract_audio_mono(video_path: str) -> str:
    """
    Extracts a clean, 16kHz mono audio stream from a video file using FFmpeg.
    If a .wav file with the same base name already exists, it returns that path.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Determine base directory and base name
    base_dir = os.path.dirname(video_path)
    if not base_dir:
        base_dir = "."
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(base_dir, f"{base_name}.wav")

    # If the output file already exists, return it directly
    if os.path.exists(wav_path):
        logger.info(f"Extracted mono audio already exists at: {wav_path}")
        return wav_path

    logger.info(f"Extracting mono audio: {video_path} -> {wav_path}")
    
    # FFmpeg arguments to extract 16kHz mono PCM 16-bit audio
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output file if it exists
        "-i", video_path,
        "-vn",              # Disable video recording
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16kHz sample rate
        "-ac", "1",         # Mono channel
        wav_path
    ]

    try:
        # Run FFmpeg command silently
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info(f"Audio extraction completed: {wav_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {e}")
        raise RuntimeError(f"FFmpeg audio extraction failed for {video_path}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during audio extraction: {e}")
        raise

    return wav_path

def run_transcription(file_path: str) -> list:
    """
    Transcribes a media file (audio or video) using faster-whisper.
    Returns a list of dictionaries with transcription segments and timestamp ranges.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    
    # Check if the file is a video and extract audio if necessary
    if ext in VIDEO_EXTENSIONS:
        logger.info(f"Detecting video format '{ext}'. Extracting audio first...")
        audio_path = extract_audio_mono(file_path)
    else:
        logger.info(f"Treating file as direct audio: {file_path}")
        audio_path = file_path

    # Determine Whisper model parameters
    device = settings.compute_device
    # Fallback to cpu using int8 quantization as requested
    compute_type = "int8" if device == "cpu" else "float16"

    logger.info(f"Initializing WhisperModel(size='{settings.whisper_model_size}', device='{device}', compute_type='{compute_type}')")
    try:
        model = WhisperModel(settings.whisper_model_size, device=device, compute_type=compute_type)
    except Exception as e:
        logger.error(f"Failed to initialize WhisperModel: {e}")
        raise RuntimeError(f"Whisper initialization error: {e}") from e

    logger.info(f"Starting transcription of audio file: {audio_path}")
    try:
        # Transcribe with beam search parameters
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        transcript_chunks = []
        for idx, segment in enumerate(segments):
            chunk = {
                "chunk_id": idx,
                "start_seconds": segment.start,
                "end_seconds": segment.end,
                "start_timestamp": format_seconds(segment.start),
                "end_timestamp": format_seconds(segment.end),
                "text": segment.text.strip()
            }
            transcript_chunks.append(chunk)
            logger.debug(f"Segment {idx} [{chunk['start_timestamp']} -> {chunk['end_timestamp']}]: {chunk['text']}")

        logger.info(f"Transcription completed. Extracted {len(transcript_chunks)} chunks.")
        return transcript_chunks
        
    except Exception as e:
        logger.error(f"Error during transcription process: {e}")
        raise RuntimeError(f"Transcription execution failed: {e}") from e
