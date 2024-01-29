""" Example handler file. """

import tempfile
import runpod
import requests
import os
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline,
)
import yt_dlp as youtube_dl
from diarization import diarize, build_result

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.


def download_file(url):
    """Helper function to download a file from a URL to a temporary file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            temp_filename = f.name
    return temp_filename


# def download_yt_audio(yt_url, filename):
#     info_loader = youtube_dl.YoutubeDL()
#     try:
#         info = info_loader.extract_info(yt_url, download=False)
#     except youtube_dl.utils.DownloadError as err:
#         raise Exception(str(err))

#     file_length = info["duration_string"]
#     file_h_m_s = file_length.split(":")
#     file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
#     if len(file_h_m_s) == 1:
#         file_h_m_s.insert(0, 0)
#     if len(file_h_m_s) == 2:
#         file_h_m_s.insert(0, 0)

#     file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
#     if file_length_s > YT_LENGTH_LIMIT_S:
#         yt_length_limit_hms = time.strftime(
#             "%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S)
#         )
#         file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
#         raise gr.Error(
#             f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video."
#         )

#     ydl_opts = {
#         "outtmpl": filename,
#         "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
#     }
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         try:
#             ydl.download([yt_url])
#         except youtube_dl.utils.ExtractorError as err:
#             raise gr.Error(str(err))


def download_youtube_video(yt_url):
    """Helper function to download a YouTube video to a temporary file."""
    ydl_opts = {
        "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(yt_url, download=False)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                ydl_opts.update({"outtmpl": f.name})
                with youtube_dl.YoutubeDL(ydl_opts) as ydl_temp:
                    ydl_temp.download([yt_url])
                return f.name
        except youtube_dl.utils.DownloadError as err:
            raise Exception(str(err))


def transcribe(audio_path, chunk_length, batch_size):
    """Run Whisper model inference on the given audio file."""
    model_id = "openai/whisper-large-v3"
    torch_dtype = torch.float16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_cache = "/cache/huggingface/hub"
    local_files_only = True

    # Load the model, tokenizer, and feature extractor
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=model_cache,
        local_files_only=local_files_only,
    ).to(device)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_id, cache_dir=model_cache, local_files_only=local_files_only
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, cache_dir=model_cache, local_files_only=local_files_only
    )

    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
        torch_dtype=torch_dtype,
        device=device,
    )

    # Run the transcription
    outputs = pipe(
        audio_path,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        generate_kwargs={"task": "transcribe", "language": None},
        return_timestamps=True,
    )

    return outputs


# https://docs.runpod.io/serverless/workers/handlers/handler-async
def handler(job):
    job_input = job["input"]

    input_source = job_input.get("type", "URL")  # URL or YOUTUBE
    file_url = job_input.get("audio_url", None)
    model_id = job_input.get("model_id", "openai/whisper-large-v3")
    chunk_length = job_input.get("chunk_length", 30)
    batch_size = job_input.get("batch_size", 24)
    diarization = job_input.get("diarization", False)
    translation = job_input.get("translation", False)
    language = job_input.get("language", "en")
    flash = job_input.get("flash", False)
    device_id = job_input.get("device_id", "0")

    if not file_url:
        return "No audio URL provided."

    # Download the audio file
    # TODO: Handle YouTube URLs
    # TODO: stoarge of audio files
    audio_file_path = None
    if input_source.upper() == "URL":
        audio_file_path = download_file(file_url)
    elif input_source.upper() == "YOUTUBE":
        audio_file_path = download_youtube_video(file_url)

    # Run Whisper model inference
    # transcription = transcribe(audio_file_path, chunk_length, batch_size)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16,
        device="mps" if device_id == "mps" else f"cuda:{device_id}",
        model_kwargs={"attn_implementation": "flash_attention_2"}
        if flash
        else {"attn_implementation": "sdpa"},
    )
    if device_id == "mps":
        torch.mps.empty_cache()
    ts = True  # Timestamps
    generate_kwargs = {
        "task": "translate" if translation else "transcribe",
        "language": language,
    }
    transcription = pipe(
        audio_file_path,
        chunk_length_s=30,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=ts,
    )

    if diarization:
        diarization_model = "pyannote/speaker-diarization-3.1"
        hf_token = os.getenv("HF_TOKEN")
        device_id = "0"
        speakers_transcript = diarize(
            diarization_model, hf_token, device_id, audio_file_path, transcription
        )
        return build_result(speakers_transcript, transcription)

    # Cleanup: Remove the downloaded file
    os.remove(audio_file_path)

    return transcription


runpod.serverless.start({"handler": handler})
