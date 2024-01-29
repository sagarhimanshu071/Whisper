""" Example handler file. """

import tempfile
import runpod
import requests
import logging
import json
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


def transcribe(audio_path, chunk_length, batch_size, generate_kwargs, model_kwargs, model_id= "openai/whisper-large-v3"):
    """Run Whisper model inference on the given audio file."""
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
        model_kwargs=model_kwargs,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Run the transcription
    outputs = pipe(
        audio_path,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )

    return outputs

def call_webhook(data, url):
    headers = {
    'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return "Success"
        return "Failed"
    except Exception as e:
        logging.error("Failed to send data on webhook" + str(e))
        return "Failed"


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
    group_by_speaker = job_input.get("group_by_speaker", False)
    flash = job_input.get("flash", False)
    device_id = job_input.get("device_id", "0")

    webhook_url = job_input.get("webhook_url", None)
    user_id = job_input.get("user_id")
    session_id = job_input.get("session_id")
    if webhook_url:
        if not user_id or not session_id:
            return "For sending data to webhook, user_id and session_id is required"
    

    if not file_url:
        return "No audio URL provided."

    # Download the audio file
    audio_file_path = None
    if input_source.upper() == "URL":
        audio_file_path = download_file(file_url)
    elif input_source.upper() == "YOUTUBE":
        audio_file_path = download_youtube_video(file_url)

    if device_id == "mps":
        torch.mps.empty_cache()
    ts = True  # Timestamps
    generate_kwargs = {
        "task": "translate" if translation else "transcribe",
        "language": language,
    }
    model_kwargs = {"attn_implementation": "flash_attention_2"} if flash else {"attn_implementation": "sdpa"}
    # Run Whisper model inference
    transcription = transcribe(audio_file_path, chunk_length, batch_size, generate_kwargs, model_kwargs, model_id)
    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model_id,
    #     torch_dtype=torch.float16,
    #     device="mps" if device_id == "mps" else f"cuda:{device_id}",
    #     model_kwargs=model_kwargs,
    # )
    
    # transcription = pipe(
    #     audio_file_path,
    #     chunk_length_s=30,
    #     batch_size=batch_size,
    #     generate_kwargs=generate_kwargs,
    #     return_timestamps=ts,
    # )
    outputs = transcription
    if diarization:
        diarization_model = "pyannote/speaker-diarization-3.1"
        hf_token = os.getenv("HF_TOKEN")
        device_id = "0"
        speakers_transcript = diarize(
            diarization_model, hf_token, device_id, audio_file_path, transcription, group_by_speaker
        )
        outputs = build_result(speakers_transcript, transcription)
        # Call Webhook

    # Cleanup: Remove the downloaded file
    os.remove(audio_file_path)
    status = None
    if webhook_url:
        webhook_data = {
            "output": {
                "user_id": user_id,
                "session_id": session_id,
                "transcript_text": outputs.get("text", ""),
                "diarized_text": outputs.get("speakers", []),
                "detected_language": "en",
                "segments": outputs.get("chunks", [])
            }
        }
        status = call_webhook(webhook_data, webhook_url)

    outputs = {**outputs, "webhook_status": status}
    return outputs


runpod.serverless.start({"handler": handler})
