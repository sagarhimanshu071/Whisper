import torch
import requests
import torch
import numpy as np
from torchaudio import functional as F
import torch
from pyannote.audio import Pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import sys
from typing import TypedDict

# Code lifted from https://github.com/huggingface/speechbox/blob/main/src/speechbox/diarize.py
# and from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py


class JsonTranscriptionResult(TypedDict):
    speakers: list
    chunks: list
    text: str


def build_result(transcript, outputs) -> JsonTranscriptionResult:
    return {
        "speakers": transcript,
        "chunks": outputs["chunks"],
        "text": outputs["text"],
    }


def preprocess_inputs(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(diarizer_inputs, diarization_pipeline):
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
    )

    # segments = []
    # for segment, track, label in diarization.itertracks(yield_label=True):
    #     segments.append(
    #         {
    #             "segment": {"start": segment.start, "end": segment.end},
    #             "track": track,
    #             "label": label,
    #         }
    #     )
    return diarization


def post_process_segments_and_transcripts(
    new_segments, transcript, group_by_speaker
) -> list:
    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array(
        [
            chunk["timestamp"][-1]
            if chunk["timestamp"][-1] is not None
            else sys.float_info.max
            for chunk in transcript
        ]
    )
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

        if len(end_timestamps) == 0:
            break

    return segmented_preds


def diarize(diarization_model, hf_token, device_id, file_name, outputs):
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path=diarization_model,
        use_auth_token=hf_token,
    )
    diarization_pipeline.to(
        torch.device("mps" if device_id == "mps" else f"cuda:{device_id}")
    )

    inputs, diarizer_inputs = preprocess_inputs(inputs=file_name)

    segments = diarize_audio(diarizer_inputs, diarization_pipeline)

    return post_process_segments_and_transcripts(
        segments, outputs["chunks"], group_by_speaker=False
    )