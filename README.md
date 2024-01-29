<div align="center">

<h1>Insanely-Fast-Whisper | Worker</h1>

🚀 | Runpod worker for Insanely-Fast-Whisper.

</div>

#### Build an Image:

`docker build --build-arg HUGGING_FACE_HUB_WRITE_TOKEN=<HuggingFace Token> -t <Image Name> .`

Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand.

## Test Inputs

The following inputs can be used for testing the model:

```json
{
  "input": {
    "audio_url": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "diarization": true
  }
}
```

## Acknowledgments

- This tool is powered by Hugging Face's ASR models, primarily Whisper by OpenAI.
