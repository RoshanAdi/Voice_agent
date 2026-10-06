import whisper
import torch

def main():
    # Safety check: ensure GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Whisper will run on CPU.")

    print("Using GPU:", torch.cuda.get_device_name(0))

    # Load Whisper medium model on GPU
    model = whisper.load_model("medium.en", device="cuda")

    # Transcribe audio
    result = model.transcribe(
        "clean_clip_0001.wav",
        fp16=True,
     # remove or change if not English
        verbose=True
    )

    # Print full transcription
    print("\n--- TRANSCRIPTION ---")
    print(result["text"])

    # Optional: print segments with timestamps
    print("\n--- SEGMENTS ---")
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f} - {end:.2f}] {text}")
    

if __name__ == "__main__":
    main()
