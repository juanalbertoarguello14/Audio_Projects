from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds
Dataset(
    {
        features: [
            "path",
            "audio",
            "transcription",
            "english_transcription",
            "intent_class",
            "lang_id",
        ],
        num_rows: 654,
    }