import os
import sys
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DefaultDataCollator
)

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
# Replace this with your actual Hugging Face username
HF_USERNAME = "your_hf_username"  
MODEL_OUTPUT_NAME = "whisper-small-nyan-jenny"
REPO_ID = f"{HF_USERNAME}/{MODEL_OUTPUT_NAME}"

BASE_MODEL = "openai/whisper-small"
DATASET_ID = "RikkaBotan/nyan-jenny-format"

def main():
    print("--- Starting Fine-Tuning Pipeline ---")
    
    # Check for Hugging Face Token
    if "HF_TOKEN" not in os.environ:
        print("[ERROR] HF_TOKEN environment variable not found. Please set it before running.")
        sys.exit(1)

    # =====================================================================
    # 2. LOAD AND RESAMPLE DATASET
    # =====================================================================
    print(f"Loading dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)
    
    print("Resampling audio columns to 16,000Hz...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # =====================================================================
    # 3. INITIALIZE PROCESSOR & MODEL
    # =====================================================================
    print(f"Loading processor and base model: {BASE_MODEL}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, language="japanese", task="transcribe")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(BASE_MODEL)
    
    # Configure generation requirements for Whisper
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # =====================================================================
    # 4. PREPROCESSING FUNCTION
    # =====================================================================
    def prepare_dataset(batch):
        audio = batch["audio"]
        
        # Extract features from audio array
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Tokenize target text (using transcription_normalised for cleaner learning alignment)
        batch["labels"] = processor.tokenizer(batch["transcription_normalised"]).input_ids
        return batch

    print("Mapping and preprocessing dataset columns...")
    encoded_dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset["train"].column_names, 
        num_proc=2
    )

    # =====================================================================
    # 5. METRICS & COLLATOR SETUP
    # =====================================================================
    data_collator = DefaultDataCollator()
    metric = evaluate.load("cer")  # Character Error Rate is ideal for Japanese text

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Clean padded positions (-100 tokens)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    # =====================================================================
    # 6. TRAINING ARGUMENTS (Optimized for Budget GPU Compute)
    # =====================================================================
    print("Setting up Training Arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_NAME,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=50,
        max_steps=500,                   # 500 steps is enough to utilize a brief credit run
        gradient_checkpointing=True,
        fp16=True if torch.cuda.is_available() else False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to="none",                 # Toggle to "wandb" if you wish to use Weights & Biases
        push_to_hub=True,                 # This safely uploads your progress automatically
        hub_model_id=REPO_ID,
        hub_strategy="every_save"
    )

    # =====================================================================
    # 7. EXECUTE TRAINER
    # =====================================================================
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Beginning model training...")
    trainer.train()

    print("Training complete! Publishing finalized model and processor to the Hub...")
    # Safely save the feature extractor configuration alongside the model artifacts
    processor.push_to_hub(REPO_ID)
    trainer.push_to_hub(commit_message="Training complete: Fine-tuned Whisper on nyan-jenny dataset")
    print(f"Success! Model is available at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()