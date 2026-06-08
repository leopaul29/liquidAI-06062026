# train.py
import os
from dotenv import load_dotenv
from datasets import Audio, load_dataset
from liquid_audio import LFM2AudioProcessor, LFM2AudioModel
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.trainer import Trainer

# Load your Hugging Face token
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# --- Preprocessing ---
class TrainingSamples:
    def __init__(self, limit=0):  # limit=0 for full dataset
        self.limit = limit

    def __iter__(self):
        ds = load_dataset("RikkaBotan/nyan-jenny-format", split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, row in enumerate(ds):
            if self.limit and i >= self.limit:
                break
            yield [
                ChatMessage(role="system", content=[TextSegment(text="Perform TTS. Use the Japanese female voice.")]),
                ChatMessage(role="user", content=[TextSegment(text=row["transcription"])]),
                ChatMessage(role="assistant", content=[AudioSegment(audio=row["audio"]["bytes"])]),
            ]

processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B", device="cuda").eval()
mapper = LFM2AudioChatMapper(processor)

print("🔹 Preprocessing dataset...")
preprocess_dataset(
    data=TrainingSamples(limit=200),  # Use limit=0 for full dataset
    output_path="data/nyan_jenny/train",
    mapper=mapper,
    max_context_length=256,
)

# --- Training ---
class SimpleTrainer(Trainer):
    def log(self, model_output):
        super().log(model_output)
        if self.step % 10 == 0:
            print(f"Step {self.step}, Loss: {model_output.loss.item():.4f}")

print("🔹 Training model...")
trainer = SimpleTrainer(
    model_id="LiquidAI/LFM2.5-Audio-1.5B",
    train_data=LFM2DataLoader(dataset_path="data/nyan_jenny/train", context_length=256),
    lr=1e-4,
    batch_size=8,
    max_steps=100,  # Use 5000 for full training
    warmup_steps=10,
    dataloader_num_workers=4,
    logging_interval=10,
    save_interval=50,
    output_dir="ckpt/nyan_jenny",
)
trainer.train()
print("✅ Training complete! Model saved to ckpt/nyan_jenny")