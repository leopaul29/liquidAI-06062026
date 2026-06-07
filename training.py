from __future__ import annotations

# --- Compatibility shim -------------------------------------------------------
# torch.float8_e8m0fnu was added in PyTorch 2.6.
# transformers imports it at module level but only uses it in fine-grained FP8
# quantization paths, which this bfloat16 training script never triggers.
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", torch.bfloat16)
# ------------------------------------------------------------------------------

import argparse
import dataclasses
# ... rest of your original imports unchanged
import argparse
import dataclasses
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, login as hf_login
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
from liquid_audio.utils import LFMModality

HF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-JP"

# On gèle tout sauf le backbone LFM2 pour économiser la VRAM sur le cloud
FREEZE_PATTERNS = [
    "conformer",
    "audio_adapter",
    "depthformer",
    "depth_linear",
    "depth_embeddings",
    "audio_embedding",
]

# ---------------------------------------------------------------------------
# W&B Logger
# ---------------------------------------------------------------------------
class WandbLogger:
    def __init__(self, enabled, project, name, config):
        self.enabled = enabled
        if not enabled:
            return
        try:
            import wandb
            self.wandb = wandb
            wandb.init(project=project, name=name, config=config)
            print(f"  [W&B] {wandb.run.url}\n")
        except ImportError:
            print("  [W&B] not installed – disabled.")
            self.enabled = False

    def log(self, metrics, step=None):
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def log_summary(self, metrics):
        if self.enabled:
            for k, v in metrics.items():
                self.wandb.run.summary[k] = v

    def watch(self, model, log_freq=100):
        if self.enabled:
            self.wandb.watch(model, log="gradients", log_freq=log_freq)

    def finish(self):
        if self.enabled:
            self.wandb.finish()


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------
def apply_freeze(model: LFM2AudioModel) -> dict:
    print("  Named top-level children:")
    for child_name, child_module in model.named_children():
        freeze = any(pat in child_name for pat in FREEZE_PATTERNS)
        print(f"    {child_name:45s}  {'[FREEZE]' if freeze else '[TRAIN] '}")
        if freeze:
            for p in child_module.parameters():
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable : {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen    : {total-trainable:>12,}")
    print(f"  Total     : {total:>12,}\n")
    return {"params_trainable": trainable, "params_frozen": total - trainable, "params_total": total}


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------
class TTSDataset(Dataset):
    def __init__(self, hf_dataset, mapper: LFM2AudioChatMapper):
        self.data   = hf_dataset
        self.mapper = mapper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        try:
            audio_bytes = row.get("target_audio")
            if not audio_bytes:
                return None
            system_prompt = row.get("system_prompt") or \
                "Perform TTS in Japanese. Use a natural Japanese female voice."
            input_text = row.get("input_text") or row.get("target_text", "")
            if not input_text.strip():
                return None

            messages = [
                ChatMessage(role="system",   content=[TextSegment(text=system_prompt)]),
                ChatMessage(role="user",     content=[TextSegment(text=input_text)]),
                ChatMessage(role="assistant", content=[AudioSegment(audio=audio_bytes)]),
            ]
            return self.mapper(messages)

        except Exception:
            print(f"  [Dataset skip idx={idx}]")
            traceback.print_exc()
            return None


def collate_fn(batch):
    return [b for b in batch if b is not None]


# ---------------------------------------------------------------------------
# Loss Computation
# ---------------------------------------------------------------------------
def compute_loss(model: LFM2AudioModel, sample: Any,
                 device: torch.device, codebooks: int = 1) -> torch.Tensor:

    fields = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in dataclasses.asdict(sample).items()
    }

    text          = fields["text"]             
    audio_in      = fields["audio_in"]         
    audio_in_lens = fields["audio_in_lens"]    
    audio_out     = fields["audio_out"]        
    modality_flag = fields["modality_flag"]    
    sup_mask      = fields["supervision_mask"] 

    in_emb = model._prefill(
        text=text, audio_in=audio_in, audio_in_lens=audio_in_lens,
        audio_out=audio_out, modality_flag=modality_flag,
    )  

    lfm_out = model.lfm(inputs_embeds=in_emb, use_cache=False)
    hidden  = lfm_out.last_hidden_state  

    sup_flat = sup_mask[0]                                        
    mod_flat = modality_flag[0]                                   

    tgt_pos = sup_flat.nonzero(as_tuple=True)[0]                  
    if tgt_pos.numel() == 0:
        return hidden.new_tensor(0.0, requires_grad=True)

    src_pos = (tgt_pos - 1).clamp(min=0)
    h_pred  = hidden[0, src_pos]                                  
    sup_mod = mod_flat[tgt_pos]                                   

    losses: list[torch.Tensor] = []

    # 4a. Text Loss
    is_text = sup_mod == int(LFMModality.TEXT)
    if is_text.any():
        h_text      = h_pred[is_text]                               
        tgt_pos_t   = tgt_pos[is_text]                              
        text_cumsum = (mod_flat == int(LFMModality.TEXT)).cumsum(0)  
        local_t     = text_cumsum[tgt_pos_t] - 1                 
        target_ids  = text[0, local_t]                            
        logits      = F.linear(h_text, model.lfm.embed_tokens.weight)
        losses.append(F.cross_entropy(logits, target_ids))

    # 4b. Audio Loss
    is_audio = sup_mod == int(LFMModality.AUDIO_OUT)
    if is_audio.any():
        h_audio      = h_pred[is_audio]                                
        tgt_pos_a   = tgt_pos[is_audio]                               
        audio_cumsum = (mod_flat == int(LFMModality.AUDIO_OUT)).cumsum(0)  
        local_a      = audio_cumsum[tgt_pos_a] - 1                 

        for cb in range(min(codebooks, audio_out.shape[0])):
            target_ids = audio_out[cb, local_a]                   
            vocab = model.audio_vocab_size
            w = model.audio_embedding.embedding.weight[cb * vocab:(cb + 1) * vocab]
            logits = F.linear(h_audio, w)                                         
            losses.append(F.cross_entropy(logits, target_ids))

    if not losses:
        return hidden.new_tensor(0.0, requires_grad=True)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataset, device, codebooks, max_samples=64):
    model.eval()
    total, count = 0.0, 0
    for idx in range(min(max_samples, len(dataset))):
        sample = dataset[idx]
        if sample is None:
            continue
        try:
            loss   = compute_loss(model, sample, device, codebooks=codebooks)
            total += loss.item()
            count += 1
        except Exception:
            print(f"  [Val skip idx={idx}]")
            traceback.print_exc()
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# MODIFIÉ : Sauvegarde Globale et Clé en Main (from_pretrained compatible)
# ---------------------------------------------------------------------------
def save_checkpoint(model, processor, path, metadata=None):
    path.mkdir(parents=True, exist_ok=True)
    print(f"  [Sauvegarde] Enregistrement du modèle CLÉ EN MAIN complet dans {path}...")
    
    # On force la sauvegarde complète (poids figés + poids entraînés réunis)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    
    import json
    conf_dict = vars(model.conf) if hasattr(model, "conf") else {}
    with open(path / "model_conf.json", "w") as f:
        json.dump({k: v for k, v in conf_dict.items() if isinstance(v, (int, float, str, bool, list))}, f, indent=2)
        
    meta_str = "\n".join(f"- {k}: {v}" for k, v in metadata.items()) if metadata else ""
    (path / "README.md").write_text(
        "# LFM2.5-Audio Fine-Tuned (Modèle Complet Unifié)\n\n"
        f"Base : `LiquidAI/LFM2.5-Audio-1.5B-JP`.\n\n"
        + (f"## Métadonnées\n{meta_str}\n\n" if meta_str else "")
        + "## Code pour charger directement dans Gradio en local :\n```python\n"
        "from liquid_audio import LFM2AudioModel, LFM2AudioProcessor\n"
        "model = LFM2AudioModel.from_pretrained('./dossier_du_modele')\n"
        "processor = LFM2AudioProcessor.from_pretrained('./dossier_du_modele')\n```\n"
    )
    print("  [Sauvegarde] Terminée !")


# ---------------------------------------------------------------------------
# Main Training Logic
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}"
              f"  VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")

    if args.token:
        hf_login(token=args.token, add_to_git_credential=False)
        print("  [HF] Logged in")

    print(f"\n[Load] {HF_REPO} ...")
    processor = LFM2AudioProcessor.from_pretrained(HF_REPO, device=device).eval()
    model     = LFM2AudioModel.from_pretrained(HF_REPO, device=device).to(dtype=torch.bfloat16)

    print("\n[Freeze]")
    param_counts = apply_freeze(model)
    model.train()

    mapper = LFM2AudioChatMapper(processor, codebooks=model.codebooks)

    print(f"[Data] {args.dataset}")
    raw      = load_dataset(args.dataset)
    train_ds = TTSDataset(raw["train"],      mapper)
    val_ds   = TTSDataset(raw["validation"], mapper)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable_params, lr=args.lr,
                       weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = args.epochs * max(len(train_loader) // args.grad_accum, 1)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    logger = WandbLogger(
        enabled = args.wandb_project is not None,
        project = args.wandb_project or "lfm2-audio-ja-tts",
        name    = args.wandb_run_name or f"backbone-lr{args.lr}-ep{args.epochs}",
        config  = {"base_model": HF_REPO, "dataset": args.dataset,
                   "epochs": args.epochs, "lr": args.lr, "codebooks": args.codebooks,
                   "grad_accum": args.grad_accum, **param_counts},
    )
    logger.watch(model, log_freq=args.log_every)

    print(f"\n[Train] epochs={args.epochs}  lr={args.lr}  "
          f"grad_accum={args.grad_accum}  steps≈{total_steps}\n")

    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_steps = 0.0, 0
        accum = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            for sample in batch:
                try:
                    loss = compute_loss(model, sample, device, codebooks=args.codebooks)
                    (loss / args.grad_accum).backward()
                    accum = accum + loss.detach() / args.grad_accum
                except Exception:
                    print("  [Skip sample]")
                    traceback.print_exc()
                    continue

            if (batch_idx + 1) % args.grad_accum == 0 or batch_idx == len(train_loader) - 1:
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                step_loss    = accum.item() * args.grad_accum
                ep_loss     += step_loss
                ep_steps    += 1
                accum        = torch.tensor(0.0, device=device)
                current_lr   = scheduler.get_last_lr()[0]

                logger.log({"train/loss": step_loss, "train/grad_norm": grad_norm,
                            "train/lr": current_lr, "train/epoch": epoch}, step=global_step)

                if global_step % args.log_every == 0:
                    print(f"  ep={epoch}/{args.epochs}  step={global_step}"
                          f"  loss={step_loss:.4f}  gnorm={grad_norm:.3f}"
                          f"  lr={current_lr:.2e}  t={time.time()-t0:.0f}s")

        avg_train = ep_loss / max(ep_steps, 1)
        val_loss  = evaluate(model, val_ds, device, args.codebooks, args.val_samples)
        is_best   = val_loss < best_val_loss

        logger.log({"epoch/train_loss": avg_train, "epoch/val_loss": val_loss,
                    "epoch/epoch": epoch}, step=global_step)
        print(f"\nEpoch {epoch}  train={avg_train:.4f}  val={val_loss:.4f}"
              + ("  ← best" if is_best else "") + "\n")

        ckpt_meta = {"epoch": epoch, "global_step": global_step,
                     "train_loss": f"{avg_train:.4f}", "val_loss": f"{val_loss:.4f}"}
        
        # Sauvegarde intermédiaire complète
        save_checkpoint(model, processor, output_dir / f"checkpoint-epoch{epoch}", ckpt_meta)

        if is_best:
            best_val_loss = val_loss
            # Sauvegarde du meilleur modèle complet
            save_checkpoint(model, processor, output_dir / "best", ckpt_meta)
            logger.log_summary({"best_val_loss": best_val_loss, "best_epoch": epoch})
            print(f"  *** New best val={best_val_loss:.4f} ***\n")

    logger.finish()

    # Si l'argument push_to_hub est activé, on envoie le modèle complet sur Hugging Face
    if args.push_to_hub:
        print(f"\n[Push] → {args.push_to_hub}")
        api = HfApi(token=args.token)
        api.create_repo(args.push_to_hub, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir / "best"), repo_id=args.push_to_hub,
            repo_type="model",
            commit_message="Fine-tuned LFM2 complete model (Japanese TTS)",
        )
        print(f"[Done] https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        required=True)
    p.add_argument("--token",          default=os.environ.get("HF_TOKEN"))
    p.add_argument("--output_dir",     default="./lfm2-audio-ja-tts")
    p.add_argument("--push_to_hub",    default=None)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--grad_accum",     type=int,   default=8)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--codebooks",      type=int,   default=1)
    p.add_argument("--val_samples",    type=int,   default=64)
    p.add_argument("--log_every",      type=int,   default=50)
    p.add_argument("--wandb_project",  default=None)
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()

    train(args)