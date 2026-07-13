"""
Example: How to instantiate each model, load weights, and run inference.

Models:
  - CHAP-FT  (CHAP 2.0): Finetuned CNN-BiLSTM
  - CHAP-ZS:              Zero-shot CNN-BiLSTM with attention
  - CHAP-ViT:             Vision Transformer (requires external models_mae.py)

Data format:
  All models expect accelerometer data at 10Hz with 3 channels (x, y, z).
  Input shape: (batch_size, 42, 100, 3)
    - 42 windows of 10 seconds each (420 seconds total)
    - 100 timesteps per window (10Hz × 10s)
    - 3 accelerometer axes
  Output shape: (batch_size, 42) — binary logit per window (sedentary vs. active)
"""

import torch
import numpy as np

# ============================================================================
# 1. CHAP-FT (CHAP 2.0) — Finetuned CNN-BiLSTM
# ============================================================================

def example_chap_ft():
    from chap_model import CHAP
    from util.chap_utils import load_model_weights

    # Model parameters
    #   amp_factor=2:        channel amplification (CNN channels = 32*2, 64*2, ..., 256*2)
    #   bi_lstm_win_size=42: number of 10s windows per sample
    #   num_classes=2:       binary classification (sedentary vs. active)
    model = CHAP(amp_factor=2, bi_lstm_win_size=42, num_classes=2)

    # Load submitted weights
    checkpoint_path = "SUBMIT_RESULT/iWatch_H/CHAP-FT/checkpoint/checkpoint-submit.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Checkpoint can be a dict with 'model' key or a raw state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # --- Input format ---
    # Raw input shape: (batch_size, 42, 100, 3)
    # CHAP expects flattened windows: (batch_size * 42, 1, 100, 3)
    batch_size = 4
    x = torch.randn(batch_size, 42, 100, 3)  # simulated accelerometer data
    x_flat = x.view(-1, 1, 100, 3)            # (4*42, 1, 100, 3) = (168, 1, 100, 3)

    with torch.no_grad():
        logits = model(x_flat)                 # (batch_size, 42)
        probs = torch.sigmoid(logits)          # probability of being active
        preds = (probs > 0.5).long()           # binary predictions

    print(f"[CHAP-FT] Input: {x.shape} -> flattened: {x_flat.shape}")
    print(f"[CHAP-FT] Output logits: {logits.shape}, preds: {preds.shape}")
    print(f"[CHAP-FT] Sample predictions: {preds[0][:10]}")
    return model


# ============================================================================
# 2. CHAP-ZS — Zero-shot CNN-BiLSTM with Transformer Attention
# ============================================================================

def example_chap_zs():
    from chap_model import CHAP, CNNAttentionModel, FeatureExtractorWrapper

    # Step 1: Create the CNN-BiLSTM base model
    base_model = CHAP(amp_factor=2, bi_lstm_win_size=42, num_classes=2)

    # Step 2: Wrap it as a feature extractor (outputs BiLSTM hidden states)
    base_model_hidden_dim = base_model.fc_bilstm.in_features  # 256 (bidirectional LSTM output)
    feature_extractor = FeatureExtractorWrapper(base_model)

    # Step 3: Add Transformer attention on top
    #   num_layer:           number of Transformer encoder layers
    #   hidden_dim:          projection dimension
    #   num_heads:           attention heads
    #   ffn_multiplier:      feedforward hidden dim = hidden_dim * ffn_multiplier
    #   drop_path_rate:      dropout rate
    #   learnable_pos_embed: whether positional embeddings are learned
    model = CNNAttentionModel(
        base_model=feature_extractor,
        base_model_hidden_dim=base_model_hidden_dim,  # 256
        window_size=42,
        num_classes=2,
        num_layer=1,
        hidden_dim=256,
        num_heads=8,
        ffn_multiplier=2,
        drop_path_rate=0.1,
        learnable_pos_embed=True,
    )

    # Load submitted weights
    checkpoint_path = "SUBMIT_RESULT/iWatch_H/CHAP-ZS/checkpoint/checkpoint-submit.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # --- Input format ---
    # CNNAttentionModel handles reshaping internally: (batch_size, 42, 100, 3)
    batch_size = 4
    x = torch.randn(batch_size, 42, 100, 3)

    with torch.no_grad():
        logits = model(x)                      # (batch_size, 42, 1)
        logits = logits.squeeze(-1)            # (batch_size, 42)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

    print(f"[CHAP-ZS] Input: {x.shape}")
    print(f"[CHAP-ZS] Output logits: {logits.shape}, preds: {preds.shape}")
    print(f"[CHAP-ZS] Sample predictions: {preds[0][:10]}")
    return model


# ============================================================================
# 3. CHAP-ViT — Vision Transformer
#    Uses models_vit.py (VisionTransformer built on timm).
#    No submitted weights are available — this example creates the model only.
# ============================================================================

def example_chap_vit():
    from models_vit import vit_base_patch16, vit_tiny_patch16

    # ViT-base configuration
    #   img_size:   (num_channels, sequence_length) — treated as a 2D "image"
    #   patch_size: how the input is divided into patches
    #   in_chans:   1 (each window is a single-channel 2D input)
    #   num_classes: 2 (binary classification)
    #
    # Available factory functions:
    #   vit_base_patch16:  embed_dim=768, depth=12, num_heads=12
    #   vit_tiny_patch16:  embed_dim=768, depth=12, num_heads=3
    #   vit_large_patch16: embed_dim=1024, depth=24, num_heads=16
    #   vit_huge_patch14:  embed_dim=1280, depth=32, num_heads=16

    model = vit_base_patch16(
        img_size=(100, 3),     # each window: 100 timesteps × 3 axes
        patch_size=(5, 3),     # patch covers 5 timesteps × all 3 axes
        in_chans=1,            # single-channel input
        num_classes=2,         # binary classification
        use_cls=True,          # use CLS token for classification
    )
    model.eval()

    # --- Input format ---
    # VisionTransformer expects: (batch_size, in_chans, H, W) = (batch_size, 1, 100, 3)
    # For a sequence of 42 windows, flatten batch and window dimensions:
    batch_size = 4
    num_windows = 42
    x = torch.randn(batch_size, num_windows, 100, 3)  # raw accelerometer
    x_flat = x.view(-1, 1, 100, 3)                     # (168, 1, 100, 3)

    with torch.no_grad():
        logits = model(x_flat)                          # (168, 2) — per-window class logits
        logits = logits.view(batch_size, num_windows, -1)  # (4, 42, 2)
        preds = logits.argmax(dim=-1)                   # (4, 42)

    print(f"[CHAP-ViT] Input: {x.shape} -> flattened: {x_flat.shape}")
    print(f"[CHAP-ViT] Output logits: {logits.shape}, preds: {preds.shape}")
    print(f"[CHAP-ViT] Sample predictions: {preds[0][:10]}")
    return model


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHAP-FT (CHAP 2.0) — Finetuned CNN-BiLSTM")
    print("=" * 60)
    example_chap_ft()

    print()
    print("=" * 60)
    print("CHAP-ZS — Zero-shot CNN-BiLSTM + Attention")
    print("=" * 60)
    example_chap_zs()

    print()
    print("=" * 60)
    print("CHAP-ViT — Vision Transformer (vit-base)")
    print("=" * 60)
    example_chap_vit()
