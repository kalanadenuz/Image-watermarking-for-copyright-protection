"""
Dedicated open-source watermark detector utilities.

This module loads the ConvNeXt-based watermark classifier from the
boomb0om/watermark-detection project and caches its weights locally.
The app uses this model for the final watermarked / no-watermark decision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


_MODEL = None
_MODEL_MTIME = None
_THRESHOLD = 0.60

_WEIGHTS_DIR = Path("models")
_WEIGHTS_PATH = _WEIGHTS_DIR / "boomb0om_convnext_tiny_watermarks_detector.pth"
_WEIGHTS_URL = "https://huggingface.co/boomb0om/watermark-detectors/resolve/main/convnext-tiny_watermarks_detector.pth?download=true"


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class _Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class _ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=2,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            _LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                _LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    _Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )
        self.apply(self._init_weights)
        last_layer = self.head[-1]
        last_layer.weight.data.mul_(head_init_scale)
        last_layer.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _build_model():
    return _ConvNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), num_classes=2)


def _ensure_weights() -> Path:
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if _WEIGHTS_PATH.exists() and _WEIGHTS_PATH.stat().st_size > 5_000_000:
        return _WEIGHTS_PATH

    response = requests.get(_WEIGHTS_URL, stream=True, timeout=120)
    response.raise_for_status()
    tmp_path = _WEIGHTS_PATH.with_suffix(".download")
    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    tmp_path.replace(_WEIGHTS_PATH)
    return _WEIGHTS_PATH


def _load_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_deep_model(model_path: str | None = None) -> bool:
    global _MODEL, _MODEL_MTIME, _THRESHOLD
    if torch is None:
        return False

    try:
        path = Path(model_path) if model_path else _ensure_weights()
    except Exception:
        return False

    mtime = path.stat().st_mtime
    if _MODEL is not None and _MODEL_MTIME == mtime:
        return True

    try:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location="cpu")
        state_dict = _load_checkpoint_state(checkpoint)
        model = _build_model()
        model.load_state_dict(state_dict)
        model.eval()
        _MODEL = model
        _MODEL_MTIME = mtime
        # Prefer a precision-friendly operating point for demo use.
        _THRESHOLD = 0.60
        return True
    except Exception:
        _MODEL = None
        _MODEL_MTIME = None
        return False


def _preprocess_bgr(image_bgr: np.ndarray) -> "torch.Tensor":
    img = cv2.cvtColor(cv2.resize(image_bgr, (256, 256)), cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return torch.from_numpy(x)


def deep_watermark_probability(image_bgr: np.ndarray) -> Optional[float]:
    if not load_deep_model():
        return None

    with torch.no_grad():
        x = _preprocess_bgr(image_bgr)
        logits = _MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]
        prob = probs[1].item() if probs.numel() > 1 else probs[0].item()
    return float(np.clip(prob, 0.0, 1.0))


def deep_watermark_threshold(default: float = 0.56) -> float:
    if load_deep_model():
        return _THRESHOLD
    return default
