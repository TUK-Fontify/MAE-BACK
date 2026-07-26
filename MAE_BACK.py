"""
Fontranslate Stage2 한글 글리프 추론 서비스
===========================================

이 모듈은 다음 독립 추론 체크포인트 전용이다.

    https://fontify-986995923828-ap-northeast-2-an.s3.ap-northeast-2.amazonaws.com/inference_fp32.pt

체크포인트 형식
---------------
- format_version
- model
- architecture
    - vit_config
    - num_chars
    - style_tokens_side
- preprocess
    - image_size
    - k_refs
    - threshold
    - mean
    - std
- characters
- character_codes

서비스 동작
-----------
1. inference_fp32.pt를 S3 HTTPS URL에서 서버 로컬로 한 번 내려받는다.
2. 체크포인트 내부 architecture/preprocess/characters를 읽어 모델을 구성한다.
3. 요청으로 받은 TTF/OTF S3 HTTPS URL에서 폰트를 내려받는다.
4. 영어 A-Z/a-z 중 지원되는 글리프를 구조적으로 K개 선택한다.
5. Stage2SharpModel이 한글 14자의 1채널 mask logits를 생성한다.
6. sigmoid + 체크포인트 threshold로 이진화한다.
7. 기존 백엔드 계약을 유지하기 위해 128x128 grayscale PNG bytes로 반환한다.

필수 패키지
-----------
pip install torch torchvision transformers pillow fonttools requests

환경변수
--------
HANGUL_MODEL_URL       모델 HTTPS URL
HANGUL_MODEL_PATH      서버 내부 로컬 모델 경로
HANGUL_DEVICE          cuda 또는 cpu
HANGUL_OUTPUT_SIZE     외부 반환 PNG 크기, 기본 128
HANGUL_RENDER_SIZE     입력 영문 글리프 렌더링 크기, 기본 128
HANGUL_K_REFS          선택 사항. 설정하면 체크포인트 k_refs를 덮어쓴다.
HANGUL_THRESHOLD       선택 사항. 설정하면 체크포인트 threshold를 덮어쓴다.
HANGUL_MAX_MODEL_BYTES 모델 최대 크기
HANGUL_MAX_FONT_BYTES  입력 폰트 최대 크기

백엔드 사용 예시
----------------
from MAE_BACK import (
    initialize_service,
    generate_hangul_pngs,
)

initialize_service()
pngs = generate_hangul_pngs(font_https_url)
# {"가": b"...PNG...", ..., "히": b"...PNG..."}
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from transformers import ViTMAEConfig, ViTMAEForPreTraining
from urllib3.util.retry import Retry


# ============================================================
# 1. 기본 설정
# ============================================================

DEFAULT_MODEL_URL = "https://fontify-986995923828-ap-northeast-2-an.s3.ap-northeast-2.amazonaws.com/inference_fp32.pt"
DEFAULT_TARGET_CHARS = tuple("가나더려모부쇼야져쵸켜튜프히")
DEFAULT_ENGLISH_CANDIDATES = tuple(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

REFERENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset("IHEFfilt"),
    frozenset("OCGQocgq"),
    frozenset("AVWXYKvwxyk"),
    frozenset("BRPDbpdmn"),
    frozenset("SJUasuje"),
)


def _optional_env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"환경변수 {name}은 정수여야 합니다: {raw}") from exc


def _optional_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"환경변수 {name}은 실수여야 합니다: {raw}") from exc


@dataclass(frozen=True)
class Settings:
    model_url: str = os.getenv(
        "HANGUL_MODEL_URL",
        DEFAULT_MODEL_URL,
    ).strip()

    model_path: Path = Path(
        os.getenv(
            "HANGUL_MODEL_PATH",
            str(
                Path(__file__).resolve().parent
                / "models"
                / "stage2"
                / "inference_fp32.pt"
            ),
        )
    ).expanduser()

    device: str = os.getenv(
        "HANGUL_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    ).strip()

    output_size: int = int(os.getenv("HANGUL_OUTPUT_SIZE", "128"))
    render_size: int = int(os.getenv("HANGUL_RENDER_SIZE", "128"))

    # None이면 inference_fp32.pt 내부 preprocess 값을 사용한다.
    k_refs_override: int | None = _optional_env_int("HANGUL_K_REFS")
    threshold_override: float | None = _optional_env_float(
        "HANGUL_THRESHOLD"
    )

    connect_timeout: float = float(
        os.getenv("HANGUL_CONNECT_TIMEOUT", "5.0")
    )
    model_read_timeout: float = float(
        os.getenv("HANGUL_MODEL_READ_TIMEOUT", "300.0")
    )
    font_read_timeout: float = float(
        os.getenv("HANGUL_FONT_READ_TIMEOUT", "30.0")
    )

    max_model_bytes: int = int(
        os.getenv(
            "HANGUL_MAX_MODEL_BYTES",
            str(3 * 1024 * 1024 * 1024),
        )
    )
    max_font_bytes: int = int(
        os.getenv(
            "HANGUL_MAX_FONT_BYTES",
            str(50 * 1024 * 1024),
        )
    )

    default_dec_layers: int = 4
    default_dec_heads: int = 8
    default_style_tokens_side: int = 4
    default_image_size: int = 224
    default_k_refs: int = 12
    default_threshold: float = 0.40
    default_mean: tuple[float, float, float] = (
        0.485,
        0.456,
        0.406,
    )
    default_std: tuple[float, float, float] = (
        0.229,
        0.224,
        0.225,
    )


SETTINGS = Settings()


class InferenceServiceError(RuntimeError):
    """API 계층에서 HTTP 오류로 변환할 수 있는 추론 서비스 예외."""


# ============================================================
# 2. HTTP 다운로드
# ============================================================


def _create_http_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=20,
        pool_maxsize=20,
    )
    session.mount("https://", adapter)
    return session


_HTTP = _create_http_session()


def _validate_https_url(url: str, *, field_name: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme.lower() != "https":
        raise InferenceServiceError(
            f"{field_name}은 HTTPS URL이어야 합니다."
        )
    if not parsed.hostname:
        raise InferenceServiceError(
            f"{field_name}이 유효하지 않습니다."
        )


def _download_https_bytes(
    url: str,
    *,
    max_bytes: int,
    field_name: str,
    read_timeout: float,
) -> bytes:
    _validate_https_url(url, field_name=field_name)

    print(f"[다운로드 시작] {field_name}", flush=True)
    data = bytearray()

    try:
        with _HTTP.get(
            url,
            stream=True,
            timeout=(SETTINGS.connect_timeout, read_timeout),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            declared = response.headers.get("Content-Length")
            total_size = int(declared) if declared else None

            if total_size and total_size > max_bytes:
                raise InferenceServiceError(
                    f"{field_name} 크기가 제한을 초과합니다. "
                    f"declared={total_size}, limit={max_bytes}"
                )

            if total_size:
                print(
                    f"[전체 크기] {total_size / 1024 / 1024:.1f} MB",
                    flush=True,
                )

            last_report_mb = 0.0

            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                data.extend(chunk)

                if len(data) > max_bytes:
                    raise InferenceServiceError(
                        f"{field_name} 크기가 제한을 초과합니다."
                    )

                downloaded_mb = len(data) / 1024 / 1024
                if downloaded_mb - last_report_mb >= 50:
                    last_report_mb = downloaded_mb
                    if total_size:
                        percent = len(data) / total_size * 100
                        print(
                            f"[다운로드 중] {downloaded_mb:.1f} MB "
                            f"({percent:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[다운로드 중] {downloaded_mb:.1f} MB",
                            flush=True,
                        )

    except requests.RequestException as exc:
        raise InferenceServiceError(
            f"{field_name} 다운로드에 실패했습니다: {exc}"
        ) from exc

    if not data:
        raise InferenceServiceError(
            f"다운로드한 {field_name}이 비어 있습니다."
        )

    print(
        f"[다운로드 완료] {len(data) / 1024 / 1024:.1f} MB",
        flush=True,
    )
    return bytes(data)


# ============================================================
# 3. 모델 파일 동기화
# ============================================================

_MODEL_FILE_LOCK = threading.RLock()


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".metadata.json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_model_metadata(model_path: Path) -> dict[str, Any]:
    path = _metadata_path(model_path)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_model_metadata(
    model_path: Path,
    *,
    model_url: str,
    sha256: str,
    size: int,
) -> None:
    metadata = {
        "model_url": model_url,
        "sha256": sha256,
        "size": size,
    }

    path = _metadata_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _validate_inference_payload(checkpoint: Mapping[str, Any]) -> None:
    required_top_level = {
        "format_version",
        "model",
        "architecture",
        "preprocess",
        "characters",
    }
    missing = sorted(required_top_level.difference(checkpoint))

    if missing:
        raise InferenceServiceError(
            "inference_fp32.pt 필수 항목이 없습니다: "
            + ", ".join(missing)
        )

    format_version = checkpoint.get("format_version")
    if format_version != 1:
        raise InferenceServiceError(
            "지원하지 않는 inference 체크포인트 버전입니다: "
            f"{format_version}"
        )

    if not isinstance(checkpoint.get("model"), Mapping):
        raise InferenceServiceError(
            "inference_fp32.pt의 model이 state_dict 형식이 아닙니다."
        )
    if not isinstance(checkpoint.get("architecture"), Mapping):
        raise InferenceServiceError(
            "inference_fp32.pt의 architecture 형식이 올바르지 않습니다."
        )
    if not isinstance(checkpoint.get("preprocess"), Mapping):
        raise InferenceServiceError(
            "inference_fp32.pt의 preprocess 형식이 올바르지 않습니다."
        )


def sync_model_from_https(
    *,
    model_url: str | None = None,
    local_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> Path:
    """
    S3 inference_fp32.pt를 서버 로컬 파일로 동기화한다.

    같은 URL의 로컬 캐시가 존재하면 기본적으로 재다운로드하지 않는다.
    S3의 동일 URL 객체를 교체한 경우 force=True로 호출해야 한다.
    """
    url = (model_url or SETTINGS.model_url).strip()
    model_path = Path(
        local_path or SETTINGS.model_path
    ).expanduser().resolve()

    if not url:
        if model_path.exists():
            return model_path
        raise InferenceServiceError(
            "HANGUL_MODEL_URL이 없고 로컬 추론 모델도 없습니다."
        )

    _validate_https_url(url, field_name="모델 URL")

    with _MODEL_FILE_LOCK:
        metadata = _read_model_metadata(model_path)

        if (
            not force
            and model_path.exists()
            and metadata.get("model_url") == url
        ):
            return model_path

        model_bytes = _download_https_bytes(
            url,
            max_bytes=SETTINGS.max_model_bytes,
            field_name="inference_fp32.pt",
            read_timeout=SETTINGS.model_read_timeout,
        )

        try:
            checkpoint = torch.load(
                io.BytesIO(model_bytes),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise InferenceServiceError(
                "다운로드한 파일이 유효한 PyTorch 추론 파일이 아닙니다."
            ) from exc

        if not isinstance(checkpoint, Mapping):
            raise InferenceServiceError(
                "inference_fp32.pt 최상위 형식이 dict가 아닙니다."
            )

        _validate_inference_payload(checkpoint)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = model_path.with_suffix(model_path.suffix + ".download")
        temp_path.write_bytes(model_bytes)
        os.replace(temp_path, model_path)

        _write_model_metadata(
            model_path,
            model_url=url,
            sha256=_sha256(model_bytes),
            size=len(model_bytes),
        )

        clear_model_cache()
        return model_path


# ============================================================
# 4. 체크포인트 파싱
# ============================================================


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _extract_state_dict(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise InferenceServiceError(
            "체크포인트에서 model state_dict를 찾지 못했습니다."
        )

    result = dict(state)

    if any(key.startswith("module.") for key in result):
        result = {
            key.removeprefix("module."): value
            for key, value in result.items()
        }

    required = {
        "query_pos",
        "char_emb.weight",
        "style_encoder.embeddings.cls_token",
        "style_norm.weight",
        "to_feature.weight",
        "refine_14.block.0.weight",
        "up_224.conv.weight",
        "mask_head.weight",
    }
    missing = sorted(required.difference(result))

    if missing:
        raise InferenceServiceError(
            "Stage2SharpModel 필수 파라미터가 없습니다: "
            + ", ".join(missing)
        )

    return result


def _infer_decoder_layers(
    state_dict: Mapping[str, torch.Tensor],
) -> int:
    indexes: list[int] = []

    for key in state_dict:
        if not key.startswith("cross_attn."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            indexes.append(int(parts[1]))

    return (
        max(indexes) + 1
        if indexes
        else SETTINGS.default_dec_layers
    )


def _parse_float_triplet(
    value: Any,
    *,
    field_name: str,
    default: Sequence[float],
) -> tuple[float, float, float]:
    source = value if value is not None else default

    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise InferenceServiceError(
            f"{field_name}은 길이 3의 배열이어야 합니다."
        )
    if len(source) != 3:
        raise InferenceServiceError(
            f"{field_name} 길이는 3이어야 합니다: {len(source)}"
        )

    return (float(source[0]), float(source[1]), float(source[2]))


# ============================================================
# 5. inference_fp32.pt와 동일한 Stage2SharpModel
# ============================================================


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res = ResidualBlock(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, size=size, mode="nearest")
        return self.res(self.conv(x))


class Stage2SharpModel(nn.Module):
    """학습 노트북의 Stage2SharpModel과 state_dict 호환되는 추론 모델."""

    def __init__(
        self,
        *,
        vit_config_dict: Mapping[str, Any],
        image_size: int,
        num_chars: int,
        dec_layers: int,
        dec_heads: int,
        style_tokens_side: int,
    ) -> None:
        super().__init__()

        vit_config = ViTMAEConfig.from_dict(dict(vit_config_dict))
        scaffold = ViTMAEForPreTraining(vit_config)
        self.style_encoder = scaffold.vit
        del scaffold

        # Step2 학습과 동일하게 참조 이미지 패치를 마스킹하지 않는다.
        self.style_encoder.config.mask_ratio = 0.0

        self.image_size = int(image_size)
        self.style_tokens_side = int(style_tokens_side)

        hidden_size = int(self.style_encoder.config.hidden_size)
        patch_size = int(self.style_encoder.config.patch_size)

        if self.image_size % patch_size != 0:
            raise InferenceServiceError(
                "image_size가 ViT patch_size로 나누어지지 않습니다: "
                f"image_size={self.image_size}, patch_size={patch_size}"
            )
        if hidden_size % dec_heads != 0:
            raise InferenceServiceError(
                "hidden_size가 dec_heads로 나누어지지 않습니다: "
                f"hidden_size={hidden_size}, dec_heads={dec_heads}"
            )

        self.n_side = self.image_size // patch_size
        self.n_queries = self.n_side**2

        self.char_emb = nn.Embedding(num_chars, hidden_size)
        self.query_pos = nn.Parameter(
            torch.randn(1, self.n_queries, hidden_size) * 0.02
        )
        self.style_norm = nn.LayerNorm(hidden_size)

        self.self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_size,
                    dec_heads,
                    batch_first=True,
                )
                for _ in range(dec_layers)
            ]
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_size,
                    dec_heads,
                    batch_first=True,
                )
                for _ in range(dec_layers)
            ]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(dec_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.LayerNorm(hidden_size),
                    ]
                )
                for _ in range(dec_layers)
            ]
        )

        self.to_feature = nn.Linear(hidden_size, 256)
        self.refine_14 = ResidualBlock(256)
        self.up_28 = UpsampleBlock(256, 192)
        self.up_56 = UpsampleBlock(192, 128)
        self.up_112 = UpsampleBlock(128, 96)
        self.up_224 = UpsampleBlock(96, 64)
        self.mask_head = nn.Conv2d(64, 1, kernel_size=1)

    def encode_style(self, refs: torch.Tensor) -> torch.Tensor:
        """
        refs: [B, K, 3, image_size, image_size]
        반환: [B, K * (1 + style_tokens_side^2), hidden_size]
        """
        if refs.ndim != 5:
            raise InferenceServiceError(
                f"참조 텐서는 5차원이어야 합니다: {tuple(refs.shape)}"
            )

        batch, k = refs.shape[:2]
        flat = refs.flatten(0, 1)

        # ViT-MAE 내부 random noise에 의해 patch 순서가 섞이지 않도록
        # 학습 코드와 동일한 단조 증가 noise를 전달한다.
        deterministic_noise = torch.arange(
            self.n_side**2,
            device=flat.device,
            dtype=flat.dtype,
        ).unsqueeze(0).expand(flat.shape[0], -1)

        tokens = self.style_encoder(
            pixel_values=flat,
            noise=deterministic_noise,
        ).last_hidden_state

        expected_tokens = self.n_side**2 + 1
        if tokens.shape[1] != expected_tokens:
            raise InferenceServiceError(
                "ViT-MAE 출력 토큰 수가 예상과 다릅니다: "
                f"actual={tokens.shape[1]}, expected={expected_tokens}"
            )

        cls_token = tokens[:, :1]
        patches = tokens[:, 1:].transpose(1, 2).reshape(
            batch * k,
            tokens.shape[-1],
            self.n_side,
            self.n_side,
        )
        pooled = F.adaptive_avg_pool2d(
            patches,
            (self.style_tokens_side, self.style_tokens_side),
        )
        pooled = pooled.flatten(2).transpose(1, 2)
        compact = torch.cat([cls_token, pooled], dim=1)
        compact = compact.reshape(batch, k * compact.shape[1], -1)
        return self.style_norm(compact)

    def decode_style(
        self,
        style: torch.Tensor,
        char_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        동일한 style를 여러 한글 문자에 재사용한다.

        style: [1 또는 B, style_tokens, hidden_size]
        char_idx: [B]
        반환: [B, 1, image_size, image_size] logits
        """
        if style.ndim != 3:
            raise InferenceServiceError(
                f"style 텐서는 3차원이어야 합니다: {tuple(style.shape)}"
            )
        if char_idx.ndim != 1:
            raise InferenceServiceError(
                f"char_idx는 1차원이어야 합니다: {tuple(char_idx.shape)}"
            )

        batch = int(char_idx.shape[0])
        if style.shape[0] == 1 and batch > 1:
            style = style.expand(batch, -1, -1)
        elif style.shape[0] != batch:
            raise InferenceServiceError(
                "style batch와 char_idx batch가 다릅니다: "
                f"style={style.shape[0]}, char_idx={batch}"
            )

        hidden = self.query_pos.expand(batch, -1, -1)
        hidden = hidden + self.char_emb(char_idx).unsqueeze(1)

        for self_attn, cross_attn, ffn, norms in zip(
            self.self_attn,
            self.cross_attn,
            self.ffn,
            self.norms,
        ):
            query = norms[0](hidden)
            hidden = hidden + self_attn(
                query,
                query,
                query,
                need_weights=False,
            )[0]
            hidden = hidden + cross_attn(
                norms[1](hidden),
                style,
                style,
                need_weights=False,
            )[0]
            hidden = hidden + ffn(norms[2](hidden))

        feature = self.to_feature(hidden).transpose(1, 2).reshape(
            batch,
            256,
            self.n_side,
            self.n_side,
        )
        feature = self.refine_14(feature)
        feature = self.up_28(feature)
        feature = self.up_56(feature)
        feature = self.up_112(feature)
        feature = self.up_224(
            feature,
            size=(self.image_size, self.image_size),
        )
        return self.mask_head(feature)

    def forward(
        self,
        refs: torch.Tensor,
        char_idx: torch.Tensor,
    ) -> torch.Tensor:
        style = self.encode_style(refs)
        return self.decode_style(style, char_idx)


# ============================================================
# 6. 런타임 및 모델 캐시
# ============================================================


@dataclass(frozen=True)
class InferenceRuntime:
    model: Stage2SharpModel
    model_path: Path
    format_version: int
    characters: tuple[str, ...]
    character_codes: tuple[str, ...]
    image_size: int
    k_refs: int
    threshold: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    style_tokens_side: int
    decoder_layers: int
    decoder_heads: int
    source_step: int | None
    source_best_score: float | None


_MODEL_CACHE_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()
_MODEL_CACHE: dict[tuple[str, int, str], InferenceRuntime] = {}


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolved_device() -> str:
    requested = SETTINGS.device.lower()

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise InferenceServiceError(
            "HANGUL_DEVICE가 cuda이지만 CUDA를 사용할 수 없습니다."
        )
    return SETTINGS.device


def _model_signature(model_path: Path) -> tuple[str, int, str]:
    resolved = model_path.expanduser().resolve()

    try:
        modified = resolved.stat().st_mtime_ns
    except FileNotFoundError as exc:
        raise InferenceServiceError(
            f"로컬 inference_fp32.pt가 없습니다: {resolved}"
        ) from exc

    return (str(resolved), modified, _resolved_device())


def load_runtime(
    model_path: str | os.PathLike[str] | None = None,
) -> InferenceRuntime:
    path = Path(model_path or SETTINGS.model_path)
    signature = _model_signature(path)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(signature)
        if cached is not None:
            return cached

        try:
            checkpoint = torch.load(
                signature[0],
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise InferenceServiceError(
                f"추론 모델 로드에 실패했습니다: {signature[0]}"
            ) from exc

        if not isinstance(checkpoint, Mapping):
            raise InferenceServiceError(
                "inference_fp32.pt 최상위 형식이 dict가 아닙니다."
            )

        _validate_inference_payload(checkpoint)

        state_dict = _extract_state_dict(checkpoint)
        architecture = _as_dict(checkpoint.get("architecture"))
        preprocess = _as_dict(checkpoint.get("preprocess"))

        vit_config = architecture.get("vit_config")
        if not isinstance(vit_config, Mapping):
            raise InferenceServiceError(
                "architecture.vit_config가 없습니다."
            )

        characters_raw = checkpoint.get("characters")
        if not isinstance(characters_raw, Sequence) or isinstance(
            characters_raw, (str, bytes)
        ):
            raise InferenceServiceError(
                "characters가 문자 배열 형식이 아닙니다."
            )
        characters = tuple(str(char) for char in characters_raw)

        if not characters:
            raise InferenceServiceError("characters가 비어 있습니다.")
        if len(set(characters)) != len(characters):
            raise InferenceServiceError("characters에 중복 문자가 있습니다.")

        num_chars = int(
            architecture.get(
                "num_chars",
                state_dict["char_emb.weight"].shape[0],
            )
        )
        embedding_chars = int(state_dict["char_emb.weight"].shape[0])

        if num_chars != embedding_chars or num_chars != len(characters):
            raise InferenceServiceError(
                "체크포인트 문자 수가 서로 다릅니다: "
                f"architecture={num_chars}, "
                f"embedding={embedding_chars}, "
                f"characters={len(characters)}"
            )

        image_size = int(
            preprocess.get(
                "image_size",
                SETTINGS.default_image_size,
            )
        )
        style_tokens_side = int(
            architecture.get(
                "style_tokens_side",
                SETTINGS.default_style_tokens_side,
            )
        )
        decoder_layers = int(
            architecture.get(
                "dec_layers",
                _infer_decoder_layers(state_dict),
            )
        )
        decoder_heads = int(
            architecture.get(
                "dec_heads",
                SETTINGS.default_dec_heads,
            )
        )

        checkpoint_k_refs = int(
            preprocess.get("k_refs", SETTINGS.default_k_refs)
        )
        k_refs = (
            SETTINGS.k_refs_override
            if SETTINGS.k_refs_override is not None
            else checkpoint_k_refs
        )
        if k_refs <= 0:
            raise InferenceServiceError(
                f"k_refs는 1 이상이어야 합니다: {k_refs}"
            )

        checkpoint_threshold = float(
            preprocess.get(
                "threshold",
                SETTINGS.default_threshold,
            )
        )
        threshold = (
            SETTINGS.threshold_override
            if SETTINGS.threshold_override is not None
            else checkpoint_threshold
        )
        if not 0.0 <= threshold <= 1.0:
            raise InferenceServiceError(
                f"threshold는 0~1 범위여야 합니다: {threshold}"
            )

        mean = _parse_float_triplet(
            preprocess.get("mean"),
            field_name="preprocess.mean",
            default=SETTINGS.default_mean,
        )
        std = _parse_float_triplet(
            preprocess.get("std"),
            field_name="preprocess.std",
            default=SETTINGS.default_std,
        )

        model = Stage2SharpModel(
            vit_config_dict=vit_config,
            image_size=image_size,
            num_chars=num_chars,
            dec_layers=decoder_layers,
            dec_heads=decoder_heads,
            style_tokens_side=style_tokens_side,
        )

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise InferenceServiceError(
                "Stage2SharpModel 구조와 inference_fp32.pt의 state_dict가 "
                "일치하지 않습니다. 학습 환경의 transformers 버전도 "
                "함께 확인해야 합니다."
            ) from exc

        model.to(signature[2])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        character_codes_raw = checkpoint.get("character_codes", ())
        if isinstance(character_codes_raw, Sequence) and not isinstance(
            character_codes_raw, (str, bytes)
        ):
            character_codes = tuple(
                str(code) for code in character_codes_raw
            )
        else:
            character_codes = tuple(
                f"U+{ord(char):04X}" for char in characters
            )

        runtime = InferenceRuntime(
            model=model,
            model_path=Path(signature[0]),
            format_version=int(checkpoint["format_version"]),
            characters=characters,
            character_codes=character_codes,
            image_size=image_size,
            k_refs=int(k_refs),
            threshold=float(threshold),
            mean=mean,
            std=std,
            style_tokens_side=style_tokens_side,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            source_step=(
                int(checkpoint["source_step"])
                if checkpoint.get("source_step") is not None
                else None
            ),
            source_best_score=(
                float(checkpoint["source_best_score"])
                if checkpoint.get("source_best_score") is not None
                else None
            ),
        )

        _MODEL_CACHE.clear()
        _MODEL_CACHE[signature] = runtime
        return runtime


def load_model(
    model_path: str | os.PathLike[str] | None = None,
) -> Stage2SharpModel:
    """기존 백엔드 호출 호환용. 내부적으로 load_runtime을 사용한다."""
    return load_runtime(model_path).model


# ============================================================
# 7. 입력 폰트 다운로드 및 검증
# ============================================================


def download_font_https(font_https_url: str) -> bytes:
    font_bytes = _download_https_bytes(
        font_https_url,
        max_bytes=SETTINGS.max_font_bytes,
        field_name="입력 폰트",
        read_timeout=SETTINGS.font_read_timeout,
    )

    try:
        with TTFont(
            io.BytesIO(font_bytes),
            lazy=True,
            recalcBBoxes=False,
            recalcTimestamp=False,
        ) as font:
            if not font.getBestCmap():
                raise InferenceServiceError(
                    "입력 폰트에 유효한 cmap이 없습니다."
                )
    except InferenceServiceError:
        raise
    except (TTLibError, OSError, ValueError) as exc:
        raise InferenceServiceError(
            "입력 파일이 유효한 TTF/OTF가 아닙니다."
        ) from exc

    return font_bytes


# ============================================================
# 8. 영어 참조 글리프 선택 및 렌더링
# ============================================================


def _available_english_chars(font_bytes: bytes) -> list[str]:
    try:
        with TTFont(io.BytesIO(font_bytes), lazy=True) as font:
            cmap = font.getBestCmap() or {}
    except (TTLibError, OSError, ValueError) as exc:
        raise InferenceServiceError(
            "영어 글리프 목록을 읽지 못했습니다."
        ) from exc

    return [
        char
        for char in DEFAULT_ENGLISH_CANDIDATES
        if ord(char) in cmap
    ]


def _select_structured_reference_chars(
    available: Sequence[str],
    count: int,
    *,
    rng: random.Random | random.Random = random,
) -> list[str]:
    """학습 노트북의 structured_reference_sample과 같은 선택 정책."""
    candidates_all = list(dict.fromkeys(available))
    if not candidates_all:
        raise InferenceServiceError(
            "입력 폰트에서 A-Z/a-z 영어 글리프를 찾지 못했습니다."
        )

    selected: list[str] = []

    for group in REFERENCE_GROUPS:
        candidates = [
            char
            for char in candidates_all
            if char in group and char not in selected
        ]
        if candidates:
            selected.append(rng.choice(candidates))
        if len(selected) == count:
            return selected

    remaining = [
        char for char in candidates_all if char not in selected
    ]
    required = count - len(selected)

    if required > 0 and remaining:
        selected.extend(
            rng.sample(remaining, min(required, len(remaining)))
        )

    if len(selected) < count:
        selected.extend(
            rng.choices(candidates_all, k=count - len(selected))
        )

    return selected[:count]


def _fit_common_font_size(
    font_bytes: bytes,
    chars: Sequence[str],
    canvas_size: int,
    occupancy: float = 0.75,
) -> int:
    max_size = int(canvas_size * occupancy)
    low = 8
    high = canvas_size * 2
    best = low

    while low <= high:
        middle = (low + high) // 2
        try:
            font = ImageFont.truetype(
                io.BytesIO(font_bytes),
                size=middle,
            )
        except (OSError, ValueError) as exc:
            raise InferenceServiceError(
                "Pillow가 입력 폰트를 렌더링하지 못했습니다."
            ) from exc

        fits = True
        for char in chars:
            bbox = font.getbbox(char)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width > max_size or height > max_size:
                fits = False
                break

        if fits:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    return best


def render_random_english_references(
    font_bytes: bytes,
    *,
    k_refs: int | None = None,
    rng: random.Random | None = None,
) -> tuple[list[Image.Image], list[str]]:
    count = k_refs if k_refs is not None else SETTINGS.default_k_refs
    if count <= 0:
        raise InferenceServiceError(
            f"k_refs는 1 이상이어야 합니다: {count}"
        )

    available = _available_english_chars(font_bytes)
    random_source = rng or random
    selected = _select_structured_reference_chars(
        available,
        count,
        rng=random_source,
    )

    font_size = _fit_common_font_size(
        font_bytes,
        selected,
        SETTINGS.render_size,
    )
    try:
        font = ImageFont.truetype(
            io.BytesIO(font_bytes),
            size=font_size,
        )
    except (OSError, ValueError) as exc:
        raise InferenceServiceError(
            "Pillow가 입력 폰트를 렌더링하지 못했습니다."
        ) from exc

    images: list[Image.Image] = []

    for char in selected:
        canvas = Image.new(
            "L",
            (SETTINGS.render_size, SETTINGS.render_size),
            color=255,
        )
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        x = (SETTINGS.render_size - width) / 2 - bbox[0]
        y = (SETTINGS.render_size - height) / 2 - bbox[1]

        draw.text((x, y), char, font=font, fill=0)
        images.append(canvas)

    return images, selected


def _build_transform(runtime: InferenceRuntime) -> T.Compose:
    return T.Compose(
        [
            T.Resize(
                (runtime.image_size, runtime.image_size),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=runtime.mean, std=runtime.std),
        ]
    )


# ============================================================
# 9. 모델 logits -> 기존 128x128 grayscale PNG
# ============================================================


def _logits_to_png(
    logits_chw: torch.Tensor,
    *,
    threshold: float,
    output_size: int,
) -> bytes:
    if logits_chw.ndim != 3 or logits_chw.shape[0] != 1:
        raise InferenceServiceError(
            "모델 출력 shape이 올바르지 않습니다. "
            f"expected=[1,H,W], actual={tuple(logits_chw.shape)}"
        )

    probability = torch.sigmoid(logits_chw[0].float())
    ink = probability >= threshold

    # 학습 target: 흰 배경=0, 검은 글자=1 잉크 마스크
    # PNG 출력: 흰 배경=255, 검은 글자=0
    grayscale = (
        (1.0 - ink.float())
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    image = Image.fromarray(grayscale, mode="L")

    if image.size != (output_size, output_size):
        image = image.resize(
            (output_size, output_size),
            resample=Image.Resampling.NEAREST,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# ============================================================
# 10. 공개 서비스 API
# ============================================================


def initialize_service(
    *,
    force_model_sync: bool = False,
) -> dict[str, Any]:
    import time

    total_start = time.perf_counter()

    print(f"[초기화] device={SETTINGS.device}", flush=True)
    print(f"[초기화] model_url={SETTINGS.model_url}", flush=True)
    print(f"[초기화] model_path={SETTINGS.model_path}", flush=True)

    sync_start = time.perf_counter()
    print("[1/2] inference_fp32.pt 준비 중...", flush=True)
    model_path = sync_model_from_https(force=force_model_sync)
    print(
        f"[1/2] 모델 파일 준비 완료 "
        f"({time.perf_counter() - sync_start:.1f}초)",
        flush=True,
    )

    load_start = time.perf_counter()
    print("[2/2] Stage2SharpModel 로딩 중...", flush=True)
    runtime = load_runtime(model_path)
    print(
        f"[2/2] 모델 로드 완료 "
        f"({time.perf_counter() - load_start:.1f}초)",
        flush=True,
    )

    metadata = _read_model_metadata(model_path)
    print(
        f"[초기화 완료] 총 {time.perf_counter() - total_start:.1f}초",
        flush=True,
    )

    return {
        "status": "ready",
        "device": SETTINGS.device,
        "model_path": str(model_path),
        "model_url": metadata.get("model_url", SETTINGS.model_url),
        "model_sha256": metadata.get("sha256"),
        "format_version": runtime.format_version,
        "target_chars": list(runtime.characters),
        "character_codes": list(runtime.character_codes),
        "image_size": runtime.image_size,
        "output_size": SETTINGS.output_size,
        "k_refs": runtime.k_refs,
        "threshold": runtime.threshold,
        "style_tokens_side": runtime.style_tokens_side,
        "decoder_layers": runtime.decoder_layers,
        "decoder_heads": runtime.decoder_heads,
        "source_step": runtime.source_step,
        "source_best_score": runtime.source_best_score,
    }


def split_english_pngs_from_font_url(
    font_https_url: str,
    *,
    k_refs: int | None = None,
) -> dict[str, bytes]:
    """입력 폰트에서 모델 참조용 영어 글리프 PNG를 분리한다."""
    font_bytes = download_font_https(font_https_url)

    if k_refs is None:
        try:
            model_path = sync_model_from_https()
            k_refs = load_runtime(model_path).k_refs
        except InferenceServiceError:
            # 모델 준비 없이도 영어 PNG 분리 기능은 사용할 수 있게 한다.
            k_refs = SETTINGS.default_k_refs

    images, selected_chars = render_random_english_references(
        font_bytes,
        k_refs=k_refs,
    )

    pngs: dict[str, bytes] = {}
    for char, image in zip(selected_chars, images):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        pngs[char] = buffer.getvalue()

    return pngs


def save_english_pngs(
    pngs: Mapping[str, bytes],
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    """대소문자 파일명 충돌 없이 영어 참조 PNG를 저장한다."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    duplicate_counts: dict[str, int] = {}

    for char, data in pngs.items():
        prefix = "upper" if char.isupper() else "lower"
        base_name = f"{prefix}_{char}"
        duplicate_index = duplicate_counts.get(base_name, 0)
        duplicate_counts[base_name] = duplicate_index + 1

        suffix = "" if duplicate_index == 0 else f"_{duplicate_index}"
        path = output / f"{base_name}{suffix}.png"
        path.write_bytes(data)
        saved.append(path)

    return saved


def _generate_hangul_result(
    font_https_url: str,
) -> tuple[dict[str, bytes], list[str], InferenceRuntime]:
    model_path = sync_model_from_https()
    runtime = load_runtime(model_path)

    font_bytes = download_font_https(font_https_url)
    reference_images, selected_chars = render_random_english_references(
        font_bytes,
        k_refs=runtime.k_refs,
    )

    transform = _build_transform(runtime)
    ref_tensor = torch.stack(
        [transform(image) for image in reference_images]
    ).unsqueeze(0).to(
        SETTINGS.device,
        non_blocking=True,
    )

    char_count = len(runtime.characters)
    char_indices = torch.arange(
        char_count,
        device=SETTINGS.device,
        dtype=torch.long,
    )

    # 모델 하나를 여러 요청이 동시에 사용하며 GPU 메모리를 과도하게
    # 점유하지 않도록 기본적으로 추론 구간을 직렬화한다.
    with _INFERENCE_LOCK, torch.inference_mode():
        # 같은 영어 참조를 14회 ViT에 넣지 않고 한 번만 인코딩한다.
        style = runtime.model.encode_style(ref_tensor)
        logits = runtime.model.decode_style(style, char_indices)

    if logits.shape != (
        char_count,
        1,
        runtime.image_size,
        runtime.image_size,
    ):
        raise InferenceServiceError(
            "Stage2SharpModel 최종 출력 shape이 올바르지 않습니다: "
            f"actual={tuple(logits.shape)}"
        )

    pngs = {
        char: _logits_to_png(
            logits[index],
            threshold=runtime.threshold,
            output_size=SETTINGS.output_size,
        )
        for index, char in enumerate(runtime.characters)
    }

    return pngs, selected_chars, runtime


def generate_hangul_pngs(
    font_https_url: str,
) -> dict[str, bytes]:
    """
    기존 백엔드 계약을 유지하는 기본 함수.

    입력:
        사용자 TTF/OTF의 S3 HTTPS URL

    반환:
        {"가": PNG bytes, ..., "히": PNG bytes}
    """
    pngs, _, _ = _generate_hangul_result(font_https_url)
    return pngs


def generate_hangul_pngs_with_metadata(
    font_https_url: str,
) -> dict[str, Any]:
    """개발 및 디버깅용 메타데이터 포함 함수."""
    pngs, selected_chars, runtime = _generate_hangul_result(font_https_url)

    return {
        "selected_reference_chars": selected_chars,
        "characters": list(runtime.characters),
        "character_codes": list(runtime.character_codes),
        "k_refs": runtime.k_refs,
        "threshold": runtime.threshold,
        "model_image_size": runtime.image_size,
        "output_size": SETTINGS.output_size,
        "pngs": pngs,
    }


def save_pngs(
    pngs: Mapping[str, bytes],
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    """generate_hangul_pngs 결과를 PNG 파일로 저장하고 검증한다."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not pngs:
        raise InferenceServiceError("저장할 한글 PNG 결과가 없습니다.")

    saved: list[Path] = []

    for char, data in pngs.items():
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise InferenceServiceError(
                f"'{char}' PNG 데이터가 비어 있거나 bytes가 아닙니다."
            )

        path = output / f"{char}.png"
        path.write_bytes(bytes(data))

        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (
                    SETTINGS.output_size,
                    SETTINGS.output_size,
                ):
                    raise InferenceServiceError(
                        f"{path.name} 크기 오류: {image.size}"
                    )
                if image.mode != "L":
                    raise InferenceServiceError(
                        f"{path.name} mode 오류: {image.mode}"
                    )
        except OSError as exc:
            raise InferenceServiceError(
                f"{path.name}이 유효한 PNG가 아닙니다."
            ) from exc

        saved.append(path)

    return saved


# ============================================================
# 11. 단독 실행 테스트
# ============================================================

if __name__ == "__main__":
    import time
    import traceback

    TEST_FONT_URL = "https://fontify-986995923828-ap-northeast-2-an.s3.ap-northeast-2.amazonaws.com/english_only_google_fonts/abeezee/ABeeZee-Italic.ttf"
    OUTPUT_DIR = Path(__file__).resolve().parent / "generated_hangul"

    try:
        total_start = time.perf_counter()

        print("=" * 70, flush=True)
        print("inference_fp32.pt 한글 생성 테스트 시작", flush=True)
        print(f"모델 URL: {SETTINGS.model_url}", flush=True)
        print(f"입력 폰트 URL: {TEST_FONT_URL}", flush=True)
        print(f"결과 저장 폴더: {OUTPUT_DIR}", flush=True)
        print("=" * 70, flush=True)

        print("\n[1/3] 추론 서비스 초기화", flush=True)
        service_info = initialize_service()
        print(
            json.dumps(
                service_info,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            flush=True,
        )

        print("\n[2/3] 한글 14자 생성", flush=True)
        inference_start = time.perf_counter()
        result = generate_hangul_pngs_with_metadata(TEST_FONT_URL)
        inference_seconds = time.perf_counter() - inference_start

        print(
            "선택된 영어 참조 문자:",
            result["selected_reference_chars"],
            flush=True,
        )
        print(f"threshold: {result['threshold']}", flush=True)
        print(f"추론 완료: {inference_seconds:.1f}초", flush=True)

        print("\n[3/3] PNG 저장 및 검증", flush=True)
        saved_files = save_pngs(result["pngs"], OUTPUT_DIR)

        print(f"한글 PNG {len(saved_files)}개 저장 완료", flush=True)
        for path in saved_files:
            print(f"  - {path}", flush=True)

        total_seconds = time.perf_counter() - total_start
        print("\n" + "=" * 70, flush=True)
        print(f"전체 테스트 완료: {total_seconds:.1f}초", flush=True)
        print(f"결과 폴더: {OUTPUT_DIR.resolve()}", flush=True)
        print("=" * 70, flush=True)

    except Exception as exc:
        print("\n" + "=" * 70, flush=True)
        print("Stage2 추론 테스트 실패", flush=True)
        print(f"오류 종류: {type(exc).__name__}", flush=True)
        print(f"오류 내용: {exc}", flush=True)
        print("=" * 70, flush=True)
        traceback.print_exc()
        raise
