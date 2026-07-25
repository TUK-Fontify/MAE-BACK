"""
hangul_font_inference_service.py

서비스 구조
-----------
1. Stage2 모델(best.pt)은 S3 HTTPS URL에서 로컬로 한 번 다운로드한다.
2. 실제 torch.load()는 항상 로컬 체크포인트를 사용한다.
3. 입력 TTF/OTF는 요청마다 전달받은 S3 HTTPS URL에서 다운로드한다.
4. 영문 A-Z/a-z 중 폰트가 지원하는 글리프에서 매 요청마다 K개를 무작위 선택한다.
5. Stage2 모델로 한글 14자를 생성하고 각 결과를 128x128 grayscale PNG bytes로 반환한다.
6. 모델 URL을 바꾸거나 force=True로 동기화하면 새 모델로 교체할 수 있다.

필수 패키지
-----------
pip install torch torchvision transformers pillow fonttools requests

환경변수 예시
-------------
HANGUL_MODEL_URL=https://.../best.pt
HANGUL_MODEL_PATH=./models/stage2/best.pt
HANGUL_DEVICE=cuda
HANGUL_K_REFS=8

백엔드 사용 예시
----------------
from hangul_font_inference_service import initialize_service, generate_hangul_pngs

initialize_service()  # 서버 시작 시 한 번 호출
result = generate_hangul_pngs(font_https_url)
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
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
from fontTools.ttLib import TTFont, TTLibError
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from transformers import ViTMAEConfig, ViTMAEForPreTraining
from urllib3.util.retry import Retry


# ============================================================
# 1. 설정
# ============================================================

@dataclass(frozen=True)
class Settings:
    # S3에 업로드된 Stage2 best.pt HTTPS 주소
    model_url: str = os.getenv(
        "HANGUL_MODEL_URL",
        "https://fontify-986995923828-ap-northeast-2-an.s3.ap-northeast-2.amazonaws.com/best.pt",
    ).strip()

    # S3 모델을 서버 내부에 내려받아 저장할 로컬 위치
    # 실행 위치가 달라져도 항상 이 Python 파일 기준으로 저장한다.
    model_path: Path = Path(
    os.getenv(
        "HANGUL_MODEL_PATH",
        str(
            Path(__file__).resolve().parent
            / "models"
            / "stage2"
            / "best.pt"
        ),
    )
).expanduser()

    device: str = os.getenv(
        "HANGUL_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    target_chars: tuple[str, ...] = tuple(
        "가나더려모부쇼야져쵸켜튜프히"
    )

    english_candidates: tuple[str, ...] = tuple(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )

    k_refs: int = int(
        os.getenv("HANGUL_K_REFS", "8")
    )

    render_size: int = 128
    encoder_size: int = 224
    output_size: int = 128

    norm_mean: tuple[float, float, float] = (
        0.485,
        0.456,
        0.406,
    )
    norm_std: tuple[float, float, float] = (
        0.229,
        0.224,
        0.225,
    )

    default_dec_layers: int = 4
    default_dec_heads: int = 8
    default_mask_ratio: float = 0.75

    connect_timeout: float = 5.0
    read_timeout: float = 120.0

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


SETTINGS = Settings()


class InferenceServiceError(RuntimeError):
    """외부 API 계층에서 처리할 추론 서비스 예외."""


# ============================================================
# 2. HTTP 세션
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
            timeout=(
                SETTINGS.connect_timeout,
                read_timeout,
            ),
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            declared = response.headers.get("Content-Length")
            total_size = int(declared) if declared else None

            if total_size and total_size > max_bytes:
                raise InferenceServiceError(
                    f"{field_name} 크기가 제한을 초과합니다."
                )

            if total_size:
                print(
                    f"[전체 크기] "
                    f"{total_size / 1024 / 1024:.1f} MB",
                    flush=True,
                )

            last_report = 0

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                data.extend(chunk)

                if len(data) > max_bytes:
                    raise InferenceServiceError(
                        f"{field_name} 크기가 제한을 초과합니다."
                    )

                downloaded_mb = len(data) / 1024 / 1024

                if downloaded_mb - last_report >= 50:
                    last_report = downloaded_mb

                    if total_size:
                        percent = len(data) / total_size * 100
                        print(
                            f"[다운로드 중] "
                            f"{downloaded_mb:.1f} MB "
                            f"({percent:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[다운로드 중] "
                            f"{downloaded_mb:.1f} MB",
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
        f"[다운로드 완료] "
        f"{len(data) / 1024 / 1024:.1f} MB",
        flush=True,
    )

    return bytes(data)


# ============================================================
# 3. S3 HTTPS 모델 → 로컬 파일 동기화
# ============================================================

_MODEL_FILE_LOCK = threading.RLock()


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(
        model_path.suffix + ".metadata.json"
    )


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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temp, path)


def sync_model_from_https(
    *,
    model_url: str | None = None,
    local_path: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> Path:
    """
    모델 S3 HTTPS URL을 로컬 파일로 동기화한다.

    기본 동작:
    - 로컬 파일이 있고 metadata의 URL도 같으면 다운로드하지 않는다.
    - force=True이면 다시 다운로드한다.
    - 다운로드는 .download 파일에 먼저 저장한 뒤 os.replace로 원자적 교체한다.
    """
    url = (model_url or SETTINGS.model_url).strip()
    model_path = Path(
        local_path or SETTINGS.model_path
    ).expanduser().resolve()

    if not url:
        if model_path.exists():
            return model_path

        raise InferenceServiceError(
            "HANGUL_MODEL_URL이 없고 로컬 체크포인트도 없습니다. "
            "모델 HTTPS URL 또는 로컬 모델 파일을 지정해야 합니다."
        )

    _validate_https_url(
        url,
        field_name="모델 URL",
    )

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
            field_name="모델 체크포인트",
            read_timeout=SETTINGS.read_timeout,
        )

        # torch 체크포인트인지 최소 검증한다.
        try:
            checkpoint = torch.load(
                io.BytesIO(model_bytes),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise InferenceServiceError(
                "다운로드한 파일이 유효한 PyTorch 체크포인트가 아닙니다."
            ) from exc

        if not isinstance(checkpoint, Mapping):
            raise InferenceServiceError(
                "Stage2 체크포인트 형식이 dict가 아닙니다."
            )

        if not any(
            key in checkpoint
            for key in ("model", "model_state_dict", "state_dict")
        ):
            raise InferenceServiceError(
                "체크포인트에 model/state_dict가 없습니다."
            )

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = model_path.with_suffix(
            model_path.suffix + ".download"
        )
        temp_path.write_bytes(model_bytes)

        # 같은 파일 시스템에서 원자적으로 교체한다.
        os.replace(temp_path, model_path)

        _write_model_metadata(
            model_path,
            model_url=url,
            sha256=_sha256(model_bytes),
            size=len(model_bytes),
        )

        # 다음 load_model에서 새 파일을 읽도록 기존 모델 캐시를 비운다.
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
    for key in (
        "model",
        "model_state_dict",
        "state_dict",
    ):
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            result = dict(state)
            break
    else:
        raise InferenceServiceError(
            "체크포인트에서 모델 state_dict를 찾지 못했습니다."
        )

    if any(key.startswith("module.") for key in result):
        result = {
            key.removeprefix("module."): value
            for key, value in result.items()
        }

    required = {
        "pos_query",
        "char_emb.weight",
        "style_encoder.embeddings.cls_token",
        "to_pixels.weight",
    }
    missing = sorted(required.difference(result))

    if missing:
        raise InferenceServiceError(
            "Stage2 필수 파라미터가 없습니다: "
            + ", ".join(missing)
        )

    return result


def _infer_decoder_layers(
    state_dict: Mapping[str, torch.Tensor],
) -> int:
    indexes: list[int] = []

    for key in state_dict:
        if not key.startswith("cross_attn_layers."):
            continue

        index = key.split(".")[1]
        if index.isdigit():
            indexes.append(int(index))

    return (
        max(indexes) + 1
        if indexes
        else SETTINGS.default_dec_layers
    )


def _pick_config(
    config: Mapping[str, Any],
    names: tuple[str, ...],
    default: Any,
) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


# ============================================================
# 5. Stage2 모델
# ============================================================

class Stage2Model(nn.Module):
    def __init__(
        self,
        *,
        num_chars: int,
        dec_layers: int,
        dec_heads: int,
        mask_ratio: float,
    ) -> None:
        super().__init__()

        vit_config = ViTMAEConfig(
            mask_ratio=mask_ratio,
        )
        scaffold = ViTMAEForPreTraining(
            vit_config,
        )
        self.style_encoder = scaffold.vit
        del scaffold

        hidden_size = (
            self.style_encoder.config.hidden_size
        )
        self.patch_size = (
            self.style_encoder.config.patch_size
        )
        self.image_size = (
            self.style_encoder.config.image_size
        )
        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2

        self.char_emb = nn.Embedding(
            num_chars,
            hidden_size,
        )
        self.pos_query = nn.Parameter(
            torch.randn(
                1,
                self.num_patches,
                hidden_size,
            ) * 0.02
        )

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_size,
                dec_heads,
                batch_first=True,
            )
            for _ in range(dec_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_size,
                dec_heads,
                batch_first=True,
            )
            for _ in range(dec_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    hidden_size,
                    hidden_size * 4,
                ),
                nn.GELU(),
                nn.Linear(
                    hidden_size * 4,
                    hidden_size,
                ),
            )
            for _ in range(dec_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(dec_layers * 3)
        ])
        self.to_pixels = nn.Linear(
            hidden_size,
            self.patch_size
            * self.patch_size
            * 3,
        )

    def encode_style(
        self,
        ref_images: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, ref_count = ref_images.shape[:2]
        flat = ref_images.flatten(0, 1)

        # Stage2 학습 코드와 동일하게 고정 noise를 전달하지 않는다.
        tokens = self.style_encoder(
            pixel_values=flat,
        ).last_hidden_state

        return tokens.reshape(
            batch_size,
            ref_count * tokens.shape[1],
            tokens.shape[2],
        )

    def forward(
        self,
        ref_images: torch.Tensor,
        char_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = ref_images.shape[0]
        style_tokens = self.encode_style(
            ref_images,
        )

        hidden = (
            self.pos_query.expand(
                batch_size,
                -1,
                -1,
            )
            + self.char_emb(
                char_indices,
            ).unsqueeze(1)
        )

        for index in range(
            len(self.cross_attn_layers)
        ):
            norm_offset = index * 3

            normalized = self.norms[
                norm_offset
            ](hidden)
            self_output, _ = (
                self.self_attn_layers[index](
                    normalized,
                    normalized,
                    normalized,
                    need_weights=False,
                )
            )
            hidden = hidden + self_output

            cross_output, _ = (
                self.cross_attn_layers[index](
                    self.norms[
                        norm_offset + 1
                    ](hidden),
                    style_tokens,
                    style_tokens,
                    need_weights=False,
                )
            )
            hidden = hidden + cross_output

            hidden = hidden + self.ffns[index](
                self.norms[
                    norm_offset + 2
                ](hidden)
            )

        patches = self.to_pixels(hidden)

        patch = self.patch_size
        side = self.image_size // patch

        images = patches.reshape(
            batch_size,
            side,
            side,
            patch,
            patch,
            3,
        )
        return images.permute(
            0,
            5,
            1,
            3,
            2,
            4,
        ).reshape(
            batch_size,
            3,
            self.image_size,
            self.image_size,
        )


# ============================================================
# 6. 모델 캐시
# ============================================================

_MODEL_CACHE_LOCK = threading.RLock()
_MODEL_CACHE: dict[
    tuple[str, int, str],
    Stage2Model,
] = {}


def clear_model_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_signature(
    model_path: Path,
) -> tuple[str, int, str]:
    resolved = model_path.expanduser().resolve()

    try:
        modified = resolved.stat().st_mtime_ns
    except FileNotFoundError as exc:
        raise InferenceServiceError(
            f"로컬 체크포인트가 없습니다: {resolved}"
        ) from exc

    return (
        str(resolved),
        modified,
        SETTINGS.device,
    )


def load_model(
    model_path: str | os.PathLike[str] | None = None,
) -> Stage2Model:
    path = Path(
        model_path or SETTINGS.model_path
    )
    signature = _model_signature(path)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(signature)
        if cached is not None:
            return cached

        checkpoint = torch.load(
            signature[0],
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, Mapping):
            raise InferenceServiceError(
                "체크포인트 형식이 dict가 아닙니다."
            )

        state_dict = _extract_state_dict(
            checkpoint,
        )
        config = _as_dict(
            checkpoint.get("config")
        )

        dec_layers = int(_pick_config(
            config,
            (
                "dec_layers",
                "decoder_layers",
                "num_decoder_layers",
            ),
            _infer_decoder_layers(state_dict),
        ))
        dec_heads = int(_pick_config(
            config,
            (
                "dec_heads",
                "decoder_heads",
                "num_attention_heads",
            ),
            SETTINGS.default_dec_heads,
        ))
        mask_ratio = float(_pick_config(
            config,
            (
                "mask_ratio",
                "mae_mask_ratio",
            ),
            SETTINGS.default_mask_ratio,
        ))

        num_chars = int(
            state_dict["char_emb.weight"].shape[0]
        )

        if num_chars != len(
            SETTINGS.target_chars
        ):
            raise InferenceServiceError(
                "체크포인트 문자 수와 TARGET_CHARS 수가 다릅니다. "
                f"checkpoint={num_chars}, "
                f"service={len(SETTINGS.target_chars)}"
            )

        model = Stage2Model(
            num_chars=num_chars,
            dec_layers=dec_layers,
            dec_heads=dec_heads,
            mask_ratio=mask_ratio,
        )

        try:
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        except RuntimeError as exc:
            raise InferenceServiceError(
                "Stage2Model 구조와 체크포인트가 일치하지 않습니다."
            ) from exc

        model.to(SETTINGS.device)
        model.eval()

        _MODEL_CACHE.clear()
        _MODEL_CACHE[signature] = model
        return model


# ============================================================
# 7. 입력 폰트 HTTPS 다운로드 및 검증
# ============================================================

def download_font_https(
    font_https_url: str,
) -> bytes:
    font_bytes = _download_https_bytes(
        font_https_url,
        max_bytes=SETTINGS.max_font_bytes,
        field_name="입력 폰트",
        read_timeout=30.0,
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
    except (
        TTLibError,
        OSError,
        ValueError,
    ) as exc:
        raise InferenceServiceError(
            "입력 파일이 유효한 TTF/OTF가 아닙니다."
        ) from exc

    return font_bytes


# ============================================================
# 8. 영어 참조 글리프 무작위 선택 및 렌더링
# ============================================================

def _available_english_chars(
    font_bytes: bytes,
) -> list[str]:
    with TTFont(
        io.BytesIO(font_bytes),
        lazy=True,
    ) as font:
        cmap = font.getBestCmap() or {}

    return [
        char
        for char in SETTINGS.english_candidates
        if ord(char) in cmap
    ]


def _fit_common_font_size(
    font_bytes: bytes,
    chars: list[str],
    canvas_size: int,
    occupancy: float = 0.75,
) -> int:
    max_size = int(
        canvas_size * occupancy
    )
    low = 8
    high = canvas_size * 2
    best = low

    while low <= high:
        middle = (low + high) // 2
        font = ImageFont.truetype(
            io.BytesIO(font_bytes),
            size=middle,
        )

        fits = True
        for char in chars:
            bbox = font.getbbox(char)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if (
                width > max_size
                or height > max_size
            ):
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
) -> tuple[list[Image.Image], list[str]]:
    count = (
        SETTINGS.k_refs
        if k_refs is None
        else k_refs
    )

    available = _available_english_chars(
        font_bytes,
    )

    if len(available) < count:
        raise InferenceServiceError(
            "영어 참조 글리프가 부족합니다. "
            f"available={len(available)}, "
            f"required={count}"
        )

    # seed를 고정하지 않는다.
    selected = random.sample(
        available,
        count,
    )

    font_size = _fit_common_font_size(
        font_bytes,
        selected,
        SETTINGS.render_size,
    )
    font = ImageFont.truetype(
        io.BytesIO(font_bytes),
        size=font_size,
    )

    images: list[Image.Image] = []

    for char in selected:
        canvas = Image.new(
            "L",
            (
                SETTINGS.render_size,
                SETTINGS.render_size,
            ),
            color=255,
        )
        draw = ImageDraw.Draw(canvas)

        bbox = draw.textbbox(
            (0, 0),
            char,
            font=font,
        )
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        x = (
            SETTINGS.render_size - width
        ) / 2 - bbox[0]
        y = (
            SETTINGS.render_size - height
        ) / 2 - bbox[1]

        draw.text(
            (x, y),
            char,
            font=font,
            fill=0,
        )
        images.append(canvas)

    return images, selected


def _build_transform() -> T.Compose:
    return T.Compose([
        T.Resize(
            (
                SETTINGS.encoder_size,
                SETTINGS.encoder_size,
            ),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.Grayscale(
            num_output_channels=3,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=SETTINGS.norm_mean,
            std=SETTINGS.norm_std,
        ),
    ])


# ============================================================
# 9. 모델 출력 → 128x128 PNG
# ============================================================

def _output_to_png(
    image_chw: torch.Tensor,
) -> bytes:
    if (
        image_chw.ndim != 3
        or image_chw.shape[0] != 3
    ):
        raise InferenceServiceError(
            "모델 출력 shape이 올바르지 않습니다: "
            f"{tuple(image_chw.shape)}"
        )

    mean = torch.tensor(
        SETTINGS.norm_mean,
        dtype=image_chw.dtype,
        device=image_chw.device,
    ).view(3, 1, 1)
    std = torch.tensor(
        SETTINGS.norm_std,
        dtype=image_chw.dtype,
        device=image_chw.device,
    ).view(3, 1, 1)

    image01 = (
        image_chw * std + mean
    ).clamp(0.0, 1.0)

    grayscale = image01.mean(dim=0)

    array = (
        grayscale
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    image = Image.fromarray(
        array,
        mode="L",
    )

    if image.size != (
        SETTINGS.output_size,
        SETTINGS.output_size,
    ):
        image = image.resize(
            (
                SETTINGS.output_size,
                SETTINGS.output_size,
            ),
            resample=Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )
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

    print(
        f"[초기화] device={SETTINGS.device}",
        flush=True,
    )
    print(
        f"[초기화] model_url={SETTINGS.model_url}",
        flush=True,
    )
    print(
        f"[초기화] model_path={SETTINGS.model_path}",
        flush=True,
    )

    sync_start = time.perf_counter()

    print(
        "[1/2] 모델 다운로드 또는 로컬 캐시 확인 중...",
        flush=True,
    )

    model_path = sync_model_from_https(
        force=force_model_sync,
    )

    print(
        f"[1/2] 모델 파일 준비 완료 "
        f"({time.perf_counter() - sync_start:.1f}초)",
        flush=True,
    )

    load_start = time.perf_counter()

    print(
        "[2/2] PyTorch 모델 로딩 중...",
        flush=True,
    )

    load_model(model_path)

    print(
        f"[2/2] 모델 로드 완료 "
        f"({time.perf_counter() - load_start:.1f}초)",
        flush=True,
    )

    metadata = _read_model_metadata(model_path)

    print(
        f"[초기화 완료] 총 "
        f"{time.perf_counter() - total_start:.1f}초",
        flush=True,
    )

    return {
        "status": "ready",
        "device": SETTINGS.device,
        "model_path": str(model_path),
        "model_url": metadata.get("model_url"),
        "model_sha256": metadata.get("sha256"),
        "target_chars": list(
            SETTINGS.target_chars
        ),
        "k_refs": SETTINGS.k_refs,
    }

def split_english_pngs_from_font_url(
    font_https_url: str,
    *,
    k_refs: int | None = None,
) -> dict[str, bytes]:
    """
    입력 TTF/OTF의 S3 HTTPS URL을 받아 영어 글리프를 무작위로 골라
    128x128 grayscale PNG로 나눠 반환한다. (Stage2 모델은 사용하지 않는다.)

    입력:
        font_https_url: 예)
        https://fontify-986995923828-ap-northeast-2-an.s3.ap-northeast-2.amazonaws.com/
        english_only_google_fonts/abeezee/ABeeZee-Italic.ttf 같은 S3 HTTPS URL

    반환:
        {"A": PNG bytes, "b": PNG bytes, ...}
    """
    font_bytes = download_font_https(
        font_https_url,
    )

    images, selected_chars = (
        render_random_english_references(
            font_bytes,
            k_refs=k_refs,
        )
    )

    pngs: dict[str, bytes] = {}
    for char, image in zip(
        selected_chars,
        images,
    ):
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="PNG",
            optimize=True,
        )
        pngs[char] = buffer.getvalue()

    return pngs


def save_english_pngs(
    pngs: Mapping[str, bytes],
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    """
    split_english_pngs_from_font_url 결과를 디스크에 저장한다.

    Windows 등 대소문자를 구분하지 않는 파일시스템에서 'A.png'와
    'a.png'가 충돌하지 않도록 대소문자 접두사를 붙여 저장한다.
    """
    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved: list[Path] = []

    for char, data in pngs.items():
        prefix = (
            "upper"
            if char.isupper()
            else "lower"
        )
        path = output / f"{prefix}_{char}.png"
        path.write_bytes(data)
        saved.append(path)

    return saved


def generate_hangul_pngs(
    font_https_url: str,
) -> dict[str, bytes]:
    """
    백엔드에서 사용할 기본 함수.

    입력:
        사용자가 업로드한 TTF/OTF의 S3 HTTPS URL

    반환:
        {"가": PNG bytes, ..., "히": PNG bytes}
    """
    # 서버 시작 초기화를 누락해도 첫 요청에서 자동 준비한다.
    model_path = sync_model_from_https()
    model = load_model(model_path)

    font_bytes = download_font_https(
        font_https_url,
    )
    reference_images, _ = (
        render_random_english_references(
            font_bytes,
        )
    )

    transform = _build_transform()

    ref_tensor = torch.stack([
        transform(image)
        for image in reference_images
    ]).unsqueeze(0).to(
        SETTINGS.device,
        non_blocking=True,
    )

    char_count = len(
        SETTINGS.target_chars
    )

    ref_batch = ref_tensor.expand(
        char_count,
        -1,
        -1,
        -1,
        -1,
    ).contiguous()

    char_indices = torch.arange(
        char_count,
        device=SETTINGS.device,
        dtype=torch.long,
    )

    with torch.inference_mode():
        output = model(
            ref_batch,
            char_indices,
        )

    return {
        char: _output_to_png(
            output[index]
        )
        for index, char in enumerate(
            SETTINGS.target_chars
        )
    }


def generate_hangul_pngs_with_metadata(
    font_https_url: str,
) -> dict[str, Any]:
    """
    개발/디버깅용 함수.
    이번 요청에서 선택된 영어 참조 글자도 반환한다.
    """
    model_path = sync_model_from_https()
    model = load_model(model_path)

    font_bytes = download_font_https(
        font_https_url,
    )
    reference_images, selected_chars = (
        render_random_english_references(
            font_bytes,
        )
    )

    transform = _build_transform()
    ref_tensor = torch.stack([
        transform(image)
        for image in reference_images
    ]).unsqueeze(0).to(
        SETTINGS.device,
        non_blocking=True,
    )

    char_count = len(
        SETTINGS.target_chars
    )
    ref_batch = ref_tensor.expand(
        char_count,
        -1,
        -1,
        -1,
        -1,
    ).contiguous()
    char_indices = torch.arange(
        char_count,
        device=SETTINGS.device,
        dtype=torch.long,
    )

    with torch.inference_mode():
        output = model(
            ref_batch,
            char_indices,
        )

    pngs = {
        char: _output_to_png(
            output[index]
        )
        for index, char in enumerate(
            SETTINGS.target_chars
        )
    }

    return {
        "selected_reference_chars": selected_chars,
        "pngs": pngs,
    }


def save_pngs(
    pngs: Mapping[str, bytes],
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved: list[Path] = []

    for char in SETTINGS.target_chars:
        data = pngs.get(char)
        if data is None:
            raise InferenceServiceError(
                f"결과에 '{char}'가 없습니다."
            )

        path = output / f"{char}.png"
        path.write_bytes(data)

        with Image.open(path) as image:
            if image.size != (
                SETTINGS.output_size,
                SETTINGS.output_size,
            ):
                raise InferenceServiceError(
                    f"{path.name} 크기 오류: "
                    f"{image.size}"
                )
            if image.mode != "L":
                raise InferenceServiceError(
                    f"{path.name} mode 오류: "
                    f"{image.mode}"
                )

        saved.append(path)

    return saved


# ============================================================
# 11. 단독 실행 테스트
# ============================================================

if __name__ == "__main__":
    import time
    import traceback

    # 테스트할 영어 TTF의 S3 HTTPS URL
    TEST_FONT_URL = "https://fontify-986995923828-ap-northeast-2-an.s3.ap-northeast-2.amazonaws.com/english_only_google_fonts/abeezee/ABeeZee-Italic.ttf"

    # 생성 결과를 저장할 폴더
    OUTPUT_DIR = (
        Path(__file__).resolve().parent
        / "generated_hangul"
    )

    try:
        total_start = time.perf_counter()

        print("=" * 60, flush=True)
        print("Stage2 한글 생성 테스트 시작", flush=True)
        print(f"입력 폰트 URL: {TEST_FONT_URL}", flush=True)
        print(f"결과 저장 폴더: {OUTPUT_DIR}", flush=True)
        print("=" * 60, flush=True)

        # 1. 모델 다운로드/캐시 확인 및 GPU 로드
        print("\n[1/3] Stage2 모델 초기화", flush=True)

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

        # 2. 입력 폰트에서 영어 8글자를 뽑고 한글 14자 생성
        print("\n[2/3] 한글 14자 생성 시작", flush=True)

        inference_start = time.perf_counter()

        result = generate_hangul_pngs_with_metadata(
            TEST_FONT_URL
        )

        inference_seconds = (
            time.perf_counter() - inference_start
        )

        selected_chars = result[
            "selected_reference_chars"
        ]

        print(
            "선택된 영어 참조 문자:",
            selected_chars,
            flush=True,
        )
        print(
            f"추론 완료: {inference_seconds:.1f}초",
            flush=True,
        )

        # 3. 생성된 한글 PNG 저장
        print("\n[3/3] PNG 파일 저장", flush=True)

        saved_files = save_pngs(
            result["pngs"],
            OUTPUT_DIR,
        )

        print(
            f"\n한글 PNG {len(saved_files)}개 저장 완료",
            flush=True,
        )

        for path in saved_files:
            print(f"  - {path}", flush=True)

        total_seconds = (
            time.perf_counter() - total_start
        )

        print("\n" + "=" * 60, flush=True)
        print(
            f"전체 테스트 완료: {total_seconds:.1f}초",
            flush=True,
        )
        print(
            f"결과 폴더: {OUTPUT_DIR.resolve()}",
            flush=True,
        )
        print("=" * 60, flush=True)

    except Exception as exc:
        print("\n" + "=" * 60, flush=True)
        print("Stage2 테스트 실패", flush=True)
        print(f"오류 종류: {type(exc).__name__}", flush=True)
        print(f"오류 내용: {exc}", flush=True)
        print("=" * 60, flush=True)

        traceback.print_exc()
        raise