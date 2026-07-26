# MAE-BACK

MAE로 학습한 Stage2 한글 글리프 생성 모델을 백엔드에서 불러와 추론하는 서비스 코드입니다.
Fontify 프로젝트의 한글 폰트 생성 파이프라인에서, 사용자가 업로드한 영문 폰트(TTF/OTF)를 참고 스타일로 삼아
한글 14자를 생성하는 역할을 담당합니다.

## 서비스 동작

1. Stage2 체크포인트(`inference_fp32.pt`)를 S3 HTTPS URL에서 서버 로컬로 한 번 내려받는다.
2. 체크포인트 내부 `architecture` / `preprocess` / `characters` 메타데이터를 읽어 모델을 구성한다.
3. 요청으로 받은 TTF/OTF S3 HTTPS URL에서 폰트를 내려받는다.
4. 영어 A-Z/a-z 중 폰트가 지원하는 글리프에서 K개를 무작위로 선택해 스타일 참조로 사용한다.
5. Stage2 모델이 한글 14자(가나더려모부쇼야져쵸켜튜프히)의 1채널 mask logits를 생성한다.
6. sigmoid + 체크포인트 threshold로 이진화한다.
7. 각 결과를 128x128 grayscale PNG bytes로 반환한다.

## 요구 사항

```bash
pip install -r requirements.txt
```

## 환경 변수

| 변수 | 설명 |
| --- | --- |
| `HANGUL_MODEL_URL` | 모델 체크포인트 HTTPS URL (S3) |
| `HANGUL_MODEL_PATH` | 서버 내부 로컬 모델 캐시 경로 |
| `HANGUL_DEVICE` | `cuda` 또는 `cpu` |
| `HANGUL_OUTPUT_SIZE` | 외부 반환 PNG 크기 (기본 128) |
| `HANGUL_RENDER_SIZE` | 입력 영문 글리프 렌더링 크기 (기본 128) |
| `HANGUL_K_REFS` | 선택 사항. 설정 시 체크포인트 `k_refs` 값을 덮어씀 |
| `HANGUL_THRESHOLD` | 선택 사항. 설정 시 체크포인트 `threshold` 값을 덮어씀 |
| `HANGUL_MAX_MODEL_BYTES` | 모델 다운로드 최대 크기 |
| `HANGUL_MAX_FONT_BYTES` | 입력 폰트 다운로드 최대 크기 |

## 사용 예시

```python
from MAE_BACK import initialize_service, generate_hangul_pngs

initialize_service()  # 서버 시작 시 한 번 호출
pngs = generate_hangul_pngs(font_https_url)
# {"가": b"...PNG...", ..., "히": b"...PNG..."}
```

`generated_hangul/` 폴더는 위 흐름으로 생성한 샘플 출력물입니다.
