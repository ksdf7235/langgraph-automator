# YouTube Shorts 자동 생성 에이전트

LangGraph를 사용한 상태 기반 워크플로우로 유튜브 쇼츠를 자동 생성하는 에이전트입니다.

## 🚀 기능

- **대본 작성**: Ollama의 `deepseek-r1:14b` 모델을 사용하여 주제에 맞는 60초 이내 쇼츠 대본 자동 생성
- **오디오 생성**: `edge-tts`를 사용하여 한국어 음성으로 대본을 오디오로 변환
- **이미지 생성**: ComfyUI API를 통해 각 장면에 맞는 이미지 자동 생성
- **비디오 편집**: `moviepy`를 사용하여 이미지와 오디오를 결합하고 Ken Burns 효과 적용

## 📋 요구사항

### 하드웨어
- Windows 환경
- RTX 4080 (또는 CUDA 지원 GPU)

### 소프트웨어
- Python 3.13+
- `uv` 패키지 매니저
- Ollama (로컬 실행, `deepseek-r1:14b` 모델 필요)
- ComfyUI (로컬 실행, `http://127.0.0.1:8188`)

## 🛠️ 설치

### 1. 의존성 설치

```bash
# uv를 사용하여 의존성 설치
uv sync
```

또는 수동으로 설치:

```bash
uv add langgraph langchain-ollama langchain-core edge-tts moviepy requests websocket-client pydantic
```

### 2. Ollama 설정

Ollama를 설치하고 `deepseek-r1:14b` 모델을 다운로드합니다:

```bash
# Ollama 설치 후
ollama pull deepseek-r1:14b

# Ollama 서버 실행 (기본 포트: 11434)
ollama serve
```

### 3. ComfyUI 설정

ComfyUI를 설치하고 실행합니다:

```bash
# ComfyUI 실행 (기본 포트: 8188)
# ComfyUI가 http://127.0.0.1:8188 에서 실행 중이어야 합니다
```

**참고**: ComfyUI 워크플로우는 `comfyui_workflow.json` 파일에 저장되어 있습니다. 필요에 따라 이 파일을 수정하여 원하는 워크플로우를 사용할 수 있습니다.

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 설정을 추가하세요:

```env
# Ollama 설정
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:14b
OLLAMA_TEMPERATURE=0.7

# ComfyUI 설정
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_WS_URL=ws://127.0.0.1:8188/ws
COMFYUI_WORKFLOW_PATH=comfyui_workflow.json
COMFYUI_TIMEOUT=300
COMFYUI_RETRY_COUNT=2
COMFYUI_CHECKPOINT=animagine-xl-4.0-opt.safetensors  # 사용할 체크포인트 모델명 (선택사항)

# 출력 설정
OUTPUT_DIR=output

# 비디오 설정
VIDEO_FPS=24
VIDEO_CODEC=libx264
VIDEO_AUDIO_CODEC=aac
ZOOM_FACTOR=1.15

# TTS 설정
TTS_VOICE=ko-KR-InJoonNeural
```

**중요**: `COMFYUI_CHECKPOINT`를 설정하면 해당 모델을 우선적으로 사용합니다. 설정하지 않으면 ComfyUI에서 사용 가능한 첫 번째 모델을 자동으로 선택합니다.

## 📁 프로젝트 구조

```
langgraph-automator/
├── src/
│   ├── main.py              # 메인 에이전트 코드
│   ├── config.py            # 설정 관리 모듈
│   ├── agents/              # 에이전트 모듈
│   │   ├── script_writer.py # 대본 작성 에이전트
│   │   ├── tts.py          # TTS 에이전트
│   │   └── vision.py       # 이미지 생성 에이전트
│   ├── tools/               # 도구 모듈
│   │   ├── comfyui_client.py # ComfyUI 클라이언트
│   │   └── video_editor.py  # 비디오 편집 도구
│   └── workflow/            # 워크플로우 정의
│       └── graph.py         # LangGraph 워크플로우
├── .env                     # 환경 변수 설정 (생성 필요)
├── comfyui_workflow.json   # ComfyUI 워크플로우 설정
├── output/                  # 생성된 파일 저장 디렉토리 (자동 생성)
│   ├── audio_scene_*.mp3    # 생성된 오디오 파일
│   ├── image_scene_*.png    # 생성된 이미지 파일
│   └── final_output.mp4     # 최종 비디오 파일
├── pyproject.toml           # 프로젝트 설정 및 의존성
└── README.md                # 이 파일
```

## 🎬 사용 방법

### 기본 실행

```bash
# src/main.py 실행
uv run python src/main.py
```

기본적으로 "AI의 미래"라는 주제로 비디오를 생성합니다.

### 주제 변경

`src/main.py`의 `main()` 함수에서 `initial_state`의 `topic` 값을 변경하세요:

```python
initial_state: VideoState = {
    "topic": "원하는 주제",  # 여기를 변경
    "scenes": [],
    "final_video_path": ""
}
```

## 🔧 워크플로우 설명

에이전트는 다음 순서로 작업을 수행합니다:

1. **대본 작성** (`node_script_writer`)
   - LLM을 사용하여 주제에 맞는 대본 생성
   - 각 장면별 대사와 이미지 프롬프트 생성
   - DeepSeek R1의 `<think>` 태그를 제거하고 JSON만 추출

2. **오디오 생성** (`node_audio_generator`)
   - `edge-tts`를 사용하여 각 장면의 대사를 한국어 음성으로 변환
   - 비동기 처리로 모든 오디오를 병렬 생성

3. **이미지 생성** (`node_visual_generator`)
   - ComfyUI API를 호출하여 각 장면의 이미지 생성
   - 웹소켓을 통해 생성 완료를 대기
   - 웹소켓 실패 시 히스토리 API로 폴백하여 이미지 정보 조회
   - 생성된 이미지를 다운로드하여 저장
   - 환경 변수 `COMFYUI_CHECKPOINT`로 지정된 모델 우선 사용

4. **비디오 편집** (`node_video_editor`)
   - `moviepy 2.2.1`를 사용하여 이미지와 오디오 결합
   - Ken Burns 효과 (Zoom-in) 적용 (VideoClip 기반)
   - 모든 장면을 이어 붙여 최종 비디오 생성

## ⚙️ 설정 커스터마이징

### 환경 변수 설정

`.env` 파일을 통해 모든 설정을 변경할 수 있습니다. 주요 설정 항목:

- **OLLAMA_URL**: Ollama 서버 URL (기본값: `http://localhost:11434`)
- **OLLAMA_MODEL**: 사용할 Ollama 모델 (기본값: `deepseek-r1:14b`)
- **COMFYUI_URL**: ComfyUI 서버 URL (기본값: `http://127.0.0.1:8188`)
- **COMFYUI_CHECKPOINT**: 사용할 체크포인트 모델명 (예: `animagine-xl-4.0-opt.safetensors`)
- **TTS_VOICE**: TTS 음성 (기본값: `ko-KR-InJoonNeural`)
- **ZOOM_FACTOR**: Ken Burns 효과 줌 팩터 (기본값: `1.15`)

### ComfyUI 워크플로우 수정

`comfyui_workflow.json` 파일을 수정하여 원하는 ComfyUI 워크플로우를 사용할 수 있습니다. 워크플로우에서 프롬프트 노드의 `text` 필드가 자동으로 업데이트됩니다.

### ComfyUI 체크포인트 모델 설정

`.env` 파일에서 `COMFYUI_CHECKPOINT`를 설정하면 해당 모델을 우선적으로 사용합니다:

```env
COMFYUI_CHECKPOINT=animagine-xl-4.0-opt.safetensors
```

모델 파일은 ComfyUI의 체크포인트 디렉토리(`models/checkpoints/`)에 있어야 합니다.

### 음성 변경

`node_audio_generator` 함수의 `generate_audio_async` 호출 부분에서 `voice` 매개변수를 변경하세요:

```python
# 한국어 남성: "ko-KR-InJoonNeural"
# 한국어 여성: "ko-KR-SunHiNeural"
await generate_audio_async(script, str(audio_path), voice="ko-KR-SunHiNeural")
```

사용 가능한 음성 목록은 `edge-tts --list-voices` 명령으로 확인할 수 있습니다.

### Ken Burns 효과 조정

`node_video_editor` 함수의 `apply_ken_burns_effect` 호출 부분에서 `zoom_factor`를 조정하세요:

```python
# zoom_factor: 1.0 = 줌 없음, 1.2 = 20% 확대, 1.5 = 50% 확대
image_clip = apply_ken_burns_effect(image_clip, audio_duration, zoom_factor=1.15)
```

## 🐛 문제 해결

### Ollama 연결 오류

- Ollama가 `http://localhost:11434`에서 실행 중인지 확인
- `deepseek-r1:14b` 모델이 다운로드되었는지 확인: `ollama list`

### ComfyUI 연결 오류

- ComfyUI가 `http://127.0.0.1:8188`에서 실행 중인지 확인
- ComfyUI 웹 인터페이스에서 API가 활성화되어 있는지 확인
- 워크플로우 JSON 파일이 올바른 형식인지 확인
- `.env` 파일의 `COMFYUI_CHECKPOINT`에 지정한 모델이 ComfyUI에 존재하는지 확인
- 모델이 없으면 환경 변수를 비우고 자동 선택 기능 사용
- 웹소켓 연결 실패 시 히스토리 API로 자동 폴백되므로 잠시 대기

### 오디오 생성 오류

- 인터넷 연결 확인 (edge-tts는 온라인 서비스를 사용)
- 사용 가능한 음성 목록 확인: `edge-tts --list-voices`

### 비디오 편집 오류

- FFmpeg가 설치되어 있는지 확인 (moviepy 필수)
- 이미지와 오디오 파일이 올바르게 생성되었는지 확인
- MoviePy 2.2.1 이상 버전 사용 확인 (API 변경사항 적용됨)
- `with_*` 메서드 사용 (구버전 `set_*` 메서드 대신)

## 🔧 기술 스택

- **Python**: 3.13+
- **LangGraph**: 1.0.4+ (워크플로우 관리)
- **LangChain**: 1.1.0+ (LLM 통합)
- **Ollama**: 로컬 LLM 실행
- **MoviePy**: 2.2.1+ (비디오 편집)
- **Edge-TTS**: 7.2.3+ (텍스트 음성 변환)
- **ComfyUI**: 이미지 생성 (Stable Diffusion)

## 📝 라이선스

이 프로젝트는 자유롭게 사용 및 수정할 수 있습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

