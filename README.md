# YouTube Shorts 자동 생성 에이전트

LangGraph를 사용한 상태 기반 워크플로우로 유튜브 쇼츠를 자동 생성하는 에이전트입니다.

## 🚀 기능

- **대본 작성**: Ollama의 `deepseek-r1:14b` 모델을 사용하여 주제에 맞는 60초 이내 쇼츠 대본 자동 생성
- **오디오 생성**: `edge-tts`를 사용하여 한국어 음성으로 대본을 오디오로 변환
- **이미지 생성**: ComfyUI API를 통해 각 장면에 맞는 이미지 자동 생성
- **모션 생성**: Wan2.2 I2V 모델을 사용하여 정적 이미지를 애니메이션 프레임 시퀀스로 변환 (선택사항)
- **비디오 편집**: `moviepy`를 사용하여 이미지/프레임과 오디오를 결합하고 Ken Burns 효과 적용

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

**중요**: ComfyUI 서버는 **반드시 수동으로 실행**해야 합니다.
이 프로젝트는 ComfyUI 서버를 직접 실행하지 않고, HTTP API만 사용합니다.

**tqdm 비활성화 (필수)**: ComfyUI 서버 실행 시 tqdm을 비활성화하지 않으면 `OSError [Errno 22] Invalid argument` 에러가 발생할 수 있습니다.

**Windows PowerShell에서 실행:**
```powershell
# 환경변수 설정
$env:TQDM_DISABLE="1"
$env:TQDM_MININTERVAL="999999"
$env:TQDM_NCOLS="0"

# ComfyUI 서버 실행
python main.py --port 8188
```

**Windows CMD에서 실행:**
```cmd
set TQDM_DISABLE=1
set TQDM_MININTERVAL=999999
set TQDM_NCOLS=0
python main.py --port 8188
```

**Linux/Mac에서 실행:**
```bash
TQDM_DISABLE=1 TQDM_MININTERVAL=999999 TQDM_NCOLS=0 python main.py --port 8188
```

**또는 ComfyUI 서버 시작 스크립트(.bat 또는 .sh)에 환경변수를 추가하세요:**
```batch
@echo off
set TQDM_DISABLE=1
set TQDM_MININTERVAL=999999
set TQDM_NCOLS=0
python main.py --port 8188
```

**중요**: 환경변수 없이 ComfyUI 서버를 실행하면 모든 sampler에서 `OSError [Errno 22]` 에러가 발생할 수 있습니다.

```bash
# ComfyUI를 별도로 실행 (기본 포트: 8188)
# ComfyUI가 http://127.0.0.1:8188 에서 실행 중이어야 합니다
# 이 프로젝트는 서버 프로세스를 관리하지 않습니다
```

**모션 생성 사용 시 추가 설정**:

- Wan2.2 I2V 모델 파일 준비:
  - `Wan2.2-Distill-Loras.safetensors` → `ComfyUI/models/checkpoints/`
  - `Wan2.2-I2V-Distill-LORA.safetensors` → `ComfyUI/models/loras/`
- I2V 노드 설치: ComfyUI에 Image-to-Video 노드가 설치되어 있어야 합니다
  - 일반적으로 `ImageToVideo` 노드 사용
  - ComfyUI 웹 인터페이스에서 I2V 노드를 추가하여 정확한 노드 이름 확인

**참고**: ComfyUI 워크플로우는 `comfyui_workflow.json` 파일에 저장되어 있습니다. 필요에 따라 이 파일을 수정하여 원하는 워크플로우를 사용할 수 있습니다.

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 설정을 추가하세요:

```bash
# .env.example 파일을 복사하여 .env 파일 생성
cp .env.example .env
```

또는 직접 `.env` 파일을 생성하고 다음 설정을 추가하세요:

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
COMFYUI_CHECKPOINT=JANKUTrainedNoobaiRouwei_v60.safetensors  # 사용할 체크포인트 모델명 (선택사항)

# 출력 설정
OUTPUT_DIR=output

# 비디오 설정
VIDEO_FPS=24
VIDEO_CODEC=libx264
VIDEO_AUDIO_CODEC=aac
ZOOM_FACTOR=1.15

# TTS 설정
TTS_VOICE=ko-KR-InJoonNeural

# Wan2.2 I2V 모션 생성 설정
MOTION_ENABLED=true                    # 모션 생성 활성화 여부
MOTION_FPS=12                          # 모션 프레임레이트
MOTION_DURATION=3.0                    # 장면당 모션 길이 (초)
MOTION_MODEL_TYPE=wan2.2_distill       # 모션 모델 타입
I2V_CHECKPOINT=Wan2.2-Distill-Loras.safetensors  # I2V 체크포인트 모델
I2V_LORA=Wan2.2-I2V-Distill-LORA.safetensors     # I2V LoRA 모델
I2V_NODE_TYPE=ImageToVideo             # I2V 노드 타입 (ComfyUI에 설치된 실제 노드 이름)
I2V_STEPS=4                            # I2V 추론 스텝 (Lightning 4-step)
I2V_GUIDANCE=3.5                       # I2V Guidance 스케일
```

**중요**:

- `COMFYUI_CHECKPOINT`를 설정하면 해당 모델을 우선적으로 사용합니다. 설정하지 않으면 ComfyUI에서 사용 가능한 첫 번째 모델을 자동으로 선택합니다.
- `I2V_NODE_TYPE`은 ComfyUI에 설치된 실제 I2V 노드 이름으로 설정해야 합니다. 일반적인 값: `ImageToVideo`, `I2V`, `Wan2I2V` 등

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
│   │   ├── motion_gen.py    # 모션 생성 (Wan2.2 I2V)
│   │   └── video_editor.py  # 비디오 편집 도구
│   └── workflow/            # 워크플로우 정의
│       └── graph.py         # LangGraph 워크플로우
├── .env                     # 환경 변수 설정 (생성 필요)
├── .env.example            # 환경 변수 예제 파일
├── comfyui_workflow.json   # ComfyUI 워크플로우 설정
├── output/                  # 생성된 파일 저장 디렉토리 (자동 생성)
│   ├── {topic}/            # 주제별 폴더
│   │   ├── audio_scene_*.mp3    # 생성된 오디오 파일
│   │   ├── image_scene_*.png    # 생성된 이미지 파일
│   │   ├── motion/              # 모션 프레임 (선택사항)
│   │   │   └── scene_*/frame_*.png
│   │   ├── script.json          # 생성된 대본
│   │   └── final_output.mp4     # 최종 비디오 파일
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

4. **모션 생성** (`node_motion_generator`) - 선택사항

   - Wan2.2 I2V 모델을 사용하여 정적 이미지를 애니메이션 프레임으로 변환
   - 각 장면당 3초 분량의 프레임 시퀀스 생성 (기본값: 12fps × 3초 = 36프레임)
   - Lightning 4-step 추론 방식 사용
   - `MOTION_ENABLED=false`로 설정하면 스킵

5. **비디오 편집** (`node_video_editor`)
   - `moviepy 2.2.1`를 사용하여 이미지/프레임과 오디오 결합
   - 모션 프레임이 있으면 애니메이션 사용, 없으면 정적 이미지 + Ken Burns 효과
   - 모든 장면을 이어 붙여 최종 비디오 생성

## ⚙️ 설정 커스터마이징

### 환경 변수 설정

`.env` 파일을 통해 모든 설정을 변경할 수 있습니다. 주요 설정 항목:

- **OLLAMA_URL**: Ollama 서버 URL (기본값: `http://localhost:11434`)
- **OLLAMA_MODEL**: 사용할 Ollama 모델 (기본값: `deepseek-r1:14b`)
- **COMFYUI_URL**: ComfyUI 서버 URL (기본값: `http://127.0.0.1:8188`)
- **COMFYUI_CHECKPOINT**: 사용할 체크포인트 모델명 (예: `JANKUTrainedNoobaiRouwei_v60.safetensors`)
- **TTS_VOICE**: TTS 음성 (기본값: `ko-KR-InJoonNeural`)
- **ZOOM_FACTOR**: Ken Burns 효과 줌 팩터 (기본값: `1.15`)
- **MOTION_ENABLED**: 모션 생성 활성화 여부 (기본값: `true`)
- **MOTION_FPS**: 모션 프레임레이트 (기본값: `12`)
- **MOTION_DURATION**: 장면당 모션 길이 초 (기본값: `3.0`)
- **I2V_NODE_TYPE**: I2V 노드 타입 (기본값: `ImageToVideo`)
- **I2V_STEPS**: I2V 추론 스텝 (기본값: `4`)
- **I2V_GUIDANCE**: I2V Guidance 스케일 (기본값: `3.5`)

### ComfyUI 워크플로우 수정

`comfyui_workflow.json` 파일을 수정하여 원하는 ComfyUI 워크플로우를 사용할 수 있습니다. 워크플로우에서 프롬프트 노드의 `text` 필드가 자동으로 업데이트됩니다.

### ComfyUI 체크포인트 모델 설정

`.env` 파일에서 `COMFYUI_CHECKPOINT`를 설정하면 해당 모델을 우선적으로 사용합니다:

```env
COMFYUI_CHECKPOINT=JANKUTrainedNoobaiRouwei_v60.safetensors
```

모델 파일은 ComfyUI의 체크포인트 디렉토리(`models/checkpoints/`)에 있어야 합니다.

### 모션 생성 설정

모션 생성을 비활성화하려면:

```env
MOTION_ENABLED=false
```

I2V 노드 이름이 다른 경우 (예: `Wan2I2V`, `I2V` 등):

```env
I2V_NODE_TYPE=ImageToVideo  # 실제 ComfyUI 노드 이름으로 변경
```

ComfyUI 웹 인터페이스에서 I2V 노드를 추가하여 정확한 노드 이름을 확인할 수 있습니다.

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

### 모션 생성 오류

- **"node Wan2I2V does not exist" 오류**:
  - `.env` 파일의 `I2V_NODE_TYPE`을 실제 ComfyUI에 설치된 노드 이름으로 변경
  - 일반적인 값: `ImageToVideo`, `I2V`, `Wan2I2V` 등
  - ComfyUI 웹 인터페이스에서 I2V 노드를 추가하여 정확한 이름 확인
- **I2V 모델 파일 없음**:
  - `I2V_CHECKPOINT`와 `I2V_LORA` 파일이 올바른 경로에 있는지 확인
  - 체크포인트: `ComfyUI/models/checkpoints/`
  - LoRA: `ComfyUI/models/loras/`
- **프레임 개수 불일치**:
  - `MOTION_FPS`와 `MOTION_DURATION` 설정 확인
  - 예상 프레임 수 = `MOTION_FPS × MOTION_DURATION` (예: 12 × 3.0 = 36)

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
- **Wan2.2 I2V**: 이미지-투-비디오 변환 (Lightning 4-step)

## 📝 라이선스

이 프로젝트는 자유롭게 사용 및 수정할 수 있습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
