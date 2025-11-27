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

## 📁 프로젝트 구조

```
langgraph-automator/
├── src/
│   └── main.py              # 메인 에이전트 코드
├── comfyui_workflow.json    # ComfyUI 워크플로우 설정
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
   - 생성된 이미지를 다운로드하여 저장

4. **비디오 편집** (`node_video_editor`)
   - `moviepy`를 사용하여 이미지와 오디오 결합
   - Ken Burns 효과 (Zoom-in) 적용
   - 모든 장면을 이어 붙여 최종 비디오 생성

## ⚙️ 설정 커스터마이징

### ComfyUI 워크플로우 수정

`comfyui_workflow.json` 파일을 수정하여 원하는 ComfyUI 워크플로우를 사용할 수 있습니다. 워크플로우에서 프롬프트 노드의 `text` 필드가 자동으로 업데이트됩니다.

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

### 오디오 생성 오류

- 인터넷 연결 확인 (edge-tts는 온라인 서비스를 사용)
- 사용 가능한 음성 목록 확인: `edge-tts --list-voices`

### 비디오 편집 오류

- FFmpeg가 설치되어 있는지 확인 (moviepy 필수)
- 이미지와 오디오 파일이 올바르게 생성되었는지 확인

## 📝 라이선스

이 프로젝트는 자유롭게 사용 및 수정할 수 있습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

