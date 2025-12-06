# Asset Generation 처리 흐름

## 전체 흐름

```
[START]
  ↓
[scenes 확인]
  ↓
[audio_generator 실행]
  ↓ (TTS 병렬 생성)
  ↓ (audio_path 업데이트)
  ↓
[visual_generator 실행]
  ↓ (ComfyUI 이미지 생성)
  ↓ (image_path 업데이트)
  ↓
[motion_generator 실행] (선택적)
  ↓ (Wan2.2 I2V 모션 생성)
  ↓ (motion_frames_path 업데이트)
  ↓
[END]
  ↓ (scenes 반환)
```

## 상세 처리 단계

### 1. 상태 검증
- `state`에서 `scenes` 추출
- `scenes`가 없거나 비어있으면 `ValueError` 발생

### 2. Audio Generation (audio_generator)

#### 2.1 캐시 확인
- `WorkflowCache.STAGE_AUDIO` 단계 캐시 확인
- 캐시된 `scenes`가 있고 모든 오디오 파일이 존재하면 스킵

#### 2.2 오디오 생성
- 각 장면의 `script` 필드 확인
- `generate_all_audios()` 호출
- 비동기로 모든 오디오 병렬 생성
- 각 장면에 `audio_path` 필드 업데이트
- 파일 생성 확인

### 3. Visual Generation (visual_generator)

#### 3.1 캐시 확인
- `WorkflowCache.STAGE_IMAGE` 단계 캐시 확인
- 캐시된 `scenes`가 있고 모든 이미지 파일이 존재하면 스킵

#### 3.2 이미지 생성
- 각 장면의 `image_prompt` 필드 확인
- ComfyUI 클라이언트 초기화 및 헬스체크
- 워크플로우 로드
- 각 장면별로 순차 처리:
  - 프롬프트 업데이트 및 시드 랜덤화
  - 이미지 생성 실행
  - 이미지 다운로드
  - 각 장면에 `image_path` 필드 업데이트

### 4. Motion Generation (motion_generator, 선택적)

#### 4.1 활성화 확인
- `Config.MOTION_ENABLED` 확인
- False이면 스킵

#### 4.2 캐시 확인
- `WorkflowCache.STAGE_MOTION` 단계 캐시 확인
- 캐시된 `scenes`가 있고 모든 모션 프레임 파일이 존재하면 스킵

#### 4.3 모션 프레임 생성
- 각 장면의 `image_path` 필드 확인
- 이미지 파일 존재 확인
- MotionGenerator 초기화
- 각 장면별로 순차 처리:
  - 이미지 업로드
  - 워크플로우 생성
  - 모션 생성 실행
  - 프레임 다운로드
  - 각 장면에 `motion_frames_path` 필드 업데이트

### 5. 상태 업데이트
- 모든 에셋 생성 완료 후 `scenes` 업데이트
- 업데이트된 상태 반환

## 에러 처리

### Audio Generation
- **빈 스크립트**: `ValueError` 발생
- **오디오 생성 실패**: 재시도 2회 (1초 지연)
- **파일 생성 실패**: `FileNotFoundError` 발생

### Visual Generation
- **ComfyUI 서버 연결 실패**: `RuntimeError` 발생
- **이미지 생성 실패**: 재시도 2회 (3초 지연)
- **프롬프트 없음**: `ValueError` 발생

### Motion Generation
- **이미지 파일 없음**: `FileNotFoundError` 발생
- **모션 생성 실패**: 재시도 2회 (3초 지연)
- **서버 연결 실패**: `RuntimeError` 발생

## 재시도 흐름

각 노드는 독립적으로 재시도됩니다:

```
audio_generator:
  시도 1 실패 → 1초 대기 → 시도 2 실패 → 2초 대기 → 시도 3 실패 → 에러

visual_generator:
  시도 1 실패 → 3초 대기 → 시도 2 실패 → 6초 대기 → 시도 3 실패 → 에러

motion_generator:
  시도 1 실패 → 3초 대기 → 시도 2 실패 → 6초 대기 → 시도 3 실패 → 에러
```

## 성공 케이스

```
입력: scenes = [
  {
    "script": "요즘 AI 얘기 너무 많이 나오죠?",
    "image_prompt": "masterpiece, best quality, ...",
    "audio_path": "",
    "image_path": ""
  },
  ...
]

출력: scenes = [
  {
    "script": "요즘 AI 얘기 너무 많이 나오죠?",
    "image_prompt": "masterpiece, best quality, ...",
    "audio_path": "output/topic/audio_scene_1.mp3",
    "image_path": "output/topic/image_scene_1.png",
    "motion_frames_path": ["output/topic/motion_scene_1/frame_0001.png", ...]
  },
  ...
]
```

