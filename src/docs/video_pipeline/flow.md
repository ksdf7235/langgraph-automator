# Video Assembly 처리 흐름

## 전체 흐름

```
[START]
  ↓
[scenes 확인]
  ↓
[video_editor 실행]
  ↓
[각 장면 처리]
  ↓ (오디오 로드)
  ↓ (이미지/모션 프레임 로드)
  ↓ (Ken Burns 효과 적용)
  ↓ (비디오 클립 생성)
  ↓
[모든 클립 결합]
  ↓
[최종 비디오 렌더링]
  ↓
[END]
  ↓ (final_video_path 반환)
```

## 상세 처리 단계

### 1. 상태 검증
- `state`에서 `scenes` 추출
- `scenes`가 없거나 비어있으면 `ValueError` 발생

### 2. 캐시 확인
- `WorkflowCache.STAGE_VIDEO` 단계 캐시 확인
- 캐시된 `final_video_path`가 있고 파일이 존재하면 스킵하고 반환

### 3. 비디오 편집 (video_editor 노드)

#### 3.1 각 장면 처리
각 장면에 대해 다음 작업 수행:

1. **오디오 로드**
   - `audio_path` 확인
   - 오디오 파일 존재 확인
   - 오디오 길이 확인

2. **비주얼 선택**
   - `motion_frames_path` 확인
   - 모션 프레임이 있고 비어있지 않으면 모션 프레임 사용
   - 없으면 `image_path` 사용

3. **비디오 클립 생성**
   - 모션 프레임 사용 시:
     - 프레임 시퀀스로 비디오 클립 생성
     - FPS: `Config.MOTION_FPS`
   - 정적 이미지 사용 시:
     - 이미지로 비디오 클립 생성
     - Ken Burns 효과 적용
     - FPS: `Config.VIDEO_FPS`

4. **오디오 동기화**
   - 오디오 길이에 맞춰 비주얼 길이 조정
   - 오디오와 비주얼 결합

#### 3.2 모든 클립 결합
- 모든 장면의 비디오 클립을 순차적으로 연결
- `concatenate_videoclips()` 사용

#### 3.3 최종 비디오 렌더링
- 출력 경로: `output/{topic}/final_output.mp4`
- 코덱 설정:
  - 비디오 코덱: `Config.VIDEO_CODEC`
  - 오디오 코덱: `Config.VIDEO_AUDIO_CODEC`
- 임시 오디오 파일 생성 및 정리

### 4. 상태 업데이트
- `state`에 `final_video_path` 필드 추가/업데이트
- 업데이트된 상태 반환

## 에러 처리

- **scenes가 없음**: `ValueError` 발생
- **오디오 파일 없음**: `FileNotFoundError` 발생
- **이미지/모션 프레임 없음**: `ValueError` 발생
- **비디오 생성 실패**: 재시도 1회 (2초 지연)
- 모든 에러는 로깅 후 상위로 전파

## 재시도 흐름

```
시도 1 실패
  ↓ (2초 대기)
시도 2 실패
  ↓
에러 발생
```

## 성공 케이스

```
입력: scenes = [
  {
    "audio_path": "output/topic/audio_scene_1.mp3",
    "image_path": "output/topic/image_scene_1.png",
    "motion_frames_path": ["output/topic/motion_scene_1/frame_0001.png", ...]
  },
  ...
]

출력: final_video_path = "output/topic/final_output.mp4"
```

## Ken Burns 효과

정적 이미지 사용 시 적용되는 효과:
- 초기: 이미지 전체를 보여줌
- 종료: 이미지 중앙을 확대 (zoom_factor 적용)
- 부드러운 전환 애니메이션

