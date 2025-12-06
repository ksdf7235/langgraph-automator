# Video Assembly API 명세

## Public 함수

### `build_video_graph() -> CompiledGraph[VideoState, VideoState]`

Video Assembly 서브그래프를 생성하고 컴파일합니다.

**반환값:**
- 컴파일된 StateGraph 인스턴스

**그래프 구조:**
```
[START] → video_editor → [END]
```

---

## 내부 노드 함수

### `node_video_editor(state: VideoState) -> VideoState`

비디오 편집 노드

**파라미터:**
- `state: VideoState` - 현재 상태 딕셔너리
  - 필수: `scenes` (각 장면에 `audio_path` 필요)
  - 필수: `topic` (출력 경로 생성용)

**반환값:**
- `VideoState` - 업데이트된 상태 딕셔너리
  - `final_video_path: str` - 생성된 비디오 파일 경로

**에러:**
- `ValueError` - scenes가 없거나 비어있는 경우
- `ValueError` - 장면에 audio_path가 없는 경우
- `ValueError` - 장면에 image_path 또는 motion_frames_path가 없는 경우
- `FileNotFoundError` - 오디오/이미지 파일이 없는 경우
- `Exception` - 비디오 생성 실패 시

## VideoState 필드

### 입력 필드 (필수)
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면에 다음 필드 필요:
    - `audio_path: str` - 오디오 파일 경로 (필수)
    - `image_path: str` - 정적 이미지 파일 경로 (모션 프레임이 없을 때)
    - `motion_frames_path: List[str]` - 모션 프레임 파일 경로 리스트 (선택적)
- `topic: str` - 비디오 주제 (출력 경로 생성용)

### 출력 필드
- `final_video_path: str` - 최종 비디오 파일 경로
  - 형식: `output/{topic}/final_output.mp4`

## 캐싱

- 캐시 단계: `WorkflowCache.STAGE_VIDEO`
- 캐시 키: `final_video_path`
- 캐시된 비디오 파일이 존재하면 스킵

## 재시도 정책

- 최대 재시도: 1회
- 재시도 지연: 2초

## 비디오 사양

- 코덱: `Config.VIDEO_CODEC` (기본: "libx264")
- 오디오 코덱: `Config.VIDEO_AUDIO_CODEC` (기본: "aac")
- 프레임 레이트:
  - 모션 프레임 사용 시: `Config.MOTION_FPS` (기본: 8fps)
  - 정적 이미지 사용 시: `Config.VIDEO_FPS` (기본: 30fps)

