# Video Assembly 서브그래프 개요

## 목적
Video Assembly 서브그래프는 생성된 에셋(오디오, 이미지, 모션 프레임)을 결합하여 최종 비디오를 생성합니다. 이 서브그래프는 moviepy를 사용하여 모든 장면을 편집하고 최종 MP4 파일을 출력합니다.

## 스코프
- 모든 장면의 오디오와 이미지/모션 프레임 결합
- Ken Burns 효과 적용 (정적 이미지 사용 시)
- 여러 장면을 순차적으로 연결
- 최종 비디오 파일 렌더링

## 입력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 다음 필드를 포함해야 함:
    - `audio_path: str` - 오디오 파일 경로 (필수)
    - `image_path: str` - 정적 이미지 파일 경로 (모션 프레임이 없을 때 사용)
    - `motion_frames_path: List[str]` - 모션 프레임 파일 경로 리스트 (선택적, 있으면 우선 사용)

## 출력
- `final_video_path: str` - 최종 비디오 파일 경로

## 책임 경계
- **이 서브그래프가 담당하는 것:**
  - 모든 장면을 타임라인으로 결합
  - 오디오와 비주얼 동기화
  - 최종 비디오 파일 렌더링
  
- **이 서브그래프가 담당하지 않는 것:**
  - 대본 생성 (Script Pipeline 서브그래프)
  - 에셋 생성 (Asset Generation 서브그래프)

## 내부 노드

### video_editor
- 역할: 모든 장면을 편집하여 최종 비디오 생성
- 입력: `scenes` (audio_path, image_path 또는 motion_frames_path 필요)
- 출력: `final_video_path`
- 캐시: `WorkflowCache.STAGE_VIDEO`
- 재시도: 1회, 2초 지연

## 처리 로직

### 1. 모션 프레임 우선 사용
- 각 장면에 `motion_frames_path`가 있고 비어있지 않으면 모션 프레임 사용
- 모션 프레임이 없거나 비활성화되어 있으면 정적 이미지 사용

### 2. Ken Burns 효과
- 정적 이미지 사용 시 Ken Burns 효과 적용 (확대/축소 애니메이션)

### 3. 오디오 동기화
- 각 장면의 오디오 길이에 맞춰 비주얼 길이 조정
- 오디오가 없으면 에러 발생

### 4. 프레임 레이트
- 모션 프레임 사용 시: `Config.MOTION_FPS`
- 정적 이미지 사용 시: `Config.VIDEO_FPS`

## 향후 확장 포인트
- 썸네일 자동 생성
- 자막 렌더링
- 배경 음악 추가
- 전환 효과 추가

