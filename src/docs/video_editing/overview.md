# Video Editing 기능 개요

## 목적
moviepy를 사용하여 이미지/프레임 시퀀스와 오디오를 결합하여 최종 비디오를 생성합니다.

## 스코프
- 정적 이미지와 오디오 결합
- 모션 프레임 시퀀스와 오디오 결합
- Ken Burns 효과 적용
- 여러 장면을 순차적으로 연결

## 입력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 `audio_path`, `image_path` 또는 `motion_frames_path` 포함

## 출력
- 최종 비디오 파일: `output/{topic}/final_output.mp4`

