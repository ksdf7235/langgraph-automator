# Motion Generation 기능 개요

## 목적
Wan2.2 I2V 모델을 사용하여 정적 이미지를 애니메이션 프레임 시퀀스로 변환합니다.

## 스코프
- 정적 이미지를 애니메이션 프레임 시퀀스로 변환
- 각 장면별로 프레임 시퀀스 생성
- 생성된 프레임을 로컬에 저장

## 입력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 `image_path: str` 필드를 포함해야 함

## 출력
- `List[Dict[str, str]]` - 모션 프레임 경로가 추가된 장면 리스트
  - 각 장면에 `motion_frames_path: List[str]` 필드 추가

