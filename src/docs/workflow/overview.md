# Workflow 기능 개요

## 목적
LangGraph를 사용하여 비디오 생성 워크플로우를 정의하고 관리합니다. 3개의 서브그래프와 1개의 메인 오케스트레이터로 구성된 계층적 구조를 사용합니다.

## 아키텍처

### 메인 오케스트레이터 + 3개 서브그래프 구조

```
[메인 오케스트레이터 그래프]
  ↓
[Script Pipeline 서브그래프]
  ↓
[Asset Generation 서브그래프]
  ↓
[Video Assembly 서브그래프]
```

### 서브그래프

1. **Script Pipeline 서브그래프**
   - 역할: topic → scenes (대본 및 이미지 프롬프트)
   - 노드: script_writer

2. **Asset Generation 서브그래프**
   - 역할: scenes → scenes (오디오, 이미지, 모션 에셋 생성)
   - 노드: audio_generator, visual_generator, motion_generator

3. **Video Assembly 서브그래프**
   - 역할: scenes → final_video_path (최종 비디오 생성)
   - 노드: video_editor

## 스코프
- LangGraph StateGraph 구성
- 서브그래프 오케스트레이션
- 상태 관리 (VideoState)
- 재시도 로직 데코레이터
- 캐싱 로직 데코레이터

## 워크플로우 단계

### 메인 레벨
1. script_pipeline - Script Pipeline 서브그래프 호출
2. asset_pipeline - Asset Generation 서브그래프 호출
3. video_pipeline - Video Assembly 서브그래프 호출

### 서브그래프 레벨

#### Script Pipeline
- script_writer - 대본 작성

#### Asset Generation
- audio_generator - 오디오 생성
- visual_generator - 이미지 생성
- motion_generator - 모션 생성

#### Video Assembly
- video_editor - 비디오 편집

## 상태 관리

VideoState는 모든 서브그래프에서 공유되며, 각 서브그래프는 필요한 필드만 업데이트합니다:

```python
class VideoState(TypedDict):
    topic: str                      # 입력: 비디오 주제
    scenes: List[Dict[str, str]]    # 중간 및 최종 결과
    final_video_path: str           # 출력: 최종 비디오 파일 경로
```

## 장점

1. **모듈화**: 각 서브그래프가 독립적인 책임을 가짐
2. **확장성**: 서브그래프 내부 노드를 쉽게 추가/수정 가능
3. **테스트 용이성**: 각 서브그래프를 독립적으로 테스트 가능
4. **재사용성**: 서브그래프를 다른 워크플로우에서 재사용 가능

## 참고 문서

- [Script Pipeline 문서](../script_pipeline/overview.md)
- [Asset Pipeline 문서](../asset_pipeline/overview.md)
- [Video Pipeline 문서](../video_pipeline/overview.md)
