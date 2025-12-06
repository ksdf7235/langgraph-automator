# Script Pipeline 서브그래프 개요

## 목적
Script Pipeline은 비디오 주제(topic)를 받아 대본과 씬 메타데이터를 생성하는 서브그래프입니다. 이 서브그래프는 LLM을 사용하여 유튜브 쇼츠용 대본을 생성하고, 각 장면별 대사와 이미지 프롬프트를 준비합니다.

## 스코프
- 주제(topic) 기반 대본 생성
- 3~4개 장면으로 구성된 쇼츠 대본 구조
- 각 장면별 한국어 구어체 대사 생성
- 각 장면별 이미지 프롬프트 생성
- DeepSeek R1 모델의 `<think>` 태그 처리
- JSON 형식 출력 및 파싱

## 입력
- `topic: str` - 비디오 주제 (예: "AI의 미래")

## 출력
- `scenes: List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 다음 필드를 포함:
    - `script: str` - 한국어 구어체 대사
    - `image_prompt: str` - 영어 이미지 프롬프트 (고품질 태그 포함)
    - `audio_path: str` - 빈 문자열 (초기값)
    - `image_path: str` - 빈 문자열 (초기값)

## 책임 경계
- **이 서브그래프가 담당하는 것:**
  - 대본 텍스트 생성
  - 씬 구조 설계
  - 이미지 프롬프트 생성
  
- **이 서브그래프가 담당하지 않는 것:**
  - 오디오 생성 (Asset Generation 서브그래프)
  - 이미지 생성 (Asset Generation 서브그래프)
  - 비디오 편집 (Video Assembly 서브그래프)

## 내부 노드

### script_writer
- 역할: Ollama LLM을 사용하여 대본 생성
- 입력: `topic: str`
- 출력: `scenes: List[Dict[str, str]]`
- 캐시: `WorkflowCache.STAGE_SCRIPT`
- 재시도: 2회, 2초 지연

## 향후 확장 포인트
- `script_planner`: 구조/길이/톤 결정 (optional)
- `script_refiner`: 길이/톤 조정, edge cases 처리

