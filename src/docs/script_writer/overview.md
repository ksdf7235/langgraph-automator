# Script Writer 기능 개요

## 목적

Ollama LLM을 사용하여 유튜브 쇼츠용 대본을 자동 생성합니다. 주제를 입력받아 30~45초 길이의 쇼츠 대본을 생성하며, 각 장면별 대사와 이미지 프롬프트를 포함합니다.

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

- `List[Dict[str, str]]` - 장면 리스트
  - 각 장면은 다음 필드를 포함:
    - `script: str` - 한국어 구어체 대사
    - `image_prompt: str` - 영어 이미지 프롬프트 (고품질 태그 포함)
    - `audio_path: str` - 초기값 빈 문자열 (오디오 생성 후 채워짐)
    - `image_path: str` - 초기값 빈 문자열 (이미지 생성 후 채워짐)

## 의존성

- `src.config.Config` - Ollama URL, 모델명, 온도 설정
- `src.utils.json_parser.extract_json_from_text` - LLM 출력에서 JSON 추출
- `requests` - Ollama API 호출
- Ollama 서버 (로컬 실행 필요, `deepseek-r1:14b` 모델 필요)

## 품질 기준

- 대본은 30~45초 길이로 생성되어야 함
- 각 장면은 1~3문장, 한 문장은 25자 이내
- 대사는 자연스러운 한국어 구어체여야 함
- 이미지 프롬프트는 고품질 태그로 시작해야 함
- JSON 파싱이 성공적으로 완료되어야 함
- 생성된 대본은 `output/{topic}/script.json`에 저장됨
