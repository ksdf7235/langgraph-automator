# Script Writer API 명세

## Public 함수

### `call_ollama(prompt: str, model: str = None, temperature: float = None) -> str`

Ollama API를 직접 호출하여 LLM 응답을 받습니다.

**파라미터:**

- `prompt: str` - 프롬프트 텍스트
- `model: str` - 모델 이름 (기본값: `Config.OLLAMA_MODEL`)
- `temperature: float` - 온도 설정 (기본값: `Config.OLLAMA_TEMPERATURE`)

**반환값:**

- `str` - LLM 응답 텍스트

**에러:**

- `requests.RequestException` - API 호출 실패 시
- `ValueError` - 응답에 'response' 키가 없는 경우

**타임아웃:** 600초 (10분)

---

### `create_script_prompt(topic: str) -> str`

대본 작성 프롬프트를 생성합니다. DeepSeek R1의 `<think>` 블록을 활용하여 구조를 먼저 설계하게 합니다.

**파라미터:**

- `topic: str` - 비디오 주제

**반환값:**

- `str` - 완성된 프롬프트 텍스트

**프롬프트 구조:**

- `<think>` 블록: 구조 설계 단계
- 본문: 대본 작성 요구사항 및 JSON 형식 지시

---

### `generate_script(topic: str) -> List[Dict[str, str]]`

주제에 맞는 쇼츠 대본을 생성합니다.

**파라미터:**

- `topic: str` - 비디오 주제

**반환값:**

- `List[Dict[str, str]]` - 장면 리스트
  - 각 장면 딕셔너리:
    - `script: str` - 한국어 구어체 대사
    - `image_prompt: str` - 영어 이미지 프롬프트
    - `audio_path: str` - 빈 문자열 (초기값)
    - `image_path: str` - 빈 문자열 (초기값)

**에러:**

- `ValueError` - 대본 생성 또는 파싱 실패 시
  - 'scenes' 키가 없는 경우
  - scenes가 유효한 리스트가 아닌 경우
  - 장면이 유효한 딕셔너리가 아닌 경우

---

### `node_script_writer(state: Dict) -> Dict`

LangGraph 노드 함수: 대본 작성

**파라미터:**

- `state: Dict` - 현재 상태 딕셔너리
  - 필수 키: `topic: str`

**반환값:**

- `Dict` - 업데이트된 상태 딕셔너리
  - `scenes: List[Dict[str, str]]` - 생성된 장면 리스트

**에러:**

- `ValueError` - 상태에 'topic'이 없는 경우

**부가 효과:**

- 생성된 대본을 `output/{topic}/script.json`에 저장
