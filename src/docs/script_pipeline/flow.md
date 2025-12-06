# Script Pipeline 처리 흐름

## 전체 흐름

```
[START]
  ↓
[topic 확인]
  ↓
[script_writer 실행]
  ↓ (LLM 호출)
  ↓ (JSON 파싱)
  ↓ (scenes 생성)
  ↓
[END]
  ↓ (scenes 반환)
```

## 상세 처리 단계

### 1. 상태 검증
- `state`에서 `topic` 추출
- `topic`이 없거나 비어있으면 `ValueError` 발생

### 2. 캐시 확인
- `WorkflowCache.STAGE_SCRIPT` 단계 캐시 확인
- 캐시된 `scenes`가 있으면 스킵하고 반환

### 3. 대본 생성 (script_writer 노드)
- `create_script_prompt(topic)` 호출하여 프롬프트 생성
- DeepSeek R1용 프롬프트 생성 (`<think>` 블록 포함)
- Ollama API 호출 (타임아웃: 600초)
- `extract_json_from_text()`로 JSON 추출
- `scenes` 리스트 검증 및 필드 추가

### 4. 대본 저장
- `output/{topic}/script.json`에 저장
- 저장 실패 시 경고만 출력 (프로세스는 계속)

### 5. 상태 업데이트
- `state`에 `scenes` 필드 추가/업데이트
- 업데이트된 상태 반환

## 에러 처리

- **Ollama API 호출 실패**: 재시도 2회 (2초 지연)
- **JSON 파싱 실패**: `ValueError` 발생
- **데이터 검증 실패**: `ValueError` 발생
- 모든 에러는 로깅 후 상위로 전파

## 재시도 흐름

```
시도 1 실패
  ↓ (2초 대기)
시도 2 실패
  ↓ (4초 대기)
시도 3 실패
  ↓
에러 발생
```

## 성공 케이스

```
입력: topic = "AI의 미래"
출력: scenes = [
  {
    "script": "요즘 AI 얘기 너무 많이 나오죠?",
    "image_prompt": "masterpiece, best quality, ...",
    "audio_path": "",
    "image_path": ""
  },
  ...
]
```

