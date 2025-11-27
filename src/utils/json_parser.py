"""
JSON 파싱 유틸리티 모듈
LLM 출력에서 JSON을 안전하게 추출합니다.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> dict:
    """
    LLM 출력에서 JSON을 추출합니다.
    DeepSeek R1의 <think>, <think> 태그를 제거하고 JSON만 추출합니다.
    
    Args:
        text: LLM이 반환한 텍스트
        
    Returns:
        파싱된 JSON 딕셔너리
        
    Raises:
        ValueError: JSON을 찾을 수 없거나 파싱에 실패한 경우
    """
    if not text or not isinstance(text, str):
        raise ValueError("입력 텍스트가 유효하지 않습니다.")
    
    original_text = text
    
    # <think> 태그와 내용 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # <think> 태그와 내용 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # JSON 코드 블록 찾기 (```json ... ``` 또는 ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
    else:
        # JSON 객체 직접 찾기 ({ ... })
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            logger.error(f"JSON을 찾을 수 없습니다. 원본 텍스트: {original_text[:500]}")
            raise ValueError(f"JSON을 찾을 수 없습니다. 텍스트 길이: {len(original_text)}")
    
    try:
        parsed = json.loads(json_text)
        logger.debug(f"JSON 파싱 성공: {len(json.dumps(parsed))} 문자")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}\n시도한 JSON 텍스트: {json_text[:500]}")
        raise ValueError(f"JSON 파싱 실패: {e}\n원본 텍스트 일부: {original_text[:500]}")

