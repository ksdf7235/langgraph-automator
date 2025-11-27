#!/usr/bin/env python3
"""
폴더 내부의 모든 JSON 파일에서 키 이름이 "base64"로 끝나는 모든 키의 값을 빈 문자열("")로 변경하는 스크립트
"""

import json
from pathlib import Path
from typing import Any

# 처리할 폴더 경로 (변수로 지정)
TARGET_FOLDER = "."  # 현재 폴더를 기본값으로 설정


def clear_base64_values(data: Any) -> Any:
    """
    재귀적으로 JSON 데이터를 탐색하여 키 이름이 "base64"로 끝나는 모든 키의 값을 빈 문자열로 변경
    
    Args:
        data: JSON 데이터 (dict, list, 또는 기본 타입)
    
    Returns:
        수정된 데이터
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key.endswith("base64"):
                # 키 이름이 "base64"로 끝나면 값을 빈 문자열로 변경
                result[key] = ""
            else:
                # 재귀적으로 하위 구조 처리
                result[key] = clear_base64_values(value)
        return result
    elif isinstance(data, list):
        # 리스트의 각 요소를 재귀적으로 처리
        return [clear_base64_values(item) for item in data]
    else:
        # 기본 타입 (str, int, float, bool, None)은 그대로 반환
        return data


def process_json_file(json_file: Path) -> bool:
    """
    JSON 파일을 읽어서 base64로 끝나는 키의 값을 빈 문자열로 변경하고 저장
    
    Args:
        json_file: 처리할 JSON 파일 경로
    
    Returns:
        처리 성공 여부
    """
    try:
        # JSON 파일 읽기
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"오류: {json_file} - JSON 파싱 실패: {e}")
        return False
    except Exception as e:
        print(f"오류: {json_file} - 파일 읽기 실패: {e}")
        return False
    
    # base64로 끝나는 키의 값들을 빈 문자열로 변경
    modified_data = clear_base64_values(data)
    
    # 수정된 데이터를 원본 파일에 저장
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(modified_data, f, indent=2, ensure_ascii=False)
        print(f"완료: {json_file} 파일이 업데이트되었습니다.")
        return True
    except Exception as e:
        print(f"오류: {json_file} - 파일 쓰기 실패: {e}")
        return False


def main():
    """메인 함수 - 폴더 내부의 모든 JSON 파일 처리"""
    # 폴더 경로 확인
    folder_path = Path(TARGET_FOLDER)
    
    if not folder_path.exists():
        print(f"오류: 폴더를 찾을 수 없습니다: {TARGET_FOLDER}")
        return
    
    if not folder_path.is_dir():
        print(f"오류: 지정된 경로가 폴더가 아닙니다: {TARGET_FOLDER}")
        return
    
    # 폴더 내부의 모든 JSON 파일 찾기
    json_files = list(folder_path.glob("*.json"))
    
    if not json_files:
        print(f"경고: {TARGET_FOLDER} 폴더에 JSON 파일이 없습니다.")
        return
    
    print(f"{TARGET_FOLDER} 폴더에서 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    print("-" * 50)
    
    # 각 JSON 파일 처리
    success_count = 0
    for json_file in json_files:
        if process_json_file(json_file):
            success_count += 1
    
    print("-" * 50)
    print(f"처리 완료: {success_count}/{len(json_files)}개 파일 성공")


if __name__ == "__main__":
    main()

