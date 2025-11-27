import os

# 설정: Python 프로젝트에 맞는 확장자와 무시할 폴더
TARGET_EXTENSIONS = {
    '.py',       # Python 소스 코드
    '.pyi',      # Python 타입 힌트
    '.pyx',      # Cython 소스 코드
    '.pyw',      # Python Windows 스크립트
    '.toml',     # pyproject.toml, poetry.lock 등
    '.yaml',     # YAML 설정 파일
    '.yml',      # YAML 설정 파일
    '.json',     # JSON 설정 파일
    '.md',       # 마크다운 문서
    '.txt',      # 텍스트 파일 (README 등)
    '.ini',      # INI 설정 파일
    '.cfg',      # 설정 파일
    '.env',      # 환경 변수 파일
    '.lock',     # uv.lock, poetry.lock 등
    '.yml',      # GitHub Actions 등
}

IGNORE_DIRS = {
    '.git',
    '__pycache__',  # Python 캐시 (중요)
    '.pytest_cache',  # pytest 캐시
    '.mypy_cache',   # mypy 캐시
    '.ruff_cache',   # ruff 캐시
    'venv',         # 가상 환경
    'env',          # 가상 환경
    '.venv',        # 가상 환경
    'node_modules', # Node.js 모듈
    'dist',         # 배포 파일
    'build',        # 빌드 파일
    '.tox',         # tox 테스트 환경
    'htmlcov',      # 커버리지 리포트
    '.coverage',    # 커버리지 데이터
    '.idea',        # PyCharm 설정
    '.vscode',      # VS Code 설정
    '.eggs',        # setuptools eggs
    '*.egg-info',   # 패키지 메타데이터
    '.hypothesis',  # hypothesis 테스트
    '.ipynb_checkpoints',  # Jupyter 체크포인트
    'logs',         # 로그 파일 디렉토리
    'output',       # 출력 디렉토리
}

def merge_project_files(output_filename="python_project_context.txt"):
    """
    Python 프로젝트의 모든 관련 파일을 하나의 텍스트 파일로 병합합니다.
    
    Args:
        output_filename: 출력 파일명
    """
    file_count = 0
    
    with open(output_filename, "w", encoding="utf-8") as outfile:
        # 프로젝트 루트부터 탐색
        for root, dirs, files in os.walk("."):
            # 무시할 폴더 제거 (inplace 수정)
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
            
            for file in files:
                # 숨김 파일 제외 (단, .env는 포함)
                if file.startswith('.') and file != '.env' and not file.endswith(('.py', '.toml', '.yaml', '.yml')):
                    continue
                
                ext = os.path.splitext(file)[1].lower()  # 확장자 대소문자 구분 없이 처리
                if ext in TARGET_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    
                    # 무시할 디렉토리 경로에 포함되어 있는지 확인
                    if any(ignore_dir in file_path.split(os.sep) for ignore_dir in IGNORE_DIRS):
                        continue
                    
                    try:
                        with open(file_path, "r", encoding="utf-8", errors='ignore') as infile:
                            content = infile.read()
                            
                            # ★ 핵심: 파일 경로를 명확히 적어줌 (LLM이 파일 위치 인식용)
                            outfile.write(f"\n\n{'='*80}\n")
                            outfile.write(f"File Path: {file_path}\n")
                            outfile.write(f"{'='*80}\n\n")
                            outfile.write(content)
                            file_count += 1
                            print(f"Added: {file_path}")
                            
                    except UnicodeDecodeError:
                        print(f"Skipped {file_path} (binary or encoding issue)")
                    except Exception as e:
                        print(f"Skipped {file_path} due to error: {e}")

    print(f"\nDone! {file_count} Python project files merged into: {output_filename}")

if __name__ == "__main__":
    merge_project_files()