"""
설정 관리 모듈
환경 변수 및 기본 설정을 로드합니다.
프로젝트 루트의 .env 파일을 자동으로 로드합니다.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 프로젝트 루트 경로 찾기 (src/config.py에서 상위 디렉토리)
_project_root = Path(__file__).parent.parent
_env_path = _project_root / ".env"

# .env 파일이 있으면 자동으로 로드
if _env_path.exists():
    load_dotenv(_env_path, override=False)  # override=False: 기존 환경 변수 우선
    print(f"[Config] .env 파일 로드됨: {_env_path}")
else:
    # .env 파일이 없어도 기본값으로 동작
    print(f"[Config] .env 파일을 찾을 수 없습니다. 기본값을 사용합니다: {_env_path}")


class Config:
    """애플리케이션 설정 클래스"""
    
    # Ollama 설정
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    
    # ComfyUI 설정
    COMFYUI_URL: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
    COMFYUI_WS_URL: str = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws")
    COMFYUI_WORKFLOW_PATH: str = os.getenv("COMFYUI_WORKFLOW_PATH", "comfyui_workflow.json")
    COMFYUI_TIMEOUT: int = int(os.getenv("COMFYUI_TIMEOUT", "300"))
    COMFYUI_RETRY_COUNT: int = int(os.getenv("COMFYUI_RETRY_COUNT", "2"))
    COMFYUI_USE_MOCK: bool = os.getenv("COMFYUI_USE_MOCK", "false").lower() == "true"  # 테스트용 더미 이미지 생성
    COMFYUI_API_KEY: str = os.getenv("COMFYUI_API_KEY", "")  # ComfyUI API 키 (선택사항)
    COMFYUI_CHECKPOINT: str = os.getenv("COMFYUI_CHECKPOINT", "")  # 사용할 체크포인트 모델명 (선택사항)
    
    # 출력 설정
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    
    # 비디오 설정
    VIDEO_FPS: int = int(os.getenv("VIDEO_FPS", "24"))
    VIDEO_CODEC: str = os.getenv("VIDEO_CODEC", "libx264")
    VIDEO_AUDIO_CODEC: str = os.getenv("VIDEO_AUDIO_CODEC", "aac")
    ZOOM_FACTOR: float = float(os.getenv("ZOOM_FACTOR", "1.15"))
    
    # TTS 설정
    TTS_VOICE: str = os.getenv("TTS_VOICE", "ko-KR-InJoonNeural")
    
    @classmethod
    def get_output_dir(cls) -> Path:
        """출력 디렉토리 경로를 반환하고 생성합니다."""
        output_path = Path(cls.OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    @classmethod
    def validate(cls) -> None:
        """설정값 검증"""
        if not cls.OLLAMA_URL.startswith(("http://", "https://")):
            raise ValueError(f"잘못된 OLLAMA_URL 형식: {cls.OLLAMA_URL}")
        if not cls.COMFYUI_URL.startswith(("http://", "https://")):
            raise ValueError(f"잘못된 COMFYUI_URL 형식: {cls.COMFYUI_URL}")

