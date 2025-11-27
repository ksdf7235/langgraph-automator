"""
TTS (Text-to-Speech) 에이전트
edge-tts를 사용하여 텍스트를 오디오로 변환합니다.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List

import edge_tts

from src.config import Config

logger = logging.getLogger(__name__)


async def generate_audio_async(script: str, output_path: str, voice: str = None) -> None:
    """
    edge-tts를 사용하여 텍스트를 오디오로 변환합니다 (비동기).
    
    Args:
        script: 변환할 텍스트
        output_path: 저장할 오디오 파일 경로
        voice: 사용할 음성 (기본값: Config.TTS_VOICE)
        
    Raises:
        Exception: 오디오 생성 실패 시
    """
    voice = voice or Config.TTS_VOICE
    
    if not script or not script.strip():
        raise ValueError("빈 스크립트는 오디오로 변환할 수 없습니다.")
    
    try:
        logger.debug(f"오디오 생성 시작: {output_path} (음성: {voice})")
        communicate = edge_tts.Communicate(script, voice)
        await communicate.save(output_path)
        
        # 파일 생성 확인
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"오디오 파일이 생성되지 않았습니다: {output_path}")
        
        logger.debug(f"오디오 생성 완료: {output_path}")
        
    except Exception as e:
        logger.error(f"오디오 생성 실패 ({output_path}): {e}")
        raise


def generate_all_audios(scenes: List[Dict], output_dir: Path) -> List[Dict]:
    """
    모든 장면의 오디오를 생성합니다.
    
    Args:
        scenes: 장면 리스트
        output_dir: 출력 디렉토리
        
    Returns:
        오디오 경로가 업데이트된 장면 리스트
        
    Raises:
        ValueError: 스크립트가 없는 장면이 있는 경우
    """
    logger.info(f"[오디오 생성] {len(scenes)}개 장면 처리 시작")
    
    async def generate_all_async():
        tasks = []
        for idx, scene in enumerate(scenes):
            script = scene.get('script', '').strip()
            if not script:
                raise ValueError(f"장면 {idx+1}에 대사가 없습니다.")
            
            audio_path = output_dir / f"audio_scene_{idx+1}.mp3"
            tasks.append(generate_audio_async(script, str(audio_path)))
        
        # 모든 오디오 생성 작업 병렬 실행
        await asyncio.gather(*tasks)
    
    # 비동기 함수 실행
    try:
        asyncio.run(generate_all_async())
    except Exception as e:
        logger.error(f"[오디오 생성] 비동기 실행 실패: {e}", exc_info=True)
        raise
    
    # 생성된 오디오 경로를 상태에 업데이트
    updated_scenes = []
    for idx, scene in enumerate(scenes):
        audio_path = output_dir / f"audio_scene_{idx+1}.mp3"
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"오디오 파일이 생성되지 않았습니다: {audio_path}")
        
        scene['audio_path'] = str(audio_path)
        updated_scenes.append(scene)
        logger.info(f"[오디오 생성] 장면 {idx+1} 완료: {audio_path}")
    
    logger.info(f"[오디오 생성] 모든 오디오 생성 완료")
    
    return updated_scenes


def node_audio_generator(state: Dict) -> Dict:
    """
    LangGraph 노드 함수: 오디오 생성
    
    Args:
        state: 현재 상태 딕셔너리
        
    Returns:
        업데이트된 상태 딕셔너리
    """
    try:
        scenes = state.get("scenes", [])
        if not scenes:
            raise ValueError("상태에 'scenes'가 없거나 비어있습니다.")
        
        output_dir = Config.get_output_dir()
        updated_scenes = generate_all_audios(scenes, output_dir)
        
        return {
            **state,
            "scenes": updated_scenes
        }
        
    except Exception as e:
        logger.error(f"[오디오 생성 노드] 오류: {e}", exc_info=True)
        raise

