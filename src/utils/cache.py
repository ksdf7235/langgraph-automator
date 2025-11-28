"""
캐싱 유틸리티 모듈
워크플로우 단계별 결과물을 캐싱하여 재실행 시 시간을 절약합니다.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import Config

logger = logging.getLogger(__name__)


class WorkflowCache:
    """워크플로우 캐시 관리 클래스"""
    
    CACHE_FILE = "workflow_cache.json"
    
    # 캐시 단계 정의
    STAGE_SCRIPT = "script"
    STAGE_AUDIO = "audio"
    STAGE_IMAGE = "image"
    STAGE_MOTION = "motion"
    STAGE_VIDEO = "video"
    
    def __init__(self, topic: str):
        """
        캐시 초기화
        
        Args:
            topic: 비디오 주제 (캐시 키로 사용)
        """
        self.topic = topic
        self.topic_hash = self._hash_topic(topic)
        # 캐시는 주제별 폴더 안에 저장
        self.cache_dir = Config.get_output_dir(topic=topic) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{self.topic_hash}.json"
        self.cache_data = self._load_cache()
        
    def _hash_topic(self, topic: str) -> str:
        """주제를 해시하여 캐시 키 생성"""
        return hashlib.md5(topic.encode('utf-8')).hexdigest()[:12]
    
    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 주제가 일치하는지 확인
                    if data.get("topic") == self.topic:
                        logger.info(f"[캐시] 기존 캐시 로드: {self.cache_file.name}")
                        return data
                    else:
                        logger.info("[캐시] 주제가 변경되어 캐시 무효화")
            except Exception as e:
                logger.warning(f"[캐시] 캐시 로드 실패: {e}")
        
        return {"topic": self.topic, "stages": {}}
    
    def _save_cache(self) -> None:
        """캐시 파일 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[캐시] 저장 완료: {self.cache_file.name}")
        except Exception as e:
            logger.warning(f"[캐시] 저장 실패: {e}")
    
    def get_stage(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        캐시된 단계 데이터 조회
        
        Args:
            stage: 단계 이름 (script, audio, image, video)
            
        Returns:
            캐시된 데이터 또는 None
        """
        stage_data = self.cache_data.get("stages", {}).get(stage)
        
        if stage_data:
            # 파일 존재 여부 확인 (audio, image, motion, video 단계)
            if stage in [self.STAGE_AUDIO, self.STAGE_IMAGE, self.STAGE_MOTION]:
                scenes = stage_data.get("scenes", [])
                for scene in scenes:
                    if stage == self.STAGE_AUDIO:
                        path_key = "audio_path"
                    elif stage == self.STAGE_IMAGE:
                        path_key = "image_path"
                    else:  # STAGE_MOTION
                        # 모션 프레임 경로 확인
                        motion_frames = scene.get("motion_frames_path", [])
                        if motion_frames:
                            for frame_path in motion_frames:
                                if frame_path and not os.path.exists(frame_path):
                                    logger.info(f"[캐시] 모션 프레임이 없어 캐시 무효화: {frame_path}")
                                    return None
                        continue
                    
                    path = scene.get(path_key)
                    if path and not os.path.exists(path):
                        logger.info(f"[캐시] 파일이 없어 캐시 무효화: {path}")
                        return None
            elif stage == self.STAGE_VIDEO:
                video_path = stage_data.get("final_video_path")
                if video_path and not os.path.exists(video_path):
                    logger.info(f"[캐시] 비디오 파일이 없어 캐시 무효화: {video_path}")
                    return None
            
            logger.info(f"[캐시] {stage} 단계 캐시 히트!")
            return stage_data
        
        return None
    
    def set_stage(self, stage: str, data: Dict[str, Any]) -> None:
        """
        단계 데이터 캐시 저장
        
        Args:
            stage: 단계 이름
            data: 저장할 데이터
        """
        if "stages" not in self.cache_data:
            self.cache_data["stages"] = {}
        
        self.cache_data["stages"][stage] = data
        self._save_cache()
        logger.info(f"[캐시] {stage} 단계 캐시 저장")
    
    def clear(self) -> None:
        """캐시 초기화"""
        self.cache_data = {"topic": self.topic, "stages": {}}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("[캐시] 캐시 초기화 완료")
    
    def get_status(self) -> Dict[str, bool]:
        """각 단계별 캐시 상태 반환"""
        stages = [self.STAGE_SCRIPT, self.STAGE_AUDIO, self.STAGE_IMAGE, self.STAGE_MOTION, self.STAGE_VIDEO]
        return {stage: self.get_stage(stage) is not None for stage in stages}


# 전역 캐시 인스턴스 (main에서 초기화)
_cache_instance: Optional[WorkflowCache] = None


def init_cache(topic: str) -> WorkflowCache:
    """캐시 초기화 및 전역 인스턴스 설정"""
    global _cache_instance
    _cache_instance = WorkflowCache(topic)
    return _cache_instance


def get_cache() -> Optional[WorkflowCache]:
    """전역 캐시 인스턴스 반환"""
    return _cache_instance
