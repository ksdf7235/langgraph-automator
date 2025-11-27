"""
비디오 편집 도구 모듈
moviepy를 사용하여 이미지와 오디오를 결합하여 비디오를 생성합니다.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip
)

from src.config import Config

logger = logging.getLogger(__name__)


def apply_ken_burns_effect(clip: ImageClip, duration: float, zoom_factor: float = None) -> ImageClip:
    """
    Ken Burns 효과 (Zoom-in)를 이미지 클립에 적용합니다.
    moviepy의 fl() 함수를 사용하여 프레임별로 줌 효과를 적용합니다.
    
    Args:
        clip: 이미지 클립
        duration: 클립 길이 (초)
        zoom_factor: 줌 인 팩터 (기본값: Config.ZOOM_FACTOR)
        
    Returns:
        효과가 적용된 클립
    """
    zoom_factor = zoom_factor or Config.ZOOM_FACTOR
    w, h = clip.size
    
    def make_frame(t):
        # 시간에 따라 선형적으로 줌 인
        progress = min(t / duration, 1.0) if duration > 0 else 0
        current_zoom = 1.0 + (zoom_factor - 1.0) * progress
        
        # 원본 프레임 가져오기
        frame = clip.get_frame(0)  # 정적 이미지이므로 항상 첫 프레임
        
        # 현재 줌에 따른 크롭 크기
        crop_w = int(w / current_zoom)
        crop_h = int(h / current_zoom)
        
        # 중앙에서 크롭할 위치 계산
        x_center = (w - crop_w) // 2
        y_center = (h - crop_h) // 2
        
        # 크롭
        cropped = frame[y_center:y_center+crop_h, x_center:x_center+crop_w]
        
        # 원본 크기로 리사이즈 (간단한 보간 사용)
        try:
            from PIL import Image
            img = Image.fromarray(cropped)
            img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
            return np.array(img_resized)
        except ImportError:
            logger.warning("PIL이 설치되지 않아 간단한 리사이즈를 사용합니다.")
            # 간단한 리사이즈 (numpy만 사용)
            from scipy.ndimage import zoom as scipy_zoom
            zoom_ratio_w = w / crop_w
            zoom_ratio_h = h / crop_h
            resized = scipy_zoom(cropped, (zoom_ratio_h, zoom_ratio_w, 1), order=1)
            if resized.shape[0] != h or resized.shape[1] != w:
                resized = resized[:h, :w]
            return resized.astype(np.uint8)
    
    # 효과 적용
    try:
        zoomed_clip = clip.fl(make_frame, apply_to=['mask'])
    except Exception as e:
        logger.warning(f"Ken Burns 효과 적용 실패, 기본 resize 사용: {e}")
        # 폴백: 간단한 resize
        def zoom_func(t):
            progress = min(t / duration, 1.0) if duration > 0 else 0
            return 1.0 + (zoom_factor - 1.0) * progress
        zoomed_clip = clip.resize(zoom_func)
    
    return zoomed_clip.set_duration(duration)


def create_video_clip(image_path: str, audio_path: str, zoom_factor: float = None) -> ImageClip:
    """
    이미지와 오디오를 결합하여 비디오 클립을 생성합니다.
    
    Args:
        image_path: 이미지 파일 경로
        audio_path: 오디오 파일 경로
        zoom_factor: Ken Burns 효과 줌 팩터
        
    Returns:
        오디오가 결합된 이미지 클립
        
    Raises:
        FileNotFoundError: 이미지 또는 오디오 파일이 없는 경우
        ValueError: 오디오 길이가 0인 경우
    """
    # 파일 존재 확인
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일이 없습니다: {audio_path}")
    
    # 오디오 클립 로드
    try:
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        
        if audio_duration <= 0:
            raise ValueError(f"오디오 길이가 0입니다: {audio_path}")
        
        logger.debug(f"오디오 로드 완료: {audio_path} (길이: {audio_duration:.2f}초)")
    except Exception as e:
        logger.error(f"오디오 로드 실패 ({audio_path}): {e}")
        raise
    
    # 이미지 클립 생성
    try:
        image_clip = ImageClip(image_path)
        image_clip = image_clip.set_duration(audio_duration)
        
        # Ken Burns 효과 적용
        zoom_factor = zoom_factor or Config.ZOOM_FACTOR
        image_clip = apply_ken_burns_effect(image_clip, audio_duration, zoom_factor)
        
        # 오디오와 결합
        video_clip = image_clip.set_audio(audio_clip)
        video_clip = video_clip.set_fps(Config.VIDEO_FPS)
        
        logger.debug(f"비디오 클립 생성 완료: {image_path}")
        return video_clip
        
    except Exception as e:
        logger.error(f"이미지 클립 생성 실패 ({image_path}): {e}")
        audio_clip.close()
        raise


def compose_video(scenes: List[Dict], output_path: Path) -> None:
    """
    모든 장면을 결합하여 최종 비디오를 생성합니다.
    
    Args:
        scenes: 장면 리스트
        output_path: 출력 비디오 파일 경로
        
    Raises:
        ValueError: 편집할 클립이 없는 경우
        Exception: 비디오 생성 실패 시
    """
    logger.info(f"[비디오 편집] {len(scenes)}개 장면 편집 시작")
    
    video_clips = []
    
    try:
        for idx, scene in enumerate(scenes):
            image_path = scene.get('image_path', '')
            audio_path = scene.get('audio_path', '')
            
            # 파일 존재 확인 (강제 오류 발생)
            if not image_path:
                raise ValueError(f"장면 {idx+1}에 이미지 경로가 없습니다.")
            if not audio_path:
                raise ValueError(f"장면 {idx+1}에 오디오 경로가 없습니다.")
            
            logger.info(f"[비디오 편집] 장면 {idx+1} 처리 중...")
            
            # 비디오 클립 생성
            video_clip = create_video_clip(image_path, audio_path)
            video_clips.append(video_clip)
            
            logger.info(f"[비디오 편집] 장면 {idx+1} 완료 (길이: {video_clip.duration:.2f}초)")
        
        if not video_clips:
            raise ValueError("편집할 비디오 클립이 없습니다.")
        
        # 모든 클립을 이어 붙이기
        logger.info(f"[비디오 편집] {len(video_clips)}개 클립 결합 중...")
        final_clip = concatenate_videoclips(video_clips, method="compose")
        
        # 최종 비디오 내보내기
        total_duration = final_clip.duration
        logger.info(f"[비디오 편집] 최종 비디오 내보내기 중... (길이: {total_duration:.2f}초)")
        
        output_dir = output_path.parent
        temp_audio_path = output_dir / 'temp_audio.m4a'
        
        final_clip.write_videofile(
            str(output_path),
            fps=Config.VIDEO_FPS,
            codec=Config.VIDEO_CODEC,
            audio_codec=Config.VIDEO_AUDIO_CODEC,
            temp_audiofile=str(temp_audio_path),
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        logger.info(f"[비디오 편집] 완료: {output_path}")
        
    finally:
        # 리소스 정리
        for clip in video_clips:
            try:
                clip.close()
            except Exception as e:
                logger.warning(f"클립 정리 실패: {e}")


def node_video_editor(state: Dict) -> Dict:
    """
    LangGraph 노드 함수: 비디오 편집
    
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
        final_video_path = output_dir / "final_output.mp4"
        
        compose_video(scenes, final_video_path)
        
        return {
            **state,
            "final_video_path": str(final_video_path)
        }
        
    except Exception as e:
        logger.error(f"[비디오 편집 노드] 오류: {e}", exc_info=True)
        raise

