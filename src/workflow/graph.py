"""
LangGraph 워크플로우 정의 모듈
상태 정의, 노드 등록, 그래프 구성을 담당합니다.
"""

import logging
from typing import TypedDict, List, Dict, Annotated
from functools import wraps

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.script_writer import node_script_writer
from src.agents.tts import node_audio_generator
from src.agents.vision import node_visual_generator
from src.tools.video_editor import node_video_editor

logger = logging.getLogger(__name__)


# ============================================================================
# 상태 정의
# ============================================================================

class VideoState(TypedDict):
    """
    비디오 생성 워크플로우의 상태를 관리하는 TypedDict
    
    Attributes:
        topic: 비디오 주제
        scenes: 장면 리스트, 각 장면은 {"script": str, "image_prompt": str, "audio_path": str, "image_path": str} 형식
        final_video_path: 최종 비디오 파일 경로
    """
    topic: str
    scenes: List[Dict[str, str]]
    final_video_path: str


# ============================================================================
# 재시도 데코레이터
# ============================================================================

def retry_node(max_retries: int = 2, delay: float = 1.0):
    """
    노드 함수에 재시도 로직을 추가하는 데코레이터
    
    Args:
        max_retries: 최대 재시도 횟수
        delay: 재시도 간 대기 시간 (초)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: VideoState) -> VideoState:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(f"{func.__name__} 실행 (시도 {attempt + 1}/{max_retries + 1})")
                    return func(state)
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        import time
                        wait_time = delay * (attempt + 1)  # 지수 백오프
                        logger.warning(
                            f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}), "
                            f"{wait_time:.1f}초 후 재시도: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} 최종 실패: {e}", exc_info=True)
            
            # 모든 재시도 실패
            raise last_error
        
        return wrapper
    return decorator


# ============================================================================
# 노드 래퍼 (재시도 로직 포함)
# ============================================================================

@retry_node(max_retries=2, delay=2.0)
def node_script_writer_with_retry(state: VideoState) -> VideoState:
    """대본 작성 노드 (재시도 포함)"""
    return node_script_writer(state)


@retry_node(max_retries=2, delay=1.0)
def node_audio_generator_with_retry(state: VideoState) -> VideoState:
    """오디오 생성 노드 (재시도 포함)"""
    return node_audio_generator(state)


@retry_node(max_retries=2, delay=3.0)
def node_visual_generator_with_retry(state: VideoState) -> VideoState:
    """이미지 생성 노드 (재시도 포함)"""
    return node_visual_generator(state)


@retry_node(max_retries=1, delay=2.0)
def node_video_editor_with_retry(state: VideoState) -> VideoState:
    """비디오 편집 노드 (재시도 포함)"""
    return node_video_editor(state)


# ============================================================================
# 그래프 구성
# ============================================================================

def create_video_graph(checkpoint: bool = False) -> StateGraph:
    """
    LangGraph StateGraph를 생성하고 노드들을 연결합니다.
    
    Args:
        checkpoint: 체크포인트 사용 여부 (메모리 기반)
        
    Returns:
        컴파일된 StateGraph
    """
    logger.info("비디오 생성 그래프 구성 중...")
    
    # 그래프 생성
    if checkpoint:
        # 체크포인트 사용 (장기 실행 시 상태 복구 가능)
        memory = MemorySaver()
        workflow = StateGraph(VideoState).compile(checkpointer=memory)
    else:
        workflow = StateGraph(VideoState)
    
    # 노드 추가
    workflow.add_node("script_writer", node_script_writer_with_retry)
    workflow.add_node("audio_generator", node_audio_generator_with_retry)
    workflow.add_node("visual_generator", node_visual_generator_with_retry)
    workflow.add_node("video_editor", node_video_editor_with_retry)
    
    # 엣지 연결 (순차 실행)
    workflow.set_entry_point("script_writer")
    workflow.add_edge("script_writer", "audio_generator")
    workflow.add_edge("audio_generator", "visual_generator")
    workflow.add_edge("visual_generator", "video_editor")
    workflow.add_edge("video_editor", END)
    
    # 컴파일
    if not checkpoint:
        workflow = workflow.compile()
    
    logger.info("비디오 생성 그래프 구성 완료")
    
    return workflow


def create_video_graph_simple() -> StateGraph:
    """
    간단한 버전의 그래프 생성 (체크포인트 없음)
    
    Returns:
        컴파일된 StateGraph
    """
    return create_video_graph(checkpoint=False)

