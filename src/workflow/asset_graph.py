"""
Asset Generation 서브그래프 모듈
오디오, 이미지, 모션 등 실제 에셋 생성을 담당합니다.
"""

import logging
from langgraph.graph import StateGraph, END

from src.workflow.graph import VideoState, retry_node, with_cache
from src.agents.tts import node_audio_generator
from src.agents.vision import node_visual_generator
from src.tools.motion_gen import node_motion_generator
from src.utils.cache import WorkflowCache
from src.config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# Asset Generation 서브그래프 노드 (캐싱 + 재시도 적용)
# ============================================================================

@with_cache(WorkflowCache.STAGE_AUDIO, ["scenes"])
@retry_node(max_retries=2, delay=1.0)
def node_audio_generator_with_retry(state: VideoState) -> VideoState:
    """오디오 생성 노드 (캐싱 + 재시도)"""
    return node_audio_generator(state)


@with_cache(WorkflowCache.STAGE_IMAGE, ["scenes"])
@retry_node(max_retries=2, delay=3.0)
def node_visual_generator_with_retry(state: VideoState) -> VideoState:
    """이미지 생성 노드 (캐싱 + 재시도)"""
    return node_visual_generator(state)


@with_cache(WorkflowCache.STAGE_MOTION, ["scenes"])
@retry_node(max_retries=2, delay=3.0)
def node_motion_generator_with_retry(state: VideoState) -> VideoState:
    """모션 생성 노드 (캐싱 + 재시도)"""
    return node_motion_generator(state)


# ============================================================================
# Asset Generation 서브그래프 구성
# ============================================================================

def build_asset_graph() -> StateGraph:
    """
    Asset Generation 서브그래프를 생성하고 컴파일합니다.
    
    이 서브그래프는 scenes 정보를 받아 오디오, 이미지, 모션 등 실제 에셋을 생성합니다.
    
    Returns:
        컴파일된 StateGraph 인스턴스
    """
    logger.debug("Asset Generation 서브그래프 구성 중...")
    
    # 그래프 생성
    workflow = StateGraph(VideoState)
    
    # 노드 추가
    workflow.add_node("audio_generator", node_audio_generator_with_retry)
    workflow.add_node("visual_generator", node_visual_generator_with_retry)
    
    # 모션 생성 노드는 활성화 여부에 따라 추가
    if Config.MOTION_ENABLED:
        workflow.add_node("motion_generator", node_motion_generator_with_retry)
    
    # 엣지 연결
    workflow.set_entry_point("audio_generator")
    workflow.add_edge("audio_generator", "visual_generator")
    
    if Config.MOTION_ENABLED:
        workflow.add_edge("visual_generator", "motion_generator")
        workflow.add_edge("motion_generator", END)
    else:
        workflow.add_edge("visual_generator", END)
    
    # 컴파일
    compiled_graph = workflow.compile()
    
    logger.debug("Asset Generation 서브그래프 구성 완료")
    return compiled_graph

