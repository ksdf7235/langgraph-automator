"""
Video Assembly 서브그래프 모듈
최종 비디오 편집 및 렌더링을 담당합니다.
"""

import logging
from langgraph.graph import StateGraph, END

from src.workflow.graph import VideoState, retry_node, with_cache
from src.tools.video_editor import node_video_editor
from src.utils.cache import WorkflowCache

logger = logging.getLogger(__name__)


# ============================================================================
# Video Assembly 서브그래프 노드 (캐싱 + 재시도 적용)
# ============================================================================

@with_cache(WorkflowCache.STAGE_VIDEO, ["final_video_path"])
@retry_node(max_retries=1, delay=2.0)
def node_video_editor_with_retry(state: VideoState) -> VideoState:
    """비디오 편집 노드 (캐싱 + 재시도)"""
    return node_video_editor(state)


# ============================================================================
# Video Assembly 서브그래프 구성
# ============================================================================

def build_video_graph() -> StateGraph:
    """
    Video Assembly 서브그래프를 생성하고 컴파일합니다.
    
    이 서브그래프는 scenes 정보를 받아 최종 비디오를 생성합니다.
    
    Returns:
        컴파일된 StateGraph 인스턴스
    """
    logger.debug("Video Assembly 서브그래프 구성 중...")
    
    # 그래프 생성
    workflow = StateGraph(VideoState)
    
    # 노드 추가
    workflow.add_node("video_editor", node_video_editor_with_retry)
    
    # 엣지 연결
    workflow.set_entry_point("video_editor")
    workflow.add_edge("video_editor", END)
    
    # 컴파일
    compiled_graph = workflow.compile()
    
    logger.debug("Video Assembly 서브그래프 구성 완료")
    return compiled_graph

