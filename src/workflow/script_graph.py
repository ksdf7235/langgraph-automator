"""
Script Pipeline 서브그래프 모듈
대본 및 씬 메타데이터 생성을 담당합니다.
"""

import logging
from langgraph.graph import StateGraph, END

from src.workflow.graph import VideoState, retry_node, with_cache
from src.agents.script_writer import node_script_writer
from src.utils.cache import WorkflowCache

logger = logging.getLogger(__name__)


# ============================================================================
# Script Pipeline 서브그래프 노드 (캐싱 + 재시도 적용)
# ============================================================================

@with_cache(WorkflowCache.STAGE_SCRIPT, ["scenes"])
@retry_node(max_retries=2, delay=2.0)
def node_script_writer_with_retry(state: VideoState) -> VideoState:
    """대본 작성 노드 (캐싱 + 재시도)"""
    return node_script_writer(state)


# ============================================================================
# Script Pipeline 서브그래프 구성
# ============================================================================

def build_script_graph() -> StateGraph:
    """
    Script Pipeline 서브그래프를 생성하고 컴파일합니다.
    
    이 서브그래프는 topic을 받아 scenes(대본 및 이미지 프롬프트)를 생성합니다.
    
    Returns:
        컴파일된 StateGraph 인스턴스
    """
    logger.debug("Script Pipeline 서브그래프 구성 중...")
    
    # 그래프 생성
    workflow = StateGraph(VideoState)
    
    # 노드 추가
    workflow.add_node("script_writer", node_script_writer_with_retry)
    
    # 엣지 연결
    workflow.set_entry_point("script_writer")
    workflow.add_edge("script_writer", END)
    
    # 컴파일
    compiled_graph = workflow.compile()
    
    logger.debug("Script Pipeline 서브그래프 구성 완료")
    return compiled_graph

