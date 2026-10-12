"""
API Endpoints for Message Analysis, Conversation Processing, and Rewriting
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import (
    MessageAnalysisRequest, MessageAnalysisResponse,
    ConversationAnalysisRequest, ConversationReportResponse,
    RewriteRequest, RewriteResponse
)
from backend.services.analysis_service import get_analysis_service
from ml.rewriter import get_rewriter

router = APIRouter(prefix="/analyze", tags=["Toxicity Analysis & Rewriting"])

@router.post("/message", response_model=MessageAnalysisResponse, summary="Analyze single message in real time")
def analyze_message(payload: MessageAnalysisRequest):
    """
    Real-time 'Before You Send' endpoint.
    Returns multi-label toxicity scores, confidence levels, tone, and constructive rewrites.
    """
    try:
        service = get_analysis_service()
        result = service.analyze_single_message(
            text=payload.text,
            sender=payload.sender or "User",
            requested_tones=payload.requested_tones,
            context=payload.context
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing message: {str(e)}"
        )

@router.post("/conversation", response_model=ConversationReportResponse, summary="Analyze group conversation")
def analyze_conversation(payload: ConversationAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analyzes an entire group chat / conversation.
    Calculates overall health, participant stats, time trends, escalation patterns, and rewrites.
    """
    try:
        service = get_analysis_service()
        messages_input = [m.model_dump() for m in payload.messages] if payload.messages else None
        
        result = service.analyze_conversation(
            db=db,
            title=payload.title,
            messages_input=messages_input,
            raw_chat_text=payload.raw_chat_text
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing conversation: {str(e)}"
        )

@router.post("/rewrite", response_model=RewriteResponse, summary="Standalone constructive text rewriting")
def rewrite_message(payload: RewriteRequest):
    """
    Generates a respectful, constructive alternative for a message in the selected target tone.
    """
    try:
        rewriter = get_rewriter()
        res = rewriter.rewrite(text=payload.text, tone=payload.target_tone or "Constructive")
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error rewriting text: {str(e)}"
        )
