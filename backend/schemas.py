"""
Pydantic Schemas for Request & Response Validation in ToxicBuddy 2.0 API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

# --- Single Message Schemas ---

class MessageAnalysisRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "That is a stupid idea, stop wasting our time."})
    sender: Optional[str] = Field(default="User", json_schema_extra={"example": "Alice"})
    requested_tones: Optional[List[str]] = Field(
        default_factory=lambda: ["Neutral", "Friendly", "Professional", "Constructive"]
    )
    context: Optional[List[str]] = Field(
        default=None,
        description="Optional preceding messages to provide conversational context."
    )

class CategoryScores(BaseModel):
    toxic: float = 0.0
    insult: float = 0.0
    harassment: float = 0.0
    threat: float = 0.0
    obscene: float = 0.0
    identity_attack: float = 0.0

class MessageAnalysisResponse(BaseModel):
    text: str
    sender: str
    is_toxic: bool
    overall_score: float
    severity_level: str
    detected_tone: str
    categories: CategoryScores
    flagged_categories: List[str]
    rewrites: Dict[str, str]

# --- Rewrite Standalone Schema ---

class RewriteRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "Shut up and fix this broken code."})
    target_tone: Optional[str] = Field(default="Constructive", json_schema_extra={"example": "Constructive"})

class RewriteResponse(BaseModel):
    original_text: str
    rewritten_text: str
    target_tone: str
    changes_applied: bool

# --- Conversation Analysis Schemas ---

class ConversationItemInput(BaseModel):
    sender: Optional[str] = Field(default="User")
    text: str
    timestamp: Optional[str] = None

class ConversationAnalysisRequest(BaseModel):
    title: Optional[str] = Field(default="Group Chat Analysis")
    messages: Optional[List[ConversationItemInput]] = None
    raw_chat_text: Optional[str] = None

class ParticipantStat(BaseModel):
    user_id: str
    total_messages: int
    toxic_messages: int
    toxic_percentage: float
    avg_toxicity_score: float
    dominant_tone: str
    risk_level: str

class TimeSeriesPoint(BaseModel):
    index: int
    sender: str
    timestamp: Optional[str]
    toxicity_score: float
    is_toxic: bool

class EscalationIndicator(BaseModel):
    type: str  # e.g., "Toxicity Spike", "Back-and-Forth Conflict", "Severe Threat/Harassment"
    description: str
    message_indexes: List[int]
    severity: str

class ConversationReportResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    health_status: str  # "Healthy", "Moderately Toxic", "Highly Toxic"
    overall_toxicity_score: float
    total_messages: int
    toxic_messages_count: int
    toxic_percentage: float
    has_escalation: bool
    participant_summary: Dict[str, ParticipantStat]
    time_series_trend: List[TimeSeriesPoint]
    escalation_indicators: List[EscalationIndicator]
    flagged_messages: List[MessageAnalysisResponse]

# --- User Stats Schemas ---

class UserStatsResponse(BaseModel):
    user_id: str
    total_messages: int
    toxic_messages: int
    toxic_percentage: float
    avg_toxicity_score: float
    primary_tone: str
    risk_level: str  # "Low", "Medium", "High"
    last_active: datetime
