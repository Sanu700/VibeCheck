"""
Core Analysis Service for ToxicBuddy 2.0
Handles single-message evaluation, multi-message chat parsing, participant statistics,
and database persistence.
"""

import re
from datetime import datetime, timezone
from typing import List, Dict, Optional
from sqlalchemy.orm import Session

from ml.inference import get_analyzer
from ml.rewriter import get_rewriter
from backend.services.escalation_detector import get_escalation_detector
from backend.models import Conversation, Message, UserStat

class AnalysisService:
    """
    Main orchestration service for toxicity analysis.
    """

    def __init__(self):
        self.analyzer = get_analyzer()
        self.rewriter = get_rewriter()
        self.escalation_detector = get_escalation_detector()

    def analyze_single_message(
        self,
        text: str,
        sender: str = "User",
        requested_tones: Optional[List[str]] = None,
        context: Optional[List[str]] = None
    ) -> dict:
        """
        Analyzes a single message and generates rewrites. Supports optional conversational context.
        """
        if requested_tones is None:
            requested_tones = ["Neutral", "Friendly", "Professional", "Constructive"]

        analysis = self.analyzer.analyze_message(text, context=context)
        
        rewrites = {}
        for tone in requested_tones:
            rw_result = self.rewriter.rewrite(text, tone)
            rewrites[tone] = rw_result["rewritten_text"]

        return {
            "text": text,
            "sender": sender,
            "is_toxic": analysis["is_toxic"],
            "overall_score": analysis["overall_score"],
            "severity_level": analysis["severity_level"],
            "detected_tone": analysis["detected_tone"],
            "categories": analysis["categories"],
            "flagged_categories": analysis["flagged_categories"],
            "rewrites": rewrites
        }

    def parse_raw_chat(self, raw_text: str) -> List[Dict]:
        """
        Parses raw text chat exports (WhatsApp/Telegram/Slack format) into structured message list.
        """
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        parsed_messages = []

        regex_timestamp_sender = re.compile(r"^(?:\[?(.*?)\]?\s+)?([^:\n]+):\s+(.*)$")

        for line in lines:
            match = regex_timestamp_sender.match(line)
            if match:
                timestamp, sender, text = match.groups()
                sender = sender.strip()
                if not sender:
                    sender = "User"
                parsed_messages.append({
                    "sender": sender,
                    "text": text.strip(),
                    "timestamp": timestamp.strip() if timestamp else None
                })
            else:
                parsed_messages.append({
                    "sender": "User",
                    "text": line.strip(),
                    "timestamp": None
                })

        return parsed_messages

    def analyze_conversation(
        self,
        db: Session,
        title: Optional[str] = "Group Chat Analysis",
        messages_input: Optional[List[Dict]] = None,
        raw_chat_text: Optional[str] = None
    ) -> dict:
        """
        Performs end-to-end conversation analysis, aggregates user stats, detects escalations,
        and saves records to the database.
        """
        items_to_analyze = []

        if raw_chat_text and raw_chat_text.strip():
            items_to_analyze = self.parse_raw_chat(raw_chat_text)
        elif messages_input:
            items_to_analyze = messages_input
        else:
            items_to_analyze = []

        if not items_to_analyze:
            items_to_analyze = [{"sender": "User", "text": "No messages provided.", "timestamp": None}]

        analyzed_messages = []
        participant_data = {}
        time_series = []

        for idx, item in enumerate(items_to_analyze):
            sender = item.get("sender") or "User"
            text = item.get("text", "")
            timestamp = item.get("timestamp")

            # Collect preceding messages as context to disambiguate multi-turn sarcasm
            context_history = [
                m.get("text", "") for m in items_to_analyze[max(0, idx - 3):idx]
                if m.get("text")
            ]

            msg_res = self.analyze_single_message(
                text=text,
                sender=sender,
                context=context_history if context_history else None
            )
            msg_res["message_index"] = idx
            msg_res["timestamp"] = timestamp

            analyzed_messages.append(msg_res)

            if sender not in participant_data:
                participant_data[sender] = {
                    "user_id": sender,
                    "total_messages": 0,
                    "toxic_messages": 0,
                    "total_score_sum": 0.0,
                    "tones": []
                }
            
            participant_data[sender]["total_messages"] += 1
            if msg_res["is_toxic"]:
                participant_data[sender]["toxic_messages"] += 1
            participant_data[sender]["total_score_sum"] += msg_res["overall_score"]
            participant_data[sender]["tones"].append(msg_res["detected_tone"])

            time_series.append({
                "index": idx,
                "sender": sender,
                "timestamp": timestamp,
                "toxicity_score": msg_res["overall_score"],
                "is_toxic": msg_res["is_toxic"]
            })

        total_msgs = len(analyzed_messages)
        toxic_msgs_count = sum(1 for m in analyzed_messages if m["is_toxic"])
        toxic_pct = round((toxic_msgs_count / total_msgs) * 100.0 if total_msgs > 0 else 0.0, 2)
        
        overall_score = round(
            sum(m["overall_score"] for m in analyzed_messages) / total_msgs if total_msgs > 0 else 0.0, 4
        )

        if overall_score < 0.25 and toxic_pct < 15.0:
            health_status = "Healthy"
        elif overall_score < 0.55 and toxic_pct < 45.0:
            health_status = "Moderately Toxic"
        else:
            health_status = "Highly Toxic"

        participant_summary = {}
        for p_id, p_stats in participant_data.items():
            tot = p_stats["total_messages"]
            tox = p_stats["toxic_messages"]
            pct = round((tox / tot) * 100.0 if tot > 0 else 0.0, 2)
            avg_s = round(p_stats["total_score_sum"] / tot if tot > 0 else 0.0, 4)

            tone_counts = {}
            for t in p_stats["tones"]:
                tone_counts[t] = tone_counts.get(t, 0) + 1
            dominant_tone = max(tone_counts, key=tone_counts.get) if tone_counts else "Neutral"

            if avg_s >= 0.55 or pct >= 40.0:
                risk = "High"
            elif avg_s >= 0.25 or pct >= 15.0:
                risk = "Medium"
            else:
                risk = "Low"

            participant_summary[p_id] = {
                "user_id": p_id,
                "total_messages": tot,
                "toxic_messages": tox,
                "toxic_percentage": pct,
                "avg_toxicity_score": avg_s,
                "dominant_tone": dominant_tone,
                "risk_level": risk
            }

            db_user = db.query(UserStat).filter(UserStat.user_id == p_id).first()
            now_utc = datetime.now(timezone.utc)
            if not db_user:
                db_user = UserStat(
                    user_id=p_id,
                    total_messages=tot,
                    toxic_messages=tox,
                    avg_toxicity_score=avg_s,
                    primary_tone=dominant_tone,
                    last_active=now_utc
                )
                db.add(db_user)
            else:
                db_user.total_messages += tot
                db_user.toxic_messages += tox
                db_user.avg_toxicity_score = round((db_user.avg_toxicity_score + avg_s) / 2.0, 4)
                db_user.primary_tone = dominant_tone
                db_user.last_active = now_utc

        escalation_indicators = self.escalation_detector.detect_escalations(analyzed_messages)
        has_escalation = len(escalation_indicators) > 0

        conv_record = Conversation(
            title=title or "Group Chat Analysis",
            total_messages=total_msgs,
            toxic_messages_count=toxic_msgs_count,
            overall_toxicity_score=overall_score,
            health_status=health_status,
            has_escalation=has_escalation,
            participant_summary=participant_summary,
            escalation_summary=escalation_indicators
        )
        db.add(conv_record)
        db.flush()

        flagged_messages = []
        for msg in analyzed_messages:
            msg_db = Message(
                conversation_id=conv_record.id,
                sender=msg["sender"],
                text=msg["text"],
                timestamp=msg.get("timestamp"),
                message_index=msg["message_index"],
                is_toxic=msg["is_toxic"],
                overall_score=msg["overall_score"],
                severity_level=msg["severity_level"],
                detected_tone=msg["detected_tone"],
                categories=msg["categories"],
                flagged_categories=msg["flagged_categories"],
                rewrites=msg["rewrites"]
            )
            db.add(msg_db)
            if msg["is_toxic"]:
                flagged_messages.append(msg)

        db.commit()

        return {
            "conversation_id": conv_record.id,
            "title": conv_record.title,
            "created_at": conv_record.created_at,
            "health_status": health_status,
            "overall_toxicity_score": overall_score,
            "total_messages": total_msgs,
            "toxic_messages_count": toxic_msgs_count,
            "toxic_percentage": toxic_pct,
            "has_escalation": has_escalation,
            "participant_summary": participant_summary,
            "time_series_trend": time_series,
            "escalation_indicators": escalation_indicators,
            "flagged_messages": flagged_messages
        }

_analysis_service = None

def get_analysis_service() -> AnalysisService:
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
