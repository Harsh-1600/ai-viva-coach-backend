"""Performance reports, session analytics, and user history router."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, InterviewSession, SessionReport
from app.schemas import (
    DetailedSessionReportResponse,
    SessionHistoryItem,
    PerformanceStatisticsResponse,
)
from app.security import get_current_user
from app.services import db_service, report_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["Performance Reports & Analytics"])


@router.post(
    "/sessions/{session_id}",
    response_model=DetailedSessionReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate or compile a performance report for a completed session"
)
def generate_session_report(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generates or refreshes a comprehensive performance report for the specified interview session.
    - Protected: requires valid JWT bearer token.
    - User access control: returns 403 Forbidden if the session belongs to a different candidate.
    - Safely compiles overall score, rubric averages, strengths, gaps, and Recharts visual payloads.
    """
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interview session with ID {session_id} not found."
        )

    # Strict access control check
    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to access reports for this session."
        )

    # Mark session as completed if it was still in_progress
    if session.status != "completed":
        session.status = "completed"
        if not session.completed_at:
            session.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)

    try:
        report = report_service.compile_session_report(db, session=session, force_regenerate=True)
        return report_service.format_detailed_report_response(session=session, report=report)
    except Exception as exc:
        logger.error(f"Error generating session report for session {session_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compile performance report: {str(exc)}"
        )


@router.get(
    "/sessions/{session_id}",
    response_model=DetailedSessionReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve an interview session report by session ID"
)
def get_session_report(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves the performance report for a specific session.
    - If the session is finished but no report exists yet, automatically generates one on the fly.
    - Strict access control: returns 403 Forbidden if session belongs to another user.
    """
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interview session with ID {session_id} not found."
        )

    if session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not have permission to access reports for this session."
        )

    # If report doesn't exist, compile it
    report = session.report
    if not report:
        report = report_service.compile_session_report(db, session=session, force_regenerate=False)

    return report_service.format_detailed_report_response(session=session, report=report)


@router.get(
    "/history",
    response_model=List[SessionHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="Retrieve session history for the authenticated user"
)
def get_user_session_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns an chronological list of interview sessions belonging to the authenticated candidate.
    Includes session topics, question counts, completion statuses, and overall scores.
    """
    try:
        return report_service.get_candidate_session_history(db, user_id=current_user.id)
    except Exception as exc:
        logger.error(f"Error fetching session history for user {current_user.id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve session history."
        )


@router.get(
    "/statistics",
    response_model=PerformanceStatisticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve cross-session performance statistics and timeline"
)
def get_user_performance_statistics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Computes cross-session metrics:
    - Total sessions completed
    - Cumulative average score
    - Session-over-session score progress timeline for Recharts
    - Strongest and weakest topic proficiencies
    - 5-dimensional rubric skill profile
    """
    try:
        return report_service.get_candidate_performance_statistics(db, user_id=current_user.id)
    except Exception as exc:
        logger.error(f"Error fetching performance statistics for user {current_user.id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not compute performance statistics."
        )
