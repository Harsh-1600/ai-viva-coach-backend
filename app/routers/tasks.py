"""Student tasks and notification router handling submissions, assignment review, and alerts."""

import logging
import uuid
import re
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import (
    StudentTaskItemResponse,
    TaskAssignmentResponse,
    NotificationResponse,
    MessageResponse,
)
from app.security import require_student, get_current_user
from app.services import db_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Student Tasks & Notifications"])

# Submissions upload directory
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
SUBMISSIONS_DIR = UPLOAD_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


@router.get(
    "/tasks/my-tasks",
    response_model=List[StudentTaskItemResponse],
    status_code=status.HTTP_200_OK,
    summary="List all tasks assigned to the authenticated student"
)
def get_my_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_student)
):
    """Retrieves all tasks assigned to the student with status (assigned, in_progress, submitted, completed, overdue)."""
    return db_service.get_student_tasks(db, student_id=current_user.id)


@router.post(
    "/tasks/{assignment_id}/submit",
    response_model=TaskAssignmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit work for an assigned task"
)
async def submit_task(
    assignment_id: int,
    submission_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_student)
):
    """
    Submits student work (text notes, file attachment, or both):
    - Updates status to 'submitted'
    - Saves submission timestamp
    - Alerts the teacher with an in-app notification
    """
    if not submission_text and not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide submission text or attach a file."
        )

    file_url = None
    if file and file.filename:
        # Validate file size (max 20MB)
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded file exceeds 20MB maximum size limit."
            )

        # Sanitize filename
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename)
        unique_prefix = uuid.uuid4().hex[:8]
        saved_filename = f"task_{assignment_id}_{unique_prefix}_{safe_name}"
        destination = SUBMISSIONS_DIR / saved_filename
        
        with open(destination, "wb") as f:
            f.write(contents)
        
        file_url = f"/uploads/submissions/{saved_filename}"

    assignment = db_service.submit_student_task(
        db,
        student_id=current_user.id,
        assignment_id=assignment_id,
        submission_text=submission_text,
        file_url=file_url
    )

    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found or does not belong to you."
        )

    return {
        "id": assignment.id,
        "task_id": assignment.task_id,
        "student_id": assignment.student_id,
        "student_name": assignment.student.name if assignment.student else current_user.name,
        "student_email": assignment.student.email if assignment.student else current_user.email,
        "status": assignment.status,
        "submission_text": assignment.submission_text,
        "submission_file_url": assignment.submission_file_url,
        "submission_date": assignment.submission_date,
        "teacher_feedback": assignment.teacher_feedback,
        "grade_score": assignment.grade_score,
        "completed_at": assignment.completed_at,
        "created_at": assignment.created_at,
    }


# ==============================================================================
# In-App Notifications
# ==============================================================================

@router.get(
    "/notifications",
    response_model=List[NotificationResponse],
    status_code=status.HTTP_200_OK,
    summary="List recent in-app notifications for authenticated user"
)
def get_notifications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves the recent alerts and notifications for the user."""
    return db_service.get_user_notifications(db, user_id=current_user.id)


@router.put(
    "/notifications/{notification_id}/read",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Mark an in-app notification as read"
)
def mark_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Marks a single notification as read."""
    success = db_service.mark_notification_as_read(db, notification_id=notification_id, user_id=current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found."
        )
    return MessageResponse(message="Notification marked as read.")


@router.put(
    "/notifications/read-all",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Mark all in-app notifications as read"
)
def mark_all_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Marks all pending notifications as read."""
    count = db_service.mark_all_notifications_as_read(db, user_id=current_user.id)
    return MessageResponse(message=f"Marked {count} notifications as read.")
