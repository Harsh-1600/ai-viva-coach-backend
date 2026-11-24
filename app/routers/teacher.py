"""Teacher management router providing analytics, student performance tracking, and assignment workflows."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import (
    TeacherDashboardStatsResponse,
    StudentSummaryItem,
    StudentDetailPerformanceResponse,
    TaskCreateRequest,
    TaskUpdateRequest,
    TaskResponse,
    TaskDetailResponse,
    TaskAssignmentResponse,
    TaskReviewRequest,
    TeacherSendMessageRequest,
    TeacherSendMessageResponse,
)
from app.security import require_teacher
from app.services import db_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teacher", tags=["Teacher Management"])


@router.get(
    "/dashboard/stats",
    response_model=TeacherDashboardStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve high-level educator metrics"
)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """
    Returns high-level statistics for the teacher portal:
    - Total enrolled students
    - Active students (tested or engaged in tasks)
    - Total assigned tasks
    - Completed tasks count
    - Pending tasks count
    - Overdue tasks count
    """
    return db_service.get_teacher_dashboard_stats(db)


@router.get(
    "/students",
    response_model=List[StudentSummaryItem],
    status_code=status.HTTP_200_OK,
    summary="List all registered students with performance summaries"
)
def list_students(
    search: Optional[str] = Query(None, description="Search by student name or email"),
    sort_by: Optional[str] = Query(None, description="Sort by: 'score', 'recent', 'tasks'"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """Lists all candidate student accounts with performance metrics and task completion stats."""
    return db_service.get_students_list(db, search=search, sort_by=sort_by)


@router.get(
    "/students/{student_id}",
    response_model=StudentDetailPerformanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve deep-dive performance dossier for a student"
)
def get_student_performance(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """
    Fetches comprehensive student analytics:
    - Academic viva vs technical interview breakdown
    - Average evaluation scores across sessions
    - Subject-wise proficiency breakdown
    - Chronological list of completed sessions
    - Aggregated strengths and areas for improvement
    - Task completion status
    - Past formal evaluation reports
    """
    performance = db_service.get_student_detailed_performance(db, student_id=student_id)
    if not performance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID {student_id} not found."
        )
    return performance


@router.get(
    "/students/{student_id}/report",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Generate printable performance report data for a student"
)
def get_student_printable_report(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """Provides a consolidated JSON payload designed for export or printable student dossier views."""
    performance = db_service.get_student_detailed_performance(db, student_id=student_id)
    if not performance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student with ID {student_id} not found."
        )
    return {
        "generated_by": current_user.name,
        "teacher_email": current_user.email,
        "performance_dossier": performance,
    }


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new task and assign to students"
)
def create_task(
    task_in: TaskCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """
    Creates an academic assignment and distributes it to specified students:
    - Verifies student IDs
    - Records deadline and instructions
    - Triggers in-app alerts for assigned students
    """
    try:
        created = db_service.create_task(db, teacher_id=current_user.id, task_in=task_in)
        # Fetch detailed metrics wrapper
        task_detail = db_service.get_teacher_task_detail(db, teacher_id=current_user.id, task_id=created.id)
        if task_detail and "task" in task_detail:
            return task_detail["task"]
        return created
    except Exception as exc:
        logger.error(f"Error creating task: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create task: {str(exc)}"
        )


@router.get(
    "/tasks",
    response_model=List[TaskResponse],
    status_code=status.HTTP_200_OK,
    summary="List all tasks created by the logged-in teacher"
)
def list_teacher_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """Retrieves all tasks created by the educator with completion metrics."""
    return db_service.get_teacher_tasks(db, teacher_id=current_user.id)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve task details and individual student assignment records"
)
def get_task_details(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """Retrieves a specific task alongside all assigned student submission statuses."""
    task_detail = db_service.get_teacher_task_detail(db, teacher_id=current_user.id, task_id=task_id)
    if not task_detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found."
        )
    return task_detail


@router.put(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Update task details and deadline"
)
def update_task_details(
    task_id: int,
    task_update: TaskUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """Updates task metadata, subject, or deadline."""
    updated = db_service.update_task(
        db,
        teacher_id=current_user.id,
        task_id=task_id,
        task_update=task_update
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found or you lack permission to edit it."
        )
    task_detail = db_service.get_teacher_task_detail(db, teacher_id=current_user.id, task_id=task_id)
    if task_detail and "task" in task_detail:
        return task_detail["task"]
    return updated


@router.post(
    "/tasks/assignments/{assignment_id}/review",
    response_model=TaskAssignmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Review, score, and provide feedback on a student submission"
)
def review_assignment(
    assignment_id: int,
    review_in: TaskReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """
    Evaluates a student's submission:
    - Saves constructive feedback notes
    - Assigns grade score (0 - 100)
    - Updates status to 'completed' or requests revisions
    - Dispatches in-app notification to the student
    """
    assignment = db_service.review_student_task(
        db,
        teacher_id=current_user.id,
        assignment_id=assignment_id,
        feedback=review_in.teacher_feedback,
        score=review_in.grade_score,
        status=review_in.status
    )
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment with ID {assignment_id} not found."
        )
    return {
        "id": assignment.id,
        "task_id": assignment.task_id,
        "student_id": assignment.student_id,
        "student_name": assignment.student.name if assignment.student else "Student",
        "student_email": assignment.student.email if assignment.student else "",
        "status": assignment.status,
        "submission_text": assignment.submission_text,
        "submission_file_url": assignment.submission_file_url,
        "submission_date": assignment.submission_date,
        "teacher_feedback": assignment.teacher_feedback,
        "grade_score": assignment.grade_score,
        "completed_at": assignment.completed_at,
        "created_at": assignment.created_at,
    }


@router.post(
    "/reminders/run",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Manually trigger automatic pending task reminders"
)
def trigger_reminders(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """
    Checks all pending student assignments against due dates.
    Dispatches email and in-app reminders to students with overdue or approaching deadlines.
    Enforces 24-hour duplicate prevention.
    """
    return db_service.trigger_automatic_reminders(db)


@router.post(
    "/messages/send",
    response_model=TeacherSendMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a direct message or announcement from teacher to students"
)
def send_teacher_message(
    payload: TeacherSendMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher)
):
    """
    Allows authorized educators to send direct in-app messages or general announcements
    to a specific student or broadcast to all enrolled candidates.
    """
    try:
        result = db_service.send_teacher_message(
            db,
            teacher_id=current_user.id,
            title=payload.title,
            message=payload.message,
            student_id=payload.student_id,
            notification_type=payload.notification_type,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error sending teacher message: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message to students."
        )

