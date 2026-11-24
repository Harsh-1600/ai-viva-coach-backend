"""Database Service layer providing clean CRUD operations, error handling, and separation of concerns."""

import json
import logging
import secrets
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_, and_
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from app.models import (
    User,
    InterviewSession,
    Question,
    Answer,
    SessionReport,
    Task,
    TaskAssignment,
    Notification,
    EmailReminderLog,
    utc_now,
)
from app.schemas import (
    UserCreate,
    TeacherRegisterRequest,
    InterviewSessionCreate,
    QuestionCreate,
    AnswerCreate,
    SessionReportCreate,
    AnswerEvaluationResult,
    RubricBreakdown,
    TaskCreateRequest,
    TaskUpdateRequest,
    TaskSubmissionRequest,
    TaskReviewRequest,
)
from app.security import hash_password
from app.services.email_service import send_task_reminder_email
from app.config import get_settings

logger = logging.getLogger(__name__)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensures a datetime object is timezone-aware in UTC, converting naive SQLite datetimes."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ==============================================================================
# User CRUD Operations
# ==============================================================================

def create_user(
    db: Session,
    user_in: Union[UserCreate, TeacherRegisterRequest],
    role: str = "student"
) -> User:
    """
    Creates a new user account with a secure bcrypt password hash and specified role.
    Never stores plaintext passwords.
    """
    try:
        hashed_pwd = hash_password(user_in.password)
        db_user = User(
            name=user_in.name.strip(),
            email=user_in.email.lower().strip(),
            hashed_password=hashed_pwd,
            role=role.strip().lower()
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError as exc:
        db.rollback()
        logger.warning(f"Integrity error creating user with email {user_in.email}: {exc}")
        raise ValueError(f"A user with email '{user_in.email}' already exists.") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Database error creating user: {exc}")
        raise


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieves a user by email (case-insensitive)."""
    try:
        return db.query(User).filter(User.email == email.lower().strip()).first()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying user by email {email}: {exc}")
        return None


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Retrieves a user by primary key ID."""
    try:
        return db.query(User).filter(User.id == user_id).first()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying user by ID {user_id}: {exc}")
        return None


def get_user_by_google_id(db: Session, google_id: str) -> Optional[User]:
    """Retrieves a user by their unique Google subject identifier."""
    try:
        return db.query(User).filter(User.google_id == google_id).first()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying user by google_id {google_id}: {exc}")
        return None


def create_google_user(
    db: Session,
    *,
    email: str,
    name: str,
    google_id: str,
    avatar_url: Optional[str] = None
) -> User:
    """
    Creates a new user account authenticated via Google OAuth.
    Generates an unusable password hash sentinel so standard password login cannot succeed.
    Never stores plaintext credentials.
    """
    try:
        # Cryptographically secure random sentinel to satisfy NOT NULL constraints
        # Cannot be cracked, guessed, or matched by plain passwords
        unusable_pwd = f"!google_oauth_{secrets.token_hex(24)}"
        db_user = User(
            name=name.strip() if name and name.strip() else email.split("@")[0],
            email=email.lower().strip(),
            hashed_password=unusable_pwd,
            auth_provider="google",
            google_id=google_id,
            avatar_url=avatar_url,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except IntegrityError as exc:
        db.rollback()
        logger.warning(f"Integrity error creating Google user with email {email}: {exc}")
        raise ValueError(f"A user with email '{email}' already exists.") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Database error creating Google user: {exc}")
        raise


# ==============================================================================
# InterviewSession CRUD Operations
# ==============================================================================

def create_interview_session(db: Session, session_in: InterviewSessionCreate) -> InterviewSession:
    """Creates a new interview session for a user."""
    try:
        db_session = InterviewSession(
            user_id=session_in.user_id,
            topic=session_in.topic.strip(),
            subject=session_in.subject.strip(),
            target_level=session_in.target_level.strip(),
            interview_mode=session_in.interview_mode.strip(),
            status="in_progress"
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Database error creating interview session: {exc}")
        raise


def get_interview_session(db: Session, session_id: int) -> Optional[InterviewSession]:
    """Retrieves an interview session by primary key ID with loaded relationships."""
    try:
        return db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying interview session {session_id}: {exc}")
        return None


def get_user_interview_sessions(db: Session, user_id: int) -> List[InterviewSession]:
    """Retrieves all interview sessions belonging to a specific user."""
    try:
        return db.query(InterviewSession).filter(InterviewSession.user_id == user_id).all()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying sessions for user {user_id}: {exc}")
        return []


def update_session_status(
    db: Session,
    session_id: int,
    status: str,
    completed_at: Optional[datetime] = None
) -> Optional[InterviewSession]:
    """Updates session status (e.g. 'completed', 'abandoned') and completion timestamp."""
    try:
        session = db.query(InterviewSession).filter(InterviewSession.id == session_id).first()
        if not session:
            return None
        session.status = status
        if status == "completed" and completed_at is None:
            session.completed_at = datetime.now(timezone.utc)
        elif completed_at is not None:
            session.completed_at = completed_at
        db.commit()
        db.refresh(session)
        return session
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Error updating interview session {session_id}: {exc}")
        raise


# ==============================================================================
# Question CRUD Operations
# ==============================================================================

def create_question(db: Session, question_in: QuestionCreate) -> Question:
    """Adds a question to an interview session."""
    try:
        db_question = Question(
            session_id=question_in.session_id,
            question_text=question_in.question_text.strip(),
            question_type=question_in.question_type.strip(),
            difficulty=question_in.difficulty,
            question_order=question_in.question_order
        )
        db.add(db_question)
        db.commit()
        db.refresh(db_question)
        return db_question
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Error creating question for session {question_in.session_id}: {exc}")
        raise


def get_session_questions(db: Session, session_id: int) -> List[Question]:
    """Retrieves all questions for a session ordered by question_order."""
    try:
        return (
            db.query(Question)
            .filter(Question.session_id == session_id)
            .order_by(Question.question_order.asc())
            .all()
        )
    except SQLAlchemyError as exc:
        logger.error(f"Error querying questions for session {session_id}: {exc}")
        return []


def get_active_unanswered_question(db: Session, session_id: int) -> Optional[Question]:
    """Finds the lowest-order question in the session that does not have an answer yet."""
    try:
        unanswered = (
            db.query(Question)
            .outerjoin(Answer, Question.id == Answer.question_id)
            .filter(Question.session_id == session_id, Answer.id == None)
            .order_by(Question.question_order.asc())
            .first()
        )
        return unanswered
    except SQLAlchemyError as exc:
        logger.error(f"Error querying active question for session {session_id}: {exc}")
        return None


def count_session_follow_up_questions(db: Session, session_id: int) -> int:
    """Counts how many follow-up questions have been added to a session."""
    try:
        return (
            db.query(Question)
            .filter(Question.session_id == session_id, Question.question_type == "follow_up")
            .count()
        )
    except SQLAlchemyError as exc:
        logger.error(f"Error counting follow up questions for session {session_id}: {exc}")
        return 0


def insert_follow_up_question(
    db: Session,
    session_id: int,
    current_question_order: int,
    question_data: dict
) -> Question:
    """
    Shifts subsequent questions up by 1 and inserts the follow-up question
    immediately following current_question_order.
    """
    try:
        subsequent_questions = (
            db.query(Question)
            .filter(Question.session_id == session_id, Question.question_order > current_question_order)
            .order_by(Question.question_order.desc())
            .all()
        )
        for q in subsequent_questions:
            q.question_order += 1

        follow_up = Question(
            session_id=session_id,
            question_text=question_data["question_text"].strip(),
            question_type="follow_up",
            difficulty=question_data.get("difficulty", "medium"),
            question_order=current_question_order + 1
        )
        db.add(follow_up)
        db.commit()
        db.refresh(follow_up)
        return follow_up
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Error inserting follow-up question for session {session_id}: {exc}")
        raise


def calculate_session_average_score(db: Session, session_id: int) -> Optional[float]:
    """Calculates average score across all answered questions for a session."""
    try:
        answers = (
            db.query(Answer)
            .join(Question, Answer.question_id == Question.id)
            .filter(Question.session_id == session_id, Answer.score != None)
            .all()
        )
        if not answers:
            return None
        total = sum(a.score for a in answers if a.score is not None)
        return round(total / len(answers), 1)
    except SQLAlchemyError as exc:
        logger.error(f"Error calculating average score for session {session_id}: {exc}")
        return None


# ==============================================================================
# Answer CRUD Operations
# ==============================================================================

def create_answer(db: Session, answer_in: AnswerCreate) -> Answer:
    """
    Creates an evaluated answer for a specific question.
    Enforces 1:1 question-answer relationship and associates question, session, and evaluation.
    """
    try:
        def to_json_str(val: Any) -> Optional[str]:
            if val is None:
                return None
            if isinstance(val, (list, dict)):
                return json.dumps(val)
            return str(val)

        db_answer = Answer(
            question_id=answer_in.question_id,
            session_id=answer_in.session_id,
            answer_text=answer_in.answer_text.strip() if answer_in.answer_text else "",
            score=answer_in.score,
            correctness_status=answer_in.correctness_status,
            feedback=answer_in.feedback,
            evaluation_feedback=answer_in.evaluation_feedback or answer_in.feedback,
            strengths=answer_in.strengths,
            improvement_areas=answer_in.improvement_areas,
            correct_concepts=to_json_str(answer_in.correct_concepts),
            incorrect_concepts=to_json_str(answer_in.incorrect_concepts),
            missing_concepts=to_json_str(answer_in.missing_concepts),
            correct_answer=answer_in.correct_answer,
            simple_explanation=answer_in.simple_explanation,
            detailed_explanation=answer_in.detailed_explanation,
            example=answer_in.example,
            key_points_to_remember=to_json_str(answer_in.key_points_to_remember),
            improved_answer=getattr(answer_in, "improved_answer", None),
            evaluation_details=answer_in.evaluation_details,
            model_answer_provider=getattr(answer_in, "model_answer_provider", "Google Gemini"),
            model_answer_error=getattr(answer_in, "model_answer_error", None),
            model_answer_disclaimer=getattr(answer_in, "model_answer_disclaimer", None)
        )
        db.add(db_answer)
        db.commit()
        db.refresh(db_answer)
        return db_answer
    except IntegrityError as exc:
        db.rollback()
        logger.warning(f"Integrity error: question {answer_in.question_id} already has an answer: {exc}")
        raise ValueError(f"An answer already exists for question ID {answer_in.question_id}.") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Error creating answer for question {answer_in.question_id}: {exc}")
        raise


def get_question_answer(db: Session, question_id: int) -> Optional[Answer]:
    """Retrieves the answer associated with a question ID."""
    try:
        return db.query(Answer).filter(Answer.question_id == question_id).first()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying answer for question {question_id}: {exc}")
        return None


def format_answer_evaluation_result(answer: Answer, question: Optional[Question] = None) -> AnswerEvaluationResult:
    """
    Safely reconstructs the full AnswerEvaluationResult from stored Answer columns,
    falling back to evaluation_details JSON when needed.
    """
    details: Dict[str, Any] = {}
    if answer.evaluation_details:
        try:
            parsed = json.loads(answer.evaluation_details)
            if isinstance(parsed, dict):
                details = parsed
        except Exception as exc:
            logger.warning(f"Error parsing evaluation_details for answer {answer.id}: {exc}")

    def parse_str_list(val: Any) -> List[str]:
        if not val:
            return []
        if isinstance(val, list):
            return [str(x) for x in val if x]
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x]
            except Exception:
                pass
            return [line.strip().lstrip("•- ") for line in val.split("\n") if line.strip()]
        return []

    score = answer.score if answer.score is not None else float(details.get("score") or details.get("numerical_score") or 0.0)
    status = answer.correctness_status or details.get("correctness_status")
    if not status:
        if score >= 75.0:
            status = "correct"
        elif score >= 40.0:
            status = "partially_correct"
        elif len(answer.answer_text.strip()) == 0:
            status = "empty"
        else:
            status = "incorrect"

    correct_c = parse_str_list(answer.correct_concepts) or parse_str_list(details.get("correct_concepts") or details.get("correct_concepts_identified"))
    incorrect_c = parse_str_list(answer.incorrect_concepts) or parse_str_list(details.get("incorrect_concepts") or details.get("technical_inaccuracies"))
    missing_c = parse_str_list(answer.missing_concepts) or parse_str_list(details.get("missing_concepts"))
    key_points = parse_str_list(answer.key_points_to_remember) or parse_str_list(details.get("key_points_to_remember"))

    correct_ans = answer.correct_answer or details.get("correct_answer") or ""
    simple_exp = answer.simple_explanation or details.get("simple_explanation") or ""
    detailed_exp = answer.detailed_explanation or details.get("detailed_explanation") or ""
    example_text = answer.example or details.get("example")
    imp_ans = getattr(answer, "improved_answer", None) or details.get("improved_answer")

    rb_data = details.get("rubric_breakdown")
    rubric_obj = None
    if isinstance(rb_data, dict):
        try:
            rubric_obj = RubricBreakdown(**rb_data)
        except Exception:
            pass

    feedback_text = answer.evaluation_feedback or answer.feedback or details.get("evaluation_feedback") or details.get("brief_explanation") or ""

    return AnswerEvaluationResult(
        question_id=answer.question_id,
        score=score,
        correctness_status=status,
        evaluation_feedback=feedback_text,
        correct_concepts=correct_c,
        incorrect_concepts=incorrect_c,
        missing_concepts=missing_c,
        correct_answer=correct_ans,
        simple_explanation=simple_exp,
        detailed_explanation=detailed_exp,
        example=example_text,
        key_points_to_remember=key_points,
        improved_answer=imp_ans,
        candidate_answer=answer.answer_text,
        is_correct=(status == "correct"),
        feedback=feedback_text,
        strengths=answer.strengths,
        improvement_areas=answer.improvement_areas,
        technical_inaccuracies=incorrect_c,
        rubric_breakdown=rubric_obj,
        next_difficulty=details.get("next_difficulty_recommendation") or details.get("next_difficulty"),
        is_ai_estimate=details.get("is_ai_estimate", True),
        model_answer_provider=getattr(answer, "model_answer_provider", None) or details.get("model_answer_provider") or "Google Gemini",
        model_answer_error=getattr(answer, "model_answer_error", None) or details.get("model_answer_error"),
        model_answer_disclaimer=getattr(answer, "model_answer_disclaimer", None) or details.get("model_answer_disclaimer") or "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
    )


# ==============================================================================
# SessionReport CRUD Operations
# ==============================================================================

def create_session_report(db: Session, report_in: SessionReportCreate) -> SessionReport:
    """
    Creates the final session evaluation report.
    Enforces 1:1 session-report relationship.
    """
    try:
        db_report = SessionReport(
            session_id=report_in.session_id,
            overall_score=report_in.overall_score,
            summary=report_in.summary,
            strengths=report_in.strengths,
            improvement_areas=report_in.improvement_areas,
            recommendations=report_in.recommendations
        )
        db.add(db_report)
        db.commit()
        db.refresh(db_report)
        return db_report
    except IntegrityError as exc:
        db.rollback()
        logger.warning(f"Integrity error: session {report_in.session_id} already has a report: {exc}")
        raise ValueError(f"A report already exists for session ID {report_in.session_id}.") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(f"Error creating report for session {report_in.session_id}: {exc}")
        raise


def get_session_report(db: Session, session_id: int) -> Optional[SessionReport]:
    """Retrieves the evaluation report for a session ID."""
    try:
        return db.query(SessionReport).filter(SessionReport.session_id == session_id).first()
    except SQLAlchemyError as exc:
        logger.error(f"Error querying report for session {session_id}: {exc}")
        return None


# ==============================================================================
# Teacher Management & Student Performance Monitoring Services
# ==============================================================================

def get_teacher_dashboard_stats(db: Session) -> Dict[str, int]:
    """
    Computes high-level statistics from real database records for the Teacher Dashboard.
    - Total Students
    - Active Students (conducted sessions or task interactions in last 30 days)
    - Total Assigned Tasks
    - Completed Tasks
    - Pending Tasks
    - Overdue Tasks
    """
    now = utc_now()
    thirty_days_ago = now - timedelta(days=30)

    # 1. Total Students
    total_students = db.query(func.count(User.id)).filter(User.role == "student").scalar() or 0

    # 2. Active Students: students with at least 1 session or task interaction in last 30 days
    active_session_ids = [
        r[0]
        for r in db.query(InterviewSession.user_id)
        .filter(InterviewSession.started_at >= thirty_days_ago)
        .distinct()
        .all()
    ]
    active_task_ids = [
        r[0]
        for r in db.query(TaskAssignment.student_id)
        .filter(TaskAssignment.created_at >= thirty_days_ago)
        .distinct()
        .all()
    ]
    active_id_set = set(active_session_ids) | set(active_task_ids)
    active_students_count = (
        db.query(func.count(User.id))
        .filter(User.role == "student", User.id.in_(active_id_set))
        .scalar()
        if active_id_set
        else 0
    )

    # 3. Task Metrics
    total_assigned_tasks = db.query(func.count(TaskAssignment.id)).scalar() or 0
    completed_tasks = (
        db.query(func.count(TaskAssignment.id))
        .filter(TaskAssignment.status == "completed")
        .scalar()
        or 0
    )
    pending_tasks = (
        db.query(func.count(TaskAssignment.id))
        .filter(TaskAssignment.status.in_(["assigned", "in_progress"]))
        .scalar()
        or 0
    )

    # 4. Overdue Tasks: due_date < now and not completed
    overdue_tasks = (
        db.query(func.count(TaskAssignment.id))
        .join(Task, TaskAssignment.task_id == Task.id)
        .filter(
            Task.due_date < now,
            TaskAssignment.status.notin_(["completed", "submitted"]),
        )
        .scalar()
        or 0
    )

    return {
        "total_students": total_students,
        "active_students": active_students_count,
        "total_assigned_tasks": total_assigned_tasks,
        "completed_tasks": completed_tasks,
        "pending_tasks": pending_tasks,
        "overdue_tasks": overdue_tasks,
    }


def get_students_list(
    db: Session,
    search: Optional[str] = None,
    sort_by: Optional[str] = "name"
) -> List[Dict[str, Any]]:
    """
    Returns a comprehensive directory of enrolled students with real calculated metrics:
    - Student Name, Email, Registration Date
    - Total Viva Sessions
    - Total Interview Sessions
    - Average Performance Score
    - Completed Tasks
    - Pending Tasks
    """
    query = db.query(User).filter(User.role == "student")

    if search and search.strip():
        term = f"%{search.strip().lower()}%"
        query = query.filter(or_(func.lower(User.name).like(term), func.lower(User.email).like(term)))

    students = query.order_by(User.name.asc()).all()
    results = []

    for student in students:
        # Viva Sessions count (academic_viva)
        total_viva = (
            db.query(func.count(InterviewSession.id))
            .filter(
                InterviewSession.user_id == student.id,
                InterviewSession.interview_mode == "academic_viva",
            )
            .scalar()
            or 0
        )

        # Technical Interview Sessions count
        total_interview = (
            db.query(func.count(InterviewSession.id))
            .filter(
                InterviewSession.user_id == student.id,
                InterviewSession.interview_mode != "academic_viva",
            )
            .scalar()
            or 0
        )

        # Average Performance Score across SessionReports
        avg_score = (
            db.query(func.avg(SessionReport.overall_score))
            .join(InterviewSession, SessionReport.session_id == InterviewSession.id)
            .filter(InterviewSession.user_id == student.id)
            .scalar()
        )
        if avg_score is not None:
            avg_score = round(float(avg_score), 1)

        # Completed tasks count
        completed_tasks = (
            db.query(func.count(TaskAssignment.id))
            .filter(
                TaskAssignment.student_id == student.id,
                TaskAssignment.status == "completed",
            )
            .scalar()
            or 0
        )

        # Pending tasks count
        pending_tasks = (
            db.query(func.count(TaskAssignment.id))
            .filter(
                TaskAssignment.student_id == student.id,
                TaskAssignment.status.in_(["assigned", "in_progress", "overdue"]),
            )
            .scalar()
            or 0
        )

        results.append({
            "id": student.id,
            "name": student.name,
            "email": student.email,
            "avatar_url": student.avatar_url,
            "registered_at": student.created_at,
            "total_viva_sessions": total_viva,
            "total_interview_sessions": total_interview,
            "average_performance_score": avg_score,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
        })

    # Sort results if requested
    if sort_by == "score":
        results.sort(key=lambda s: s["average_performance_score"] or 0, reverse=True)
    elif sort_by == "recent":
        results.sort(key=lambda s: s["registered_at"], reverse=True)
    elif sort_by == "tasks":
        results.sort(key=lambda s: s["completed_tasks"], reverse=True)

    return results


def get_student_detailed_performance(db: Session, student_id: int) -> Optional[Dict[str, Any]]:
    """
    Computes a complete student performance dossier for the teacher monitoring page:
    - Profile information
    - Viva vs Technical Interview performance
    - Assessment scores & Subject-wise performance breakdown
    - Completed sessions list
    - Aggregated strengths and areas for improvement from actual reports
    - Score progression timeline
    - Previous performance reports
    """
    student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not student:
        return None

    # Sessions query
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == student.id)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )

    viva_sessions = [s for s in sessions if s.interview_mode == "academic_viva"]
    interview_sessions = [s for s in sessions if s.interview_mode != "academic_viva"]

    # Reports query
    session_ids = [s.id for s in sessions]
    reports = []
    if session_ids:
        reports = (
            db.query(SessionReport)
            .filter(SessionReport.session_id.in_(session_ids))
            .order_by(SessionReport.created_at.desc())
            .all()
        )

    report_map = {r.session_id: r for r in reports}

    # Scores
    scores = [r.overall_score for r in reports if r.overall_score is not None]
    overall_avg = round(sum(scores) / len(scores), 1) if scores else None

    # Viva Stats
    viva_reports = [report_map[s.id] for s in viva_sessions if s.id in report_map and report_map[s.id].overall_score is not None]
    viva_avg = (
        round(sum(r.overall_score for r in viva_reports) / len(viva_reports), 1)
        if viva_reports
        else None
    )

    # Interview Stats
    interview_reports = [report_map[s.id] for s in interview_sessions if s.id in report_map and report_map[s.id].overall_score is not None]
    interview_avg = (
        round(sum(r.overall_score for r in interview_reports) / len(interview_reports), 1)
        if interview_reports
        else None
    )

    # Subject-wise Breakdown
    subject_map: Dict[str, List[float]] = {}
    for s in sessions:
        rep = report_map.get(s.id)
        if rep and rep.overall_score is not None:
            subj = s.subject or "General"
            subject_map.setdefault(subj, []).append(rep.overall_score)

    subject_breakdown = [
        {
            "subject": subj,
            "session_count": len(sc_list),
            "average_score": round(sum(sc_list) / len(sc_list), 1),
        }
        for subj, sc_list in subject_map.items()
    ]
    subject_breakdown.sort(key=lambda x: x["average_score"], reverse=True)

    # Recent Sessions
    recent_sessions = []
    for s in sessions[:8]:
        rep = report_map.get(s.id)
        recent_sessions.append({
            "session_id": s.id,
            "subject": s.subject,
            "topic": s.topic,
            "mode": s.interview_mode,
            "status": s.status,
            "started_at": s.started_at,
            "score": rep.overall_score if rep else None,
        })

    # Aggregated Strengths & Improvement Areas from actual reports
    strengths_set = []
    weaknesses_set = []
    for r in reports:
        if r.strengths:
            for line in r.strengths.split("\n"):
                clean = line.strip().lstrip("•-*0123456789. ")
                if clean and clean not in strengths_set and len(clean) > 5:
                    strengths_set.append(clean)
        if r.improvement_areas:
            for line in r.improvement_areas.split("\n"):
                clean = line.strip().lstrip("•-*0123456789. ")
                if clean and clean not in weaknesses_set and len(clean) > 5:
                    weaknesses_set.append(clean)

    # Task metrics
    completed_tasks_count = (
        db.query(func.count(TaskAssignment.id))
        .filter(TaskAssignment.student_id == student.id, TaskAssignment.status == "completed")
        .scalar()
        or 0
    )
    pending_tasks_count = (
        db.query(func.count(TaskAssignment.id))
        .filter(
            TaskAssignment.student_id == student.id,
            TaskAssignment.status.in_(["assigned", "in_progress", "overdue"]),
        )
        .scalar()
        or 0
    )

    # Previous Reports list
    previous_reports = [
        {
            "id": r.id,
            "session_id": r.session_id,
            "overall_score": r.overall_score,
            "summary": r.summary,
            "created_at": r.created_at,
        }
        for r in reports[:10]
    ]

    return {
        "profile": student,
        "viva_stats": {
            "total_sessions": len(viva_sessions),
            "completed_sessions": len([s for s in viva_sessions if s.status == "completed"]),
            "average_score": viva_avg,
            "recent_topics": list(dict.fromkeys(s.topic for s in viva_sessions[:5])),
        },
        "interview_stats": {
            "total_sessions": len(interview_sessions),
            "completed_sessions": len([s for s in interview_sessions if s.status == "completed"]),
            "average_score": interview_avg,
            "recent_topics": list(dict.fromkeys(s.topic for s in interview_sessions[:5])),
        },
        "overall_average_score": overall_avg,
        "subject_breakdown": subject_breakdown,
        "recent_sessions": recent_sessions,
        "strengths": strengths_set[:6],
        "improvement_areas": weaknesses_set[:6],
        "completed_tasks_count": completed_tasks_count,
        "pending_tasks_count": pending_tasks_count,
        "previous_reports": previous_reports,
    }


# ==============================================================================
# Task Assignment & Management Services
# ==============================================================================

def create_task(db: Session, teacher_id: int, task_in: TaskCreateRequest) -> Task:
    """
    Creates a new assignment and allocates it to one or multiple students.
    Generates in-app notifications for each enrolled student.
    """
    now = utc_now()
    db_task = Task(
        teacher_id=teacher_id,
        title=task_in.title.strip(),
        description=task_in.description.strip() if task_in.description else None,
        subject=task_in.subject.strip(),
        topic=task_in.topic.strip() if task_in.topic else None,
        priority=task_in.priority.lower().strip() if task_in.priority else "medium",
        instructions=task_in.instructions.strip() if task_in.instructions else None,
        assigned_date=now,
        due_date=task_in.due_date,
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # Assign to students
    for student_id in set(task_in.assigned_student_ids):
        # Verify student exists and has student role
        student = db.query(User).filter(User.id == student_id, User.role == "student").first()
        if student:
            assignment = TaskAssignment(
                task_id=db_task.id,
                student_id=student.id,
                status="assigned",
                created_at=now,
            )
            db.add(assignment)

            # In-app notification for student
            notif = Notification(
                user_id=student.id,
                title="New Task Assigned",
                message=f'Your teacher assigned "{db_task.title}" ({db_task.subject}). Due on {db_task.due_date.strftime("%b %d, %Y")}.',
                notification_type="task_assigned",
                related_task_id=db_task.id,
                created_at=now,
            )
            db.add(notif)

    db.commit()
    db.refresh(db_task)
    return db_task


def update_task(
    db: Session,
    teacher_id: int,
    task_id: int,
    task_update: TaskUpdateRequest
) -> Optional[Task]:
    """Updates task properties before completion."""
    task = db.query(Task).filter(Task.id == task_id, Task.teacher_id == teacher_id).first()
    if not task:
        return None

    if task_update.title is not None:
        task.title = task_update.title.strip()
    if task_update.description is not None:
        task.description = task_update.description.strip()
    if task_update.subject is not None:
        task.subject = task_update.subject.strip()
    if task_update.topic is not None:
        task.topic = task_update.topic.strip()
    if task_update.priority is not None:
        task.priority = task_update.priority.lower().strip()
    if task_update.instructions is not None:
        task.instructions = task_update.instructions.strip()
    if task_update.due_date is not None:
        task.due_date = task_update.due_date

    task.updated_at = utc_now()
    db.commit()
    db.refresh(task)
    return task


def get_teacher_tasks(db: Session, teacher_id: int) -> List[Dict[str, Any]]:
    """Retrieves all tasks created by the teacher with aggregated status counts."""
    tasks = db.query(Task).filter(Task.teacher_id == teacher_id).order_by(Task.created_at.desc()).all()
    results = []
    now = utc_now()

    for task in tasks:
        assignments = db.query(TaskAssignment).filter(TaskAssignment.task_id == task.id).all()
        total_assigned = len(assignments)
        completed_count = len([a for a in assignments if a.status == "completed"])
        submitted_count = len([a for a in assignments if a.status == "submitted"])
        pending_count = len([a for a in assignments if a.status in ("assigned", "in_progress")])
        task_due = ensure_utc(task.due_date)
        is_overdue = task_due is not None and task_due < now and completed_count < total_assigned

        results.append({
            "id": task.id,
            "teacher_id": task.teacher_id,
            "teacher_name": task.teacher.name if task.teacher else "Teacher",
            "title": task.title,
            "description": task.description,
            "subject": task.subject,
            "topic": task.topic,
            "priority": task.priority,
            "instructions": task.instructions,
            "assigned_date": task.assigned_date,
            "due_date": task.due_date,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "total_assigned": total_assigned,
            "completed_count": completed_count,
            "submitted_count": submitted_count,
            "pending_count": pending_count,
            "is_overdue": is_overdue,
        })

    return results


def get_teacher_task_detail(db: Session, teacher_id: int, task_id: int) -> Optional[Dict[str, Any]]:
    """Returns task metadata and individual assignment records."""
    task = db.query(Task).filter(Task.id == task_id, Task.teacher_id == teacher_id).first()
    if not task:
        return None

    now = utc_now()
    assignments = (
        db.query(TaskAssignment)
        .filter(TaskAssignment.task_id == task.id)
        .order_by(TaskAssignment.created_at.asc())
        .all()
    )

    assignment_items = []
    for a in assignments:
        assignment_items.append({
            "id": a.id,
            "task_id": a.task_id,
            "student_id": a.student_id,
            "student_name": a.student.name if a.student else "Student",
            "student_email": a.student.email if a.student else "",
            "status": a.status,
            "submission_text": a.submission_text,
            "submission_file_url": a.submission_file_url,
            "submission_date": a.submission_date,
            "teacher_feedback": a.teacher_feedback,
            "grade_score": a.grade_score,
            "completed_at": a.completed_at,
            "created_at": a.created_at,
        })

    completed_count = len([a for a in assignments if a.status == "completed"])
    submitted_count = len([a for a in assignments if a.status == "submitted"])
    pending_count = len([a for a in assignments if a.status in ("assigned", "in_progress")])
    task_due = ensure_utc(task.due_date)

    task_data = {
        "id": task.id,
        "teacher_id": task.teacher_id,
        "teacher_name": task.teacher.name if task.teacher else "Teacher",
        "title": task.title,
        "description": task.description,
        "subject": task.subject,
        "topic": task.topic,
        "priority": task.priority,
        "instructions": task.instructions,
        "assigned_date": task.assigned_date,
        "due_date": task.due_date,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "total_assigned": len(assignments),
        "completed_count": completed_count,
        "submitted_count": submitted_count,
        "pending_count": pending_count,
        "is_overdue": task_due is not None and task_due < now and completed_count < len(assignments),
    }

    return {"task": task_data, "assignments": assignment_items}


def get_student_tasks(db: Session, student_id: int) -> List[Dict[str, Any]]:
    """Returns all task assignments for a specific student."""
    now = utc_now()
    assignments = (
        db.query(TaskAssignment)
        .join(Task, TaskAssignment.task_id == Task.id)
        .filter(TaskAssignment.student_id == student_id)
        .order_by(Task.due_date.asc())
        .all()
    )

    results = []
    for a in assignments:
        a_task_due = ensure_utc(a.task.due_date)
        is_overdue = (
            a_task_due is not None and a_task_due < now and a.status not in ("completed", "submitted")
        )
        display_status = "overdue" if is_overdue else a.status

        results.append({
            "assignment_id": a.id,
            "task_id": a.task_id,
            "title": a.task.title,
            "description": a.task.description,
            "subject": a.task.subject,
            "topic": a.task.topic,
            "priority": a.task.priority,
            "instructions": a.task.instructions,
            "assigned_date": a.task.assigned_date,
            "due_date": a.task.due_date,
            "status": display_status,
            "submission_text": a.submission_text,
            "submission_file_url": a.submission_file_url,
            "submission_date": a.submission_date,
            "teacher_feedback": a.teacher_feedback,
            "grade_score": a.grade_score,
            "completed_at": a.completed_at,
            "teacher_name": a.task.teacher.name if a.task.teacher else "Instructor",
            "is_overdue": is_overdue,
        })

    return results


def submit_student_task(
    db: Session,
    student_id: int,
    assignment_id: int,
    submission_text: Optional[str] = None,
    file_url: Optional[str] = None
) -> Optional[TaskAssignment]:
    """
    Records student work submission and notifies the teacher.
    Updates assignment status to 'submitted'.
    """
    assignment = (
        db.query(TaskAssignment)
        .filter(TaskAssignment.id == assignment_id, TaskAssignment.student_id == student_id)
        .first()
    )
    if not assignment:
        return None

    now = utc_now()
    assignment.submission_text = submission_text.strip() if submission_text else None
    if file_url:
        assignment.submission_file_url = file_url
    assignment.submission_date = now
    assignment.status = "submitted"

    # In-app notification to teacher
    task = assignment.task
    if task and task.teacher_id:
        notif = Notification(
            user_id=task.teacher_id,
            title="Task Submitted",
            message=f'{assignment.student.name} submitted task "{task.title}".',
            notification_type="task_submitted",
            related_task_id=task.id,
            created_at=now,
        )
        db.add(notif)

    db.commit()
    db.refresh(assignment)
    return assignment


def review_student_task(
    db: Session,
    teacher_id: int,
    assignment_id: int,
    feedback: Optional[str] = None,
    score: Optional[float] = None,
    status: str = "completed"
) -> Optional[TaskAssignment]:
    """
    Allows a teacher to grade and provide evaluation feedback on a student submission.
    Notifies the student with feedback details.
    """
    assignment = (
        db.query(TaskAssignment)
        .join(Task, TaskAssignment.task_id == Task.id)
        .filter(TaskAssignment.id == assignment_id, Task.teacher_id == teacher_id)
        .first()
    )
    if not assignment:
        return None

    now = utc_now()
    if feedback is not None:
        assignment.teacher_feedback = feedback.strip()
    if score is not None:
        assignment.grade_score = score
    assignment.status = status.lower().strip()
    if assignment.status == "completed":
        assignment.completed_at = now

    # In-app notification to student
    notif = Notification(
        user_id=assignment.student_id,
        title=f'Teacher Feedback: "{assignment.task.title}"',
        message=f'Your teacher provided feedback and graded your work (Score: {score if score is not None else "Reviewed"}).',
        notification_type="teacher_feedback" if assignment.status != "completed" else "task_completed",
        related_task_id=assignment.task_id,
        created_at=now,
    )
    db.add(notif)

    db.commit()
    db.refresh(assignment)
    return assignment


# ==============================================================================
# Automatic Email Reminders & Notifications
# ==============================================================================

def trigger_automatic_reminders(db: Session) -> Dict[str, Any]:
    """
    Checks assigned tasks against deadlines and dispatches reminders for incomplete tasks.
    Enforces a duplicate reminder cooldown (e.g. 24h) and records delivery audit log.
    """
    settings = get_settings()
    now = utc_now()
    cooldown = timedelta(hours=settings.EMAIL_REMINDER_COOLDOWN_HOURS)

    # Find assignments whose deadline has passed or is approaching within 24h, and is not completed/submitted
    overdue_assignments = (
        db.query(TaskAssignment)
        .join(Task, TaskAssignment.task_id == Task.id)
        .filter(
            Task.due_date <= now,
            TaskAssignment.status.notin_(["completed", "submitted"]),
        )
        .all()
    )

    reminders_sent = 0
    reminders_skipped_cooldown = 0
    delivery_logs = []

    for a in overdue_assignments:
        student = a.student
        task = a.task
        if not student or not student.email:
            continue

        # Check last reminder log for duplicate prevention
        last_log = (
            db.query(EmailReminderLog)
            .filter(EmailReminderLog.task_assignment_id == a.id)
            .order_by(EmailReminderLog.sent_at.desc())
            .first()
        )

        if last_log and last_log.sent_at and (ensure_utc(now) - ensure_utc(last_log.sent_at)) < cooldown:
            reminders_skipped_cooldown += 1
            continue

        due_date_str = task.due_date.strftime("%B %d, %Y at %I:%M %p UTC")
        delivery_res = send_task_reminder_email(
            student_name=student.name,
            student_email=student.email,
            task_title=task.title,
            due_date_str=due_date_str,
        )

        log_entry = EmailReminderLog(
            task_assignment_id=a.id,
            recipient_email=student.email,
            subject=delivery_res.get("subject", "Pending Task Reminder"),
            status=delivery_res.get("status", "mock_simulated"),
            error_message=delivery_res.get("error"),
            sent_at=now,
        )
        db.add(log_entry)

        # In-app notification for student
        notif = Notification(
            user_id=student.id,
            title="Overdue Task Reminder",
            message=f'Reminder: Your assigned task "{task.title}" is overdue (Deadline: {due_date_str}). Please submit your work.',
            notification_type="task_overdue",
            related_task_id=task.id,
            created_at=now,
        )
        db.add(notif)
        reminders_sent += 1
        delivery_logs.append({
            "student": student.name,
            "email": student.email,
            "task": task.title,
            "status": delivery_res.get("status"),
        })

    db.commit()
    return {
        "overdue_tasks_found": len(overdue_assignments),
        "reminders_dispatched": reminders_sent,
        "skipped_due_to_cooldown": reminders_skipped_cooldown,
        "logs": delivery_logs,
    }


def get_user_notifications(db: Session, user_id: int) -> List[Notification]:
    """Retrieves the recent in-app notifications for a user."""
    return (
        db.query(Notification)
        .filter(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
        .limit(20)
        .all()
    )


def mark_notification_as_read(db: Session, notification_id: int, user_id: int) -> bool:
    """Marks a single in-app notification as read."""
    notif = (
        db.query(Notification)
        .filter(Notification.id == notification_id, Notification.user_id == user_id)
        .first()
    )
    if notif:
        notif.is_read = True
        db.commit()
        return True
    return False


def mark_all_notifications_as_read(db: Session, user_id: int) -> int:
    """Marks all notifications for a user as read."""
    count = (
        db.query(Notification)
        .filter(Notification.user_id == user_id, Notification.is_read == False)
        .update({Notification.is_read: True})
    )
    db.commit()
    return count


def send_teacher_message(
    db: Session,
    teacher_id: int,
    title: str,
    message: str,
    student_id: Optional[int] = None,
    notification_type: str = "teacher_message"
) -> Dict[str, Any]:
    """
    Sends a direct message or general announcement from a teacher to one or all students.
    Creates in-app Notification records with teacher attribution.
    """
    now = utc_now()
    teacher = db.query(User).filter(User.id == teacher_id).first()
    teacher_name = teacher.name if teacher else "Teacher"

    if student_id:
        student = db.query(User).filter(User.id == student_id, User.role == "student").first()
        if not student:
            raise ValueError(f"Student with ID {student_id} not found.")
        targets = [student]
    else:
        targets = db.query(User).filter(User.role == "student").all()

    created_count = 0
    clean_title = (title or "").strip() or "Notice from Teacher"
    clean_message = (message or "").strip()

    for s in targets:
        notif = Notification(
            user_id=s.id,
            title=f"Message from {teacher_name}: {clean_title}",
            message=clean_message,
            notification_type=notification_type,
            created_at=now,
        )
        db.add(notif)
        created_count += 1

    db.commit()
    return {
        "success": True,
        "recipient_count": created_count,
        "message": f"Successfully sent message to {created_count} student{'s' if created_count != 1 else ''}."
    }


