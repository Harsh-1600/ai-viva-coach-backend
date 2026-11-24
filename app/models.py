"""SQLAlchemy ORM models for User, InterviewSession, Question, Answer, and SessionReport."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Boolean
)
from sqlalchemy.orm import relationship
from app.database import Base


def utc_now() -> datetime:
    """Helper to return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """
    User model representing registered candidates and students.
    Fields:
        id: Primary key
        name: Full candidate name
        email: Unique email address used for login
        hashed_password: Secure bcrypt password hash (or unusable OAuth sentinel)
        auth_provider: Authentication provider ('local' or 'google')
        google_id: Unique Google user subject identifier (sub)
        avatar_url: Profile picture URL from OAuth provider
        created_at: UTC registration timestamp
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    auth_provider = Column(String(50), nullable=False, default="local")
    google_id = Column(String(255), unique=True, index=True, nullable=True)
    avatar_url = Column(String(1024), nullable=True)
    bio = Column(Text, nullable=True)
    learning_interests = Column(Text, nullable=True)
    learning_goals = Column(Text, nullable=True)
    role = Column(String(50), nullable=False, default="student", index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    sessions = relationship(
        "InterviewSession",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(InterviewSession.started_at)"
    )

    @property
    def username(self) -> str:
        return self.email.split("@")[0] if self.email else "candidate"

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', name='{self.name}')>"


class InterviewSession(Base):
    """
    InterviewSession model representing a mock viva or technical interview session.
    Fields:
        id: Primary key
        user_id: Foreign key to User
        topic: Specific subject domain (e.g., 'Virtual Memory & Paging')
        subject: Broad academic/technical discipline (e.g., 'Operating Systems')
        target_level: Difficulty/seniority level (e.g., 'Undergraduate', 'Senior SDE')
        interview_mode: Focus mode (e.g., 'academic_viva', 'technical_interview')
        status: Current session state ('in_progress', 'completed', 'abandoned')
        started_at: UTC timestamp when the session began
        completed_at: UTC timestamp when the session finished
    """
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    topic = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    target_level = Column(String(100), nullable=False)
    interview_mode = Column(String(100), nullable=False, default="academic_viva")
    status = Column(String(50), nullable=False, default="in_progress", index=True)
    started_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")
    questions = relationship(
        "Question",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Question.question_order"
    )
    report = relationship(
        "SessionReport",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<InterviewSession(id={self.id}, user_id={self.user_id}, topic='{self.topic}', status='{self.status}')>"


class Question(Base):
    """
    Question model representing each primary or adaptive follow-up prompt.
    Fields:
        id: Primary key
        session_id: Foreign key to InterviewSession
        question_text: The spoken or written question from the examiner
        question_type: Category ('main', 'follow_up', 'clarification')
        difficulty: Difficulty rating ('easy', 'medium', 'hard')
        question_order: Positional sequence index in the session (1-indexed)
    """
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False, default="main")
    difficulty = Column(String(50), nullable=True, default="medium")
    question_order = Column(Integer, nullable=False, default=1)

    # Relationships
    session = relationship("InterviewSession", back_populates="questions")
    answer = relationship(
        "Answer",
        back_populates="question",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Question(id={self.id}, session_id={self.session_id}, type='{self.question_type}', order={self.question_order})>"


class Answer(Base):
    """
    Answer model representing candidate's verbal or written response and evaluation.
    Fields:
        id: Primary key
        question_id: Foreign key to Question (1:1 relationship)
        answer_text: The candidate's articulated answer
        score: Evaluated score for this response (e.g., 0.0 - 10.0)
        feedback: Constructive examiner feedback snippet
        strengths: Key positive points demonstrated in response
        improvement_areas: Gaps, missed nuances, or misconceptions
        created_at: UTC timestamp when answer was submitted
    """
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    answer_text = Column(Text, nullable=False)
    score = Column(Float, nullable=True)
    correctness_status = Column(String(50), nullable=True)
    feedback = Column(Text, nullable=True)
    evaluation_feedback = Column(Text, nullable=True)
    strengths = Column(Text, nullable=True)
    improvement_areas = Column(Text, nullable=True)
    correct_concepts = Column(Text, nullable=True)
    incorrect_concepts = Column(Text, nullable=True)
    missing_concepts = Column(Text, nullable=True)
    correct_answer = Column(Text, nullable=True)
    simple_explanation = Column(Text, nullable=True)
    detailed_explanation = Column(Text, nullable=True)
    example = Column(Text, nullable=True)
    key_points_to_remember = Column(Text, nullable=True)
    improved_answer = Column(Text, nullable=True)
    evaluation_details = Column(Text, nullable=True)
    model_answer_provider = Column(String(50), nullable=True, default="Google Gemini")
    model_answer_error = Column(Text, nullable=True)
    model_answer_disclaimer = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    question = relationship("Question", back_populates="answer")
    session = relationship("InterviewSession")

    def __repr__(self) -> str:
        return f"<Answer(id={self.id}, question_id={self.question_id}, score={self.score}, status='{self.correctness_status}')>"


class SessionReport(Base):
    """
    SessionReport model representing the comprehensive final evaluation report.
    Fields:
        id: Primary key
        session_id: Foreign key to InterviewSession (1:1 relationship)
        overall_score: Cumulative score (0.0 - 100.0)
        summary: High-level qualitative summary of performance
        strengths: Key strengths and demonstrated proficiencies
        improvement_areas: Critical knowledge gaps and weaknesses
        recommendations: Actionable study plan and revision recommendations
        created_at: UTC timestamp when report was compiled
    """
    __tablename__ = "session_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    overall_score = Column(Float, nullable=True)
    questions_answered = Column(Integer, default=0, nullable=False)
    average_score = Column(Float, nullable=True)
    summary = Column(Text, nullable=True)
    strengths = Column(Text, nullable=True)
    improvement_areas = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    strongest_topics = Column(Text, nullable=True)
    weakest_topics = Column(Text, nullable=True)
    technical_assessment = Column(Text, nullable=True)
    communication_feedback = Column(Text, nullable=True)
    repeated_mistakes = Column(Text, nullable=True)
    suggested_next_topics = Column(Text, nullable=True)
    report_details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    session = relationship("InterviewSession", back_populates="report")

    def __repr__(self) -> str:
        return f"<SessionReport(id={self.id}, session_id={self.session_id}, overall_score={self.overall_score})>"


class Task(Base):
    """
    Task model representing an assignment created by an authorized teacher.
    Fields:
        id: Unique task identifier
        teacher_id: Foreign key to User (the educator creator)
        title: Short descriptive title of the task
        description: Detailed instructions and background
        subject: Course or academic discipline
        topic: Specific topic or module
        priority: Priority level ('low', 'medium', 'high', 'urgent')
        instructions: Specific preparation or submission guidance
        assigned_date: Timestamp when the task was issued
        due_date: Deadline timestamp
        created_at: Creation timestamp
        updated_at: Last modification timestamp
    """
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    teacher_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(100), nullable=False)
    topic = Column(String(255), nullable=True)
    priority = Column(String(50), nullable=False, default="medium")
    instructions = Column(Text, nullable=True)
    assigned_date = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    teacher = relationship("User", foreign_keys=[teacher_id])
    assignments = relationship("TaskAssignment", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title='{self.title}', subject='{self.subject}', priority='{self.priority}')>"


class TaskAssignment(Base):
    """
    TaskAssignment model linking a Task to a specific enrolled Student.
    Fields:
        id: Primary key
        task_id: Foreign key to Task
        student_id: Foreign key to User (the student)
        status: Current workflow status ('assigned', 'in_progress', 'submitted', 'completed', 'overdue')
        submission_text: Written student submission response
        submission_file_url: Relative URL of uploaded student submission attachment
        submission_date: Timestamp of submission
        teacher_feedback: Constructive evaluation remarks from the teacher
        grade_score: Numerical grade/score (e.g. 0.0 - 100.0)
        completed_at: Timestamp when teacher marked task completed
        created_at: Creation timestamp
    """
    __tablename__ = "task_assignments"
    __table_args__ = (UniqueConstraint("task_id", "student_id", name="uq_task_student"),)

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="assigned", index=True)
    submission_text = Column(Text, nullable=True)
    submission_file_url = Column(String(1024), nullable=True)
    submission_date = Column(DateTime(timezone=True), nullable=True)
    teacher_feedback = Column(Text, nullable=True)
    grade_score = Column(Float, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    task = relationship("Task", back_populates="assignments")
    student = relationship("User", foreign_keys=[student_id])
    reminder_logs = relationship("EmailReminderLog", back_populates="assignment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<TaskAssignment(id={self.id}, task_id={self.task_id}, student_id={self.student_id}, status='{self.status}')>"


class Notification(Base):
    """
    Notification model representing in-app alerts for students and teachers.
    Fields:
        id: Primary key
        user_id: Foreign key to User (recipient)
        title: Notification header
        message: Detailed alert text
        notification_type: Category ('task_assigned', 'deadline_approaching', 'task_overdue', 'teacher_feedback', 'task_completed', 'task_submitted')
        related_task_id: Optional foreign key to Task
        is_read: Boolean read/unread state
        created_at: Timestamp
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False, default="task_assigned", index=True)
    related_task_id = Column(Integer, ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True)
    is_read = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    user = relationship("User")
    task = relationship("Task")

    def __repr__(self) -> str:
        return f"<Notification(id={self.id}, user_id={self.user_id}, type='{self.notification_type}', is_read={self.is_read})>"


class EmailReminderLog(Base):
    """
    EmailReminderLog model recording automatic reminder history and duplicate suppression.
    Fields:
        id: Primary key
        task_assignment_id: Foreign key to TaskAssignment
        recipient_email: Recipient student email
        subject: Email subject line
        status: Delivery status ('sent', 'mock_simulated', 'failed')
        error_message: Optional error snippet if delivery failed
        sent_at: Timestamp
    """
    __tablename__ = "email_reminder_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    task_assignment_id = Column(Integer, ForeignKey("task_assignments.id", ondelete="CASCADE"), nullable=False, index=True)
    recipient_email = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="sent", index=True)
    error_message = Column(Text, nullable=True)
    sent_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    assignment = relationship("TaskAssignment", back_populates="reminder_logs")

    def __repr__(self) -> str:
        return f"<EmailReminderLog(id={self.id}, assignment_id={self.task_assignment_id}, status='{self.status}')>"

