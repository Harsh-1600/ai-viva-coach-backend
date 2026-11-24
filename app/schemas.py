"""Pydantic schemas for data validation and API request/response serialization."""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator


# ==============================================================================
# Health Check Schemas
# ==============================================================================

class HealthCheckResponse(BaseModel):
    """Schema for the GET /api/health endpoint response."""
    status: str = Field(default="healthy", description="Current health status of the service")
    service: str = Field(default="AI Viva & Interview Preparation Coach", description="Service name")
    version: str = Field(default="0.1.0", description="API version")
    database: str = Field(default="connected", description="Database connection status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of check")


# ==============================================================================
# Authentication & User Schemas
# ==============================================================================

class UserBase(BaseModel):
    """Base schema for user data."""
    name: str = Field(..., min_length=2, max_length=255, description="Candidate full name")
    email: EmailStr = Field(..., description="Unique email address")


class UserCreate(UserBase):
    """Schema for registering a new user."""
    password: str = Field(..., min_length=8, description="User password, minimum 8 characters")


class UserResponse(UserBase):
    """Schema for returning user data (without password hash)."""
    id: int
    created_at: datetime
    auth_provider: Optional[str] = "local"
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    learning_interests: Optional[str] = None
    learning_goals: Optional[str] = None
    role: Optional[str] = "student"
    username: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class GoogleLoginRequest(BaseModel):
    """Schema for Google Sign-In credential payload."""
    credential: str = Field(..., min_length=10, description="Google ID Token (JWT)")


class UserLogin(BaseModel):
    """Schema for authenticating a user."""
    email: EmailStr
    password: str
    role: Optional[str] = Field(None, description="Optional portal role constraint ('student' or 'teacher')")


class TeacherRegisterRequest(BaseModel):
    """Schema for registering an authorized teacher account."""
    name: str = Field(..., min_length=2, max_length=255, description="Teacher full name")
    email: EmailStr = Field(..., description="Unique educational or professional email address")
    password: str = Field(..., min_length=8, description="Password, minimum 8 characters")
    invitation_code: str = Field(..., min_length=1, description="Administrative teacher authorization code")


class Token(BaseModel):
    """Schema for JWT authentication token response."""
    access_token: str
    token_type: str = "bearer"
    user: Optional[UserResponse] = None


class TokenData(BaseModel):
    """Schema for parsed JWT token payload."""
    email: Optional[str] = None
    role: Optional[str] = None


class UserProfileUpdate(BaseModel):
    """Schema for updating user profile information."""
    name: Optional[str] = Field(None, min_length=2, max_length=255, description="Updated candidate name")
    avatar_url: Optional[str] = Field(None, max_length=1024, description="Avatar image URL or identifier")
    bio: Optional[str] = Field(None, max_length=1000, description="Short biography")
    learning_interests: Optional[str] = Field(None, max_length=1000, description="Comma-separated learning interests")
    learning_goals: Optional[str] = Field(None, max_length=5000, description="JSON string or text of learning goals")


class VivaStats(BaseModel):
    """Aggregated statistics for academic and oral viva sessions."""
    total_sessions: int = 0
    completed_sessions: int = 0
    in_progress_sessions: int = 0
    average_score: float = 0.0
    highest_score: float = 0.0
    questions_answered: int = 0
    top_topics: List[str] = Field(default_factory=list)


class InterviewStats(BaseModel):
    """Aggregated statistics for technical and mock interview sessions."""
    total_sessions: int = 0
    completed_sessions: int = 0
    in_progress_sessions: int = 0
    average_score: float = 0.0
    highest_score: float = 0.0
    questions_answered: int = 0
    top_topics: List[str] = Field(default_factory=list)


class QuickStats(BaseModel):
    """High-level summary metrics for candidate profile."""
    total_sessions: int = 0
    completed_sessions: int = 0
    in_progress_sessions: int = 0
    overall_average_score: float = 0.0
    total_questions_answered: int = 0
    completion_rate: float = 0.0


class RecentActivityItem(BaseModel):
    """Represents a discrete candidate activity item for the profile timeline."""
    id: str
    activity_type: str  # 'viva', 'interview', 'account', 'chat'
    title: str
    description: str
    status: Optional[str] = None
    score: Optional[float] = None
    questions_count: Optional[int] = None
    timestamp: datetime
    session_id: Optional[int] = None



class LearningGoalItem(BaseModel):
    """Schema for an individual candidate learning goal."""
    id: str
    title: str
    target_date: Optional[str] = None
    completed: bool = False
    created_at: Optional[str] = None


class AddGoalRequest(BaseModel):
    """Schema for adding a new learning goal."""
    title: str = Field(..., min_length=2, max_length=255, description="Goal description or milestone")
    target_date: Optional[str] = Field(None, max_length=100, description="Target timeline or date")


class UserProfileSummary(BaseModel):
    """Comprehensive user profile response including viva/interview metrics and activity feed."""
    user: UserResponse
    viva_stats: VivaStats
    interview_stats: InterviewStats
    quick_stats: QuickStats
    recent_activities: List[RecentActivityItem] = Field(default_factory=list)


class ChangePasswordRequest(BaseModel):
    """Schema for safely changing password."""
    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


class MessageResponse(BaseModel):
    """Generic status and message response."""
    message: str
    success: bool = True



# ==============================================================================
# InterviewSession Schemas
# ==============================================================================

class InterviewSessionBase(BaseModel):
    """Base schema for an interview session."""
    topic: str = Field(..., min_length=2, max_length=255, description="Subject domain/topic")
    subject: str = Field(..., min_length=2, max_length=255, description="Broad discipline")
    target_level: str = Field(..., min_length=2, max_length=100, description="Target experience/academic level")
    interview_mode: str = Field(default="academic_viva", max_length=100, description="Interview focus mode")


class InterviewSessionCreate(InterviewSessionBase):
    """Schema for creating a new interview session."""
    user_id: int = Field(..., description="ID of the candidate user")


class InterviewSessionUpdate(BaseModel):
    """Schema for updating an interview session."""
    status: Optional[str] = Field(None, description="Updated status: in_progress, completed, abandoned")
    completed_at: Optional[datetime] = Field(None, description="UTC completion timestamp")


class InterviewSessionResponse(InterviewSessionBase):
    """Schema for returning interview session data."""
    id: int
    user_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# Question Schemas
# ==============================================================================

class QuestionBase(BaseModel):
    """Base schema for an examination question."""
    question_text: str = Field(..., min_length=5, description="Examiner question prompt")
    question_type: str = Field(default="main", description="'main', 'follow_up', or 'clarification'")
    difficulty: Optional[str] = Field(default="medium", description="'easy', 'medium', or 'hard'")
    question_order: int = Field(default=1, ge=1, description="Question sequence index")


class QuestionCreate(QuestionBase):
    """Schema for adding a question to an interview session."""
    session_id: int = Field(..., description="Target interview session ID")


class QuestionResponse(QuestionBase):
    """Schema for returning question data."""
    id: int
    session_id: int

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# AI Question Generation Schemas
# ==============================================================================

class GeneratedQuestion(BaseModel):
    """Schema for an individual AI-generated question."""
    question_text: str = Field(..., min_length=5, description="The examiner question prompt")
    question_type: str = Field(default="conceptual", description="Type of question (conceptual, mechanism, trade_off, edge_case, problem_solving, main)")
    difficulty: str = Field(default="medium", description="Question difficulty: easy, medium, hard")
    topic: str = Field(..., description="Target subtopic of this question")
    evaluation_criteria: Optional[List[str]] = Field(default=None, description="Expected points, nuances, or criteria for evaluation")
    question_order: int = Field(default=1, ge=1, description="Order/sequence index of the question")


class GenerateQuestionsRequest(BaseModel):
    """Schema for requesting AI question generation."""
    subject: str = Field(..., min_length=2, max_length=255, description="Subject domain (e.g. Operating Systems, Data Structures)")
    topic: str = Field(..., min_length=2, max_length=255, description="Specific topic (e.g. Virtual Memory, Deadlocks)")
    target_level: str = Field(default="Intermediate", description="Candidate level: Beginner, Intermediate, or Advanced")
    interview_type: str = Field(default="Academic Viva", description="Type: Academic Viva, Technical Interview, or General Interview")
    num_questions: int = Field(default=4, ge=1, le=10, description="Number of questions to generate (1-10)")
    session_duration: int = Field(default=30, ge=5, le=180, description="Target session duration in minutes")
    difficulty: str = Field(default="Medium", description="Preferred difficulty: Easy, Medium, Hard, Adaptive")
    custom_instructions: Optional[str] = Field(None, max_length=1000, description="Optional custom instructions for examiner")
    save_to_database: bool = Field(default=True, description="Whether to automatically create session & save questions to database")
    session_id: Optional[int] = Field(None, description="Optional existing session ID to attach questions to")


class GenerateQuestionsResponse(BaseModel):
    """Response schema for generated interview questions."""
    success: bool = True
    session_id: Optional[int] = None
    subject: str
    topic: str
    target_level: str
    interview_type: str
    total_questions: int
    questions: List[GeneratedQuestion]


# ==============================================================================
# Answer Schemas (Database Models)
# ==============================================================================

class AnswerBase(BaseModel):
    """Base schema for candidate answer submission and evaluation."""
    answer_text: str = Field(default="", description="Candidate response text")
    score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Question score (0 - 100)")
    correctness_status: Optional[str] = Field(None, description="Status: correct | partially_correct | incorrect | empty")
    feedback: Optional[str] = Field(None, description="Examiner feedback")
    evaluation_feedback: Optional[str] = Field(None, description="Detailed evaluation feedback")
    strengths: Optional[str] = Field(None, description="Key answer strengths")
    improvement_areas: Optional[str] = Field(None, description="Areas for improvement")
    correct_concepts: Optional[Union[List[str], str]] = Field(default_factory=list, description="Correct concepts identified")
    incorrect_concepts: Optional[Union[List[str], str]] = Field(default_factory=list, description="Incorrect concepts identified")
    missing_concepts: Optional[Union[List[str], str]] = Field(default_factory=list, description="Missing concepts")
    correct_answer: Optional[str] = Field(None, description="Authoritative generated model answer")
    simple_explanation: Optional[str] = Field(None, description="Intuitive simple explanation")
    detailed_explanation: Optional[str] = Field(None, description="Deep technical explanation")
    example: Optional[str] = Field(None, description="Practical example or code snippet")
    key_points_to_remember: Optional[Union[List[str], str]] = Field(default_factory=list, description="Key points to remember")
    improved_answer: Optional[str] = Field(None, description="Improved viva answer demonstrating how to articulate for full marks")
    model_answer_provider: Optional[str] = Field(default="Google Gemini", description="AI provider that generated the model answer")
    model_answer_error: Optional[str] = Field(None, description="Clear indication if model answer generation failed")
    model_answer_disclaimer: Optional[str] = Field(
        default="AI-generated model answer for revision purposes; verify with curriculum and examiner standards.",
        description="Disclaimer indicating answer is an AI estimate"
    )


class AnswerCreate(AnswerBase):
    """Schema for creating/evaluating a candidate answer."""
    question_id: int = Field(..., description="ID of the question being answered")
    session_id: Optional[int] = Field(None, description="ID of the interview session")
    evaluation_details: Optional[str] = Field(None, description="Serialized JSON of detailed 100-point rubric evaluation")


class AnswerResponse(AnswerBase):
    """Schema for returning evaluated answer data."""
    id: int
    question_id: int
    session_id: Optional[int] = None
    evaluation_details: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ==============================================================================
# SessionReport Schemas
# ==============================================================================

class SessionReportBase(BaseModel):
    """Base schema for a comprehensive final session report."""
    overall_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Overall performance score (0 - 100)")
    questions_answered: int = Field(default=0, ge=0, description="Number of questions answered")
    average_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Average answer score (0 - 100)")
    summary: Optional[str] = Field(None, description="Executive summary of performance")
    strengths: Optional[str] = Field(None, description="Demonstrated proficiencies")
    improvement_areas: Optional[str] = Field(None, description="Critical knowledge gaps")
    recommendations: Optional[str] = Field(None, description="Personalized revision study plan")
    strongest_topics: Optional[str] = Field(None, description="JSON or text of strongest topics")
    weakest_topics: Optional[str] = Field(None, description="JSON or text of weakest topics")
    technical_assessment: Optional[str] = Field(None, description="Technical knowledge assessment")
    communication_feedback: Optional[str] = Field(None, description="Communication and clarity feedback")
    repeated_mistakes: Optional[str] = Field(None, description="Repeated misconceptions and mistakes")
    suggested_next_topics: Optional[str] = Field(None, description="Suggested next practice topics")
    report_details: Optional[str] = Field(None, description="Serialized JSON payload containing chart metrics")


class SessionReportCreate(SessionReportBase):
    """Schema for generating a new session report."""
    session_id: int = Field(..., description="Interview session ID")


class SessionReportResponse(SessionReportBase):
    """Schema for returning session report data."""
    id: int
    session_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ScoreByQuestionItem(BaseModel):
    """Visual metric item representing performance on a single question."""
    question_id: int
    question_order: int
    question_text: str
    topic: str
    difficulty: str
    score: Optional[float] = None
    correctness_status: Optional[str] = None
    is_answered: bool = False
    feedback: Optional[str] = None
    evaluation_feedback: Optional[str] = None
    candidate_answer: Optional[str] = None
    correct_concepts: List[str] = Field(default_factory=list)
    incorrect_concepts: List[str] = Field(default_factory=list)
    missing_concepts: List[str] = Field(default_factory=list)
    suggested_ideal_outline: List[str] = Field(default_factory=list)
    correct_answer: Optional[str] = None
    simple_explanation: Optional[str] = None
    detailed_explanation: Optional[str] = None
    example: Optional[str] = None
    key_points_to_remember: List[str] = Field(default_factory=list)
    improved_answer: Optional[str] = None
    model_answer_provider: Optional[str] = Field(default="Google Gemini")
    model_answer_error: Optional[str] = None
    model_answer_disclaimer: Optional[str] = Field(
        default="AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
    )


class TopicPerformanceItem(BaseModel):
    """Visual metric item showing proficiency per subtopic."""
    topic: str
    average_score: float
    question_count: int


class RubricAverages(BaseModel):
    """Aggregated averages across 100-point rubric dimensions."""
    technical_correctness: float = 0.0
    conceptual_understanding: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    relevance: float = 0.0


class DetailedSessionReportResponse(BaseModel):
    """Comprehensive performance report returned by reporting service."""
    id: int
    session_id: int
    subject: str
    topic: str
    target_level: str
    interview_mode: str
    session_status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    overall_score: float
    questions_answered: int
    total_questions: int
    average_score: float
    summary: str
    strengths: Optional[str] = None
    improvement_areas: Optional[str] = None
    recommendations: Optional[str] = None
    strongest_topics: List[str] = Field(default_factory=list)
    weakest_topics: List[str] = Field(default_factory=list)
    technical_assessment: str
    communication_feedback: str
    repeated_mistakes: List[str] = Field(default_factory=list)
    suggested_next_topics: List[str] = Field(default_factory=list)
    score_by_question: List[ScoreByQuestionItem] = Field(default_factory=list)
    performance_by_topic: List[TopicPerformanceItem] = Field(default_factory=list)
    rubric_breakdown_averages: Optional[RubricAverages] = None
    created_at: datetime


class SessionHistoryItem(BaseModel):
    """Summary of a past session for history tables."""
    session_id: int
    subject: str
    topic: str
    target_level: str
    interview_mode: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    questions_answered: int
    total_questions: int
    overall_score: Optional[float] = None
    report_id: Optional[int] = None


class SessionProgressPoint(BaseModel):
    """Data point for tracking progress across sessions over time."""
    session_id: int
    topic: str
    date: str
    score: float


class PerformanceStatisticsResponse(BaseModel):
    """Aggregated candidate performance metrics across all completed sessions."""
    total_sessions: int = 0
    completed_sessions: int = 0
    total_questions_answered: int = 0
    cumulative_average_score: Optional[float] = None
    progress_timeline: List[SessionProgressPoint] = Field(default_factory=list)
    strongest_topics: List[str] = Field(default_factory=list)
    weakest_topics: List[str] = Field(default_factory=list)
    rubric_profile: Optional[RubricAverages] = None


# ==============================================================================
# Interactive Interview Experience & 100-Point Evaluation Schemas
# ==============================================================================

class StartSessionRequest(BaseModel):
    """Parameters to start an interactive interview session."""
    subject: str = Field(..., min_length=2, max_length=255, description="Subject domain (e.g. Operating Systems)")
    topic: str = Field(..., min_length=2, max_length=255, description="Specific topic (e.g. Virtual Memory)")
    target_level: str = Field(default="Intermediate", description="Target level: Beginner, Intermediate, or Advanced")
    interview_type: str = Field(default="Academic Viva", description="Format: Academic Viva, Technical Interview, or General Interview")
    num_questions: int = Field(default=4, ge=1, le=10, description="Number of primary questions")
    session_duration: int = Field(default=30, ge=5, le=180, description="Session duration in minutes")
    difficulty: str = Field(default="Medium", description="Preferred difficulty: Easy, Medium, Hard, Adaptive")
    custom_instructions: Optional[str] = Field(None, max_length=1000, description="Optional custom examiner instructions")


class CurrentQuestionResponse(BaseModel):
    """Response containing the active question and session progress."""
    session_id: int
    is_completed: bool = False
    current_question: Optional[QuestionResponse] = None
    question_number: int = 0
    total_questions: int = 0
    progress_percent: float = 0.0
    session_status: str = "in_progress"


class SubmitAnswerRequest(BaseModel):
    """Request payload for submitting a candidate answer."""
    question_id: int = Field(..., description="ID of question being answered")
    answer_text: str = Field(default="", max_length=10000, description="Candidate response text")


class RubricBreakdown(BaseModel):
    """Component scoring breakdown based on 100-point rubric."""
    technical_correctness: float = Field(default=0.0, ge=0.0, le=30.0, description="Technical correctness (max 30)")
    conceptual_understanding: float = Field(default=0.0, ge=0.0, le=25.0, description="Conceptual understanding (max 25)")
    completeness: float = Field(default=0.0, ge=0.0, le=20.0, description="Completeness and depth (max 20)")
    clarity: float = Field(default=0.0, ge=0.0, le=15.0, description="Clarity of communication (max 15)")
    relevance: float = Field(default=0.0, ge=0.0, le=10.0, description="Relevance to prompt (max 10)")


class DetailedEvaluation(BaseModel):
    """Complete 100-point rubric evaluation output including detailed model answer and explanation."""
    question_id: Optional[int] = Field(default=None, description="ID of question evaluated")
    numerical_score: float = Field(..., ge=0.0, le=100.0, description="Numerical score out of 100")
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Alias for numerical_score")
    correctness_status: str = Field(default="incorrect", description="correct | partially_correct | incorrect | empty")
    brief_explanation: str = Field(..., description="Examiner summary explanation")
    evaluation_feedback: Optional[str] = Field(default=None, description="Detailed feedback on candidate answer")
    is_correct: Optional[bool] = Field(default=None, description="Whether candidate's response was factually correct")
    correct_concepts: List[str] = Field(default_factory=list, description="Correct concepts identified")
    correct_concepts_identified: List[str] = Field(default_factory=list, description="Alias for correct_concepts")
    incorrect_concepts: List[str] = Field(default_factory=list, description="Incorrect concepts or technical inaccuracies")
    technical_inaccuracies: List[str] = Field(default_factory=list, description="Alias for incorrect_concepts")
    missing_concepts: List[str] = Field(default_factory=list, description="Essential points omitted by candidate")
    correct_answer: str = Field(default="", description="Technically accurate model answer")
    simple_explanation: str = Field(default="", description="Explanation in simple, understandable language")
    detailed_explanation: str = Field(default="", description="In-depth technical explanation of concepts")
    example: Optional[str] = Field(default=None, description="Practical real-world or code example")
    key_points_to_remember: List[str] = Field(default_factory=list, description="Key takeaways for viva/interview revision")
    improved_answer: Optional[str] = Field(default=None, description="Improved viva answer demonstrating how to articulate for full marks")
    candidate_answer: Optional[str] = Field(default=None, description="The candidate's submitted answer text")
    specific_improvement_suggestions: List[str] = Field(default_factory=list, description="Targeted actionable suggestions")
    suggested_ideal_outline: List[str] = Field(default_factory=list, description="Suggested ideal answer outline")
    next_difficulty_recommendation: str = Field(default="same", description="'easier', 'same', or 'harder'")
    rubric_breakdown: Optional[RubricBreakdown] = None
    should_follow_up: bool = Field(default=False, description="Whether adaptive follow-up is recommended")
    follow_up_reason: Optional[str] = None
    follow_up_question: Optional[dict] = None
    is_ai_estimate: bool = Field(default=True, description="Explicit disclaimer flag indicating score is an AI estimate")
    model_answer_provider: Optional[str] = Field(default="Google Gemini", description="AI provider that generated the model answer")
    model_answer_error: Optional[str] = Field(default=None, description="Clear indication if model answer generation failed")
    model_answer_disclaimer: Optional[str] = Field(
        default="AI-generated model answer for revision purposes; verify with curriculum and examiner standards.",
        description="Disclaimer indicating answer is an AI estimate"
    )


class AnswerEvaluationResult(BaseModel):
    """Structured evaluation and model answer returned by AI examiner."""
    question_id: Optional[int] = Field(default=None, description="ID of question being answered")
    score: float = Field(..., ge=0.0, le=100.0, description="Numerical score out of 100")
    correctness_status: str = Field(default="incorrect", description="correct | partially_correct | incorrect | empty")
    evaluation_feedback: str = Field(..., description="Examiner evaluation feedback summary")
    correct_concepts: List[str] = Field(default_factory=list, description="Correct concepts identified")
    incorrect_concepts: List[str] = Field(default_factory=list, description="Technical inaccuracies or incorrect concepts")
    missing_concepts: List[str] = Field(default_factory=list, description="Missing concepts")
    correct_answer: str = Field(..., description="Technically accurate model answer")
    simple_explanation: str = Field(..., description="Explanation in simple, understandable language")
    detailed_explanation: str = Field(..., description="In-depth technical explanation")
    example: Optional[str] = Field(default=None, description="Practical example or scenario")
    key_points_to_remember: List[str] = Field(default_factory=list, description="Key points to remember")
    improved_answer: Optional[str] = Field(default=None, description="Improved viva answer demonstrating how to articulate for full marks")

    # Additional fields preserving backward compatibility:
    feedback: Optional[str] = Field(None, description="Examiner feedback summary (alias for evaluation_feedback)")
    candidate_answer: Optional[str] = Field(None, description="The candidate's submitted answer text")
    is_correct: Optional[bool] = Field(default=None, description="Whether answer was factually correct")
    strengths: Optional[str] = Field(None, description="Demonstrated strengths")
    improvement_areas: Optional[str] = Field(None, description="Gaps and improvement areas")
    technical_inaccuracies: List[str] = Field(default_factory=list, description="Alias for incorrect_concepts")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Specific suggestions")
    suggested_ideal_outline: List[str] = Field(default_factory=list, description="Outline of ideal answer")
    next_difficulty: str = Field(default="same", description="Recommended next question difficulty")
    rubric_breakdown: Optional[RubricBreakdown] = None
    should_follow_up: bool = Field(default=False, description="Whether adaptive follow-up is triggered")
    follow_up_reason: Optional[str] = None
    is_ai_estimate: bool = Field(default=True, description="Clearly labeled as an AI-generated estimate")
    model_answer_provider: Optional[str] = Field(default="Google Gemini", description="AI provider that generated the model answer")
    model_answer_error: Optional[str] = Field(default=None, description="Clear indication if model answer generation failed")
    model_answer_disclaimer: Optional[str] = Field(
        default="AI-generated model answer for revision purposes; verify with curriculum and examiner standards.",
        description="Disclaimer indicating answer is an AI estimate"
    )


class EvaluateAnswerStandaloneRequest(BaseModel):
    """Request schema for direct standalone answer evaluation."""
    question_text: str = Field(..., min_length=5, description="The interview question")
    topic: str = Field(..., min_length=2, description="The subject or topic area")
    difficulty: str = Field(default="Medium", description="Question difficulty")
    target_level: str = Field(default="Intermediate", description="Candidate target level")
    interview_type: str = Field(default="Academic Viva", description="Interview format")
    candidate_answer: str = Field(..., min_length=1, max_length=10000, description="The candidate's response")


class AnswerSubmissionResponse(BaseModel):
    """Response after submitting an answer, including evaluation and next question/follow-up."""
    success: bool = True
    session_id: int
    question_id: int
    evaluation: AnswerEvaluationResult
    has_follow_up: bool = False
    follow_up_question: Optional[QuestionResponse] = None
    next_question: Optional[QuestionResponse] = None
    is_session_completed: bool = False
    total_answered: int
    total_questions: int


class EndSessionResponse(BaseModel):
    """Summary response when an interview session is ended."""
    session_id: int
    status: str
    completed_at: datetime
    questions_answered: int
    total_questions: int
    average_score: Optional[float] = None
    summary: Optional[str] = None


# ==============================================================================
# Gemini Student Doubt-Clearing Assistant Schemas
# ==============================================================================

class DoubtChatMessage(BaseModel):
    """A single turn in the doubt-clearing conversation history."""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'model'")
    content: str = Field(..., min_length=1, max_length=10000, description="Message content text")


class AskDoubtRequest(BaseModel):
    """Request payload for student asking a doubt or clarification to Gemini AI."""
    message: str = Field(..., min_length=1, max_length=5000, description="Student's question or doubt")
    history: List[DoubtChatMessage] = Field(default_factory=list, description="Previous conversation turns for context")
    context: Optional[str] = Field(None, max_length=2000, description="Optional academic topic or interview question context")


class AskDoubtResponse(BaseModel):
    """Response returned by the Gemini AI doubt-clearing assistant."""
    success: bool = True
    answer: str = Field(..., description="Academic tutor explanation and guidance")
    model: str = Field(default="gemini-1.5-flash", description="Gemini model identifier used")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC response generation timestamp")


class ExplainQuestionRequest(BaseModel):
    """Request payload for asking Gemini for more explanation regarding an evaluated viva question."""
    question_text: str = Field(..., min_length=3, description="Original question prompt")
    topic: Optional[str] = Field(None, description="Topic or subject domain")
    candidate_answer: Optional[str] = Field(None, description="Student's submitted response")
    correct_answer: Optional[str] = Field(None, description="Model answer provided")
    doubt_query: Optional[str] = Field(None, max_length=5000, description="Specific follow-up question or clarification")
    history: List[DoubtChatMessage] = Field(default_factory=list, description="Follow-up conversation history")


# ==============================================================================
# Teacher Management & Task System Schemas
# ==============================================================================

class TaskCreateRequest(BaseModel):
    """Schema for creating and assigning a new task."""
    title: str = Field(..., min_length=3, max_length=255, description="Task headline or assignment title")
    description: Optional[str] = Field(None, description="Detailed assignment prompt and background")
    subject: str = Field(..., min_length=2, max_length=100, description="Course discipline (e.g. Operating Systems)")
    topic: Optional[str] = Field(None, max_length=255, description="Specific topic or unit (e.g. Virtual Memory)")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, urgent")
    instructions: Optional[str] = Field(None, description="Submission guidelines and formatting instructions")
    due_date: datetime = Field(..., description="UTC deadline timestamp")
    assigned_student_ids: List[int] = Field(..., min_length=1, description="List of student user IDs assigned this task")


class TaskUpdateRequest(BaseModel):
    """Schema for updating an existing task before completion."""
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    description: Optional[str] = None
    subject: Optional[str] = Field(None, min_length=2, max_length=100)
    topic: Optional[str] = Field(None, max_length=255)
    priority: Optional[str] = None
    instructions: Optional[str] = None
    due_date: Optional[datetime] = None


class TaskAssignmentResponse(BaseModel):
    """Schema for a single student's task assignment status."""
    id: int
    task_id: int
    student_id: int
    student_name: str
    student_email: str
    status: str
    submission_text: Optional[str] = None
    submission_file_url: Optional[str] = None
    submission_date: Optional[datetime] = None
    teacher_feedback: Optional[str] = None
    grade_score: Optional[float] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TaskResponse(BaseModel):
    """Schema for teacher viewing a created task with completion metrics."""
    id: int
    teacher_id: int
    teacher_name: str
    title: str
    description: Optional[str] = None
    subject: str
    topic: Optional[str] = None
    priority: str
    instructions: Optional[str] = None
    assigned_date: datetime
    due_date: datetime
    created_at: datetime
    updated_at: datetime
    total_assigned: int = 0
    completed_count: int = 0
    submitted_count: int = 0
    pending_count: int = 0
    is_overdue: bool = False

    model_config = ConfigDict(from_attributes=True)


class TaskDetailResponse(BaseModel):
    """Schema for detailed task view including all assigned student records."""
    task: TaskResponse
    assignments: List[TaskAssignmentResponse] = Field(default_factory=list)


class StudentTaskItemResponse(BaseModel):
    """Schema for student dashboard task list items."""
    assignment_id: int
    task_id: int
    title: str
    description: Optional[str] = None
    subject: str
    topic: Optional[str] = None
    priority: str
    instructions: Optional[str] = None
    assigned_date: datetime
    due_date: datetime
    status: str
    submission_text: Optional[str] = None
    submission_file_url: Optional[str] = None
    submission_date: Optional[datetime] = None
    teacher_feedback: Optional[str] = None
    grade_score: Optional[float] = None
    completed_at: Optional[datetime] = None
    teacher_name: str
    is_overdue: bool = False


class TaskSubmissionRequest(BaseModel):
    """Payload schema for submitting task response text."""
    submission_text: Optional[str] = Field(None, max_length=10000, description="Student's written submission")


class TaskReviewRequest(BaseModel):
    """Payload schema for teacher grading and reviewing a submission."""
    teacher_feedback: Optional[str] = Field(None, description="Constructive evaluation notes and feedback")
    grade_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Grade score (0 to 100)")
    status: str = Field(default="completed", description="Updated status: 'completed' or 'in_progress'")


class StudentSummaryItem(BaseModel):
    """Schema for student row item in teacher dashboard."""
    id: int
    name: str
    email: str
    avatar_url: Optional[str] = None
    registered_at: datetime
    total_viva_sessions: int = 0
    total_interview_sessions: int = 0
    average_performance_score: Optional[float] = None
    completed_tasks: int = 0
    pending_tasks: int = 0

    model_config = ConfigDict(from_attributes=True)


class TeacherDashboardStatsResponse(BaseModel):
    """Schema for high-level teacher dashboard metrics."""
    total_students: int
    active_students: int
    total_assigned_tasks: int
    completed_tasks: int
    pending_tasks: int
    overdue_tasks: int


class StudentDetailPerformanceResponse(BaseModel):
    """Schema for comprehensive student performance deep dive."""
    profile: UserResponse
    viva_stats: Dict[str, Any]
    interview_stats: Dict[str, Any]
    overall_average_score: Optional[float] = None
    subject_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    recent_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    completed_tasks_count: int = 0
    pending_tasks_count: int = 0
    previous_reports: List[Dict[str, Any]] = Field(default_factory=list)


class NotificationResponse(BaseModel):
    """Schema for user in-app notification alert."""
    id: int
    title: str
    message: str
    notification_type: str
    related_task_id: Optional[int] = None
    is_read: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TeacherSendMessageRequest(BaseModel):
    """Payload schema for teacher sending an in-app message or announcement to students."""
    student_id: Optional[int] = Field(None, description="Recipient student ID (None to broadcast to all enrolled students)")
    title: Optional[str] = Field(default="Teacher Notice", max_length=255, description="Message title or subject")
    message: str = Field(..., min_length=1, max_length=5000, description="Message body / announcement content")
    notification_type: str = Field(default="teacher_message", description="Category: 'teacher_message' or 'announcement'")

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, v: Optional[str]) -> str:
        if not v or not str(v).strip():
            return "Teacher Notice"
        return str(v).strip()


class TeacherSendMessageResponse(BaseModel):
    """Response schema for teacher message dispatch result."""
    success: bool
    recipient_count: int
    message: str




