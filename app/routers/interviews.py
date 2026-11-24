"""Interview session, adaptive AI evaluation, and question progression router."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Question, Answer
from app.schemas import (
    StartSessionRequest,
    CurrentQuestionResponse,
    SubmitAnswerRequest,
    AnswerEvaluationResult,
    AnswerSubmissionResponse,
    EndSessionResponse,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    GeneratedQuestion,
    InterviewSessionCreate,
    InterviewSessionResponse,
    QuestionCreate,
    QuestionResponse,
    AnswerCreate,
    AnswerResponse,
    DetailedEvaluation,
    EvaluateAnswerStandaloneRequest,
    RubricBreakdown,
)
from app.security import get_current_user
from app.services import db_service, ai_service, evaluation_service
from app.services.rate_limiter import check_ai_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interviews", tags=["Interviews & AI Viva Experience"])


# ==============================================================================
# 0. Generate Questions (Phase 4 Endpoint)
# ==============================================================================

@router.post(
    "/generate-questions",
    response_model=GenerateQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate interview questions using Anthropic Claude"
)
def generate_questions(
    request: GenerateQuestionsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(check_ai_rate_limit)
):
    """
    Generate structured, technically rigorous viva or interview questions using Claude.
    - Protected: requires Bearer authentication token.
    - Validates inputs (subject, topic, level, question count, difficulty, duration).
    - If save_to_database is true (default), automatically initializes an InterviewSession
      and stores the generated questions.
    """
    try:
        generated_questions = ai_service.generate_interview_questions(request)
    except ai_service.AIConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message)
    except ai_service.AIRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.message)
    except ai_service.AIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI service error: {str(exc)}")

    session_id = request.session_id
    if request.save_to_database:
        if session_id is not None:
            existing_session = db_service.get_interview_session(db, session_id=session_id)
            if not existing_session:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Interview session with ID {session_id} not found.")
            if existing_session.user_id != current_user.id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
        else:
            new_session_in = InterviewSessionCreate(
                user_id=current_user.id,
                subject=request.subject,
                topic=request.topic,
                target_level=request.target_level,
                interview_mode=request.interview_type.lower().replace(" ", "_")
            )
            created_session = db_service.create_interview_session(db, session_in=new_session_in)
            session_id = created_session.id

        for q in generated_questions:
            db_service.create_question(
                db,
                question_in=QuestionCreate(
                    session_id=session_id,
                    question_text=q.question_text,
                    question_type=q.question_type,
                    difficulty=q.difficulty,
                    question_order=q.question_order
                )
            )

    return GenerateQuestionsResponse(
        success=True,
        session_id=session_id,
        subject=request.subject,
        topic=request.topic,
        target_level=request.target_level,
        interview_type=request.interview_type,
        total_questions=len(generated_questions),
        questions=generated_questions
    )


# ==============================================================================
# 1. Start Interview Session
# ==============================================================================

@router.post(
    "/sessions/start",
    response_model=CurrentQuestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new interactive mock interview session"
)
def start_interview_session(
    request: StartSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initializes a new interview session for the authenticated candidate:
    1. Calls Claude to generate the requested primary questions.
    2. Persists the session and generated questions to SQLite.
    3. Returns ONLY the first active question to ensure candidate cannot view future questions.
    """
    # 1. Generate primary interview questions using Claude
    gen_req = GenerateQuestionsRequest(
        subject=request.subject,
        topic=request.topic,
        target_level=request.target_level,
        interview_type=request.interview_type,
        num_questions=request.num_questions,
        session_duration=request.session_duration,
        difficulty=request.difficulty,
        custom_instructions=request.custom_instructions,
        save_to_database=False
    )

    try:
        generated_questions = ai_service.generate_interview_questions(gen_req)
    except ai_service.AIConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message)
    except ai_service.AIRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.message)
    except ai_service.AIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.error(f"Failed to generate questions for session startup: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error during session startup: {str(exc)}"
        )

    # 2. Persist new session in database
    new_session_in = InterviewSessionCreate(
        user_id=current_user.id,
        subject=request.subject,
        topic=request.topic,
        target_level=request.target_level,
        interview_mode=request.interview_type.lower().replace(" ", "_")
    )
    session = db_service.create_interview_session(db, session_in=new_session_in)

    # 3. Persist questions into database
    created_questions = []
    for idx, q in enumerate(generated_questions, start=1):
        db_q = db_service.create_question(
            db,
            question_in=QuestionCreate(
                session_id=session.id,
                question_text=q.question_text,
                question_type="main",
                difficulty=q.difficulty or request.difficulty,
                question_order=idx
            )
        )
        created_questions.append(db_q)

    first_question = created_questions[0] if created_questions else None

    return CurrentQuestionResponse(
        session_id=session.id,
        is_completed=False,
        current_question=QuestionResponse.model_validate(first_question) if first_question else None,
        question_number=1,
        total_questions=len(created_questions),
        progress_percent=0.0,
        session_status=session.status
    )


# ==============================================================================
# 2. Retrieve Active Current Question (State Preservation)
# ==============================================================================

@router.get(
    "/sessions/{session_id}/current-question",
    response_model=CurrentQuestionResponse,
    summary="Get current active question for candidate"
)
def get_current_question(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves the current unanswered question for the candidate.
    - Preserves state across browser reloads.
    - Never reveals future questions.
    - If all questions are answered, marks is_completed=True.
    """
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    all_questions = db_service.get_session_questions(db, session_id=session_id)
    total_count = len(all_questions)

    # Find active unanswered question
    active_q = db_service.get_active_unanswered_question(db, session_id=session_id)

    # Count answered questions
    answered_count = (
        db.query(Answer)
        .join(Question, Answer.question_id == Question.id)
        .filter(Question.session_id == session_id)
        .count()
    )

    if session.status == "completed" or active_q is None:
        return CurrentQuestionResponse(
            session_id=session.id,
            is_completed=True,
            current_question=None,
            question_number=answered_count,
            total_questions=total_count,
            progress_percent=100.0,
            session_status="completed"
        )

    progress = round((answered_count / total_count * 100), 1) if total_count > 0 else 0.0

    return CurrentQuestionResponse(
        session_id=session.id,
        is_completed=False,
        current_question=QuestionResponse.model_validate(active_q),
        question_number=answered_count + 1,
        total_questions=total_count,
        progress_percent=progress,
        session_status=session.status
    )


# ==============================================================================
# 3. Submit Answer with AI Evaluation & Adaptive Follow-Up
# ==============================================================================

@router.post(
    "/sessions/{session_id}/answers",
    response_model=AnswerSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit answer, receive AI evaluation, and trigger adaptive follow-up"
)
def submit_answer(
    session_id: int,
    request: SubmitAnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(check_ai_rate_limit)
):
    """
    Evaluates candidate's response and dynamically decides if an adaptive follow-up is warranted:
    - Validates answer input.
    - Prevents duplicate answers for the same question.
    - Evaluates technical correctness, depth, and clarity via Claude.
    - If shallow/misconception detected and follow-up ceiling not reached, generates targeted follow-up.
    - Saves evaluation and answer to SQLite.
    """
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    # Validate that question exists and belongs to this session
    question = db.query(Question).filter(Question.id == request.question_id, Question.session_id == session_id).first()
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found in this session.")

    # Prevent duplicate submissions
    existing_answer = db_service.get_question_answer(db, question_id=question.id)
    if existing_answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An answer has already been submitted for this question."
        )

    if session.status == "completed":
        # If this valid session question has no answer yet (e.g. dynamic/retry question), allow evaluation
        if not existing_answer:
            session.status = "in_progress"
            db.commit()
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This interview session has already ended.")

    # Count existing follow-ups in this session to enforce ceiling (max 1 per main question, max 2 per session)
    follow_up_count = db_service.count_session_follow_up_questions(db, session_id=session_id)

    # Call dedicated evaluation service using the 100-point rubric
    try:
        detailed_eval = evaluation_service.evaluate_answer_rubric(
            question_text=question.question_text,
            topic=session.topic,
            difficulty=question.difficulty or "medium",
            target_level=session.target_level,
            interview_type=session.interview_mode,
            candidate_answer=request.answer_text,
            current_follow_up_count=follow_up_count,
            max_follow_ups=1,
            question_id=question.id
        )
    except ai_service.AIConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message)
    except ai_service.AIRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.message)
    except ai_service.AIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.error(f"Error evaluating answer in session {session_id}: {exc}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI evaluation service error: {str(exc)}")

    eval_details_json = detailed_eval.model_dump_json()

    # Save evaluated answer with full model answer and concept breakdown
    answer_in = AnswerCreate(
        question_id=question.id,
        session_id=session_id,
        answer_text=request.answer_text,
        score=detailed_eval.numerical_score,
        correctness_status=detailed_eval.correctness_status,
        feedback=detailed_eval.evaluation_feedback or detailed_eval.brief_explanation,
        evaluation_feedback=detailed_eval.evaluation_feedback or detailed_eval.brief_explanation,
        strengths="\n".join(f"• {c}" for c in detailed_eval.correct_concepts) if detailed_eval.correct_concepts else None,
        improvement_areas="\n".join(f"• {m}" for m in detailed_eval.missing_concepts) if detailed_eval.missing_concepts else None,
        correct_concepts=detailed_eval.correct_concepts,
        incorrect_concepts=detailed_eval.incorrect_concepts,
        missing_concepts=detailed_eval.missing_concepts,
        correct_answer=detailed_eval.correct_answer,
        simple_explanation=detailed_eval.simple_explanation,
        detailed_explanation=detailed_eval.detailed_explanation,
        example=detailed_eval.example,
        key_points_to_remember=detailed_eval.key_points_to_remember,
        improved_answer=detailed_eval.improved_answer,
        evaluation_details=eval_details_json,
        model_answer_provider=detailed_eval.model_answer_provider,
        model_answer_error=detailed_eval.model_answer_error,
        model_answer_disclaimer=detailed_eval.model_answer_disclaimer
    )
    try:
        db_service.create_answer(db, answer_in=answer_in)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Database error saving answer for question {question.id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save evaluation result due to a database error."
        )

    # If adaptive follow-up is triggered, insert it right after the current question
    follow_up_obj = None
    if detailed_eval.should_follow_up and detailed_eval.follow_up_question:
        follow_up_obj = db_service.insert_follow_up_question(
            db=db,
            session_id=session_id,
            current_question_order=question.question_order,
            question_data=detailed_eval.follow_up_question
        )

    # Check next unanswered question
    next_q = db_service.get_active_unanswered_question(db, session_id=session_id)
    is_completed = (next_q is None)

    if is_completed:
        db_service.update_session_status(db, session_id=session_id, status="completed")

    # Metrics
    total_answered = (
        db.query(Answer)
        .join(Question, Answer.question_id == Question.id)
        .filter(Question.session_id == session_id)
        .count()
    )
    total_questions = db.query(Question).filter(Question.session_id == session_id).count()

    evaluation_result = evaluation_service.detailed_to_answer_result(
        detailed_eval,
        follow_up_question=detailed_eval.follow_up_question,
        candidate_answer=request.answer_text
    )

    return AnswerSubmissionResponse(
        success=True,
        session_id=session_id,
        question_id=question.id,
        evaluation=evaluation_result,
        has_follow_up=detailed_eval.should_follow_up and follow_up_obj is not None,
        follow_up_question=QuestionResponse.model_validate(follow_up_obj) if follow_up_obj else None,
        next_question=QuestionResponse.model_validate(next_q) if next_q else None,
        is_session_completed=is_completed,
        total_answered=total_answered,
        total_questions=total_questions
    )


# ==============================================================================
# 3b. Standalone Answer Evaluation (100-Point Rubric)
# ==============================================================================

@router.post(
    "/evaluate-answer",
    response_model=DetailedEvaluation,
    status_code=status.HTTP_200_OK,
    summary="Evaluate an answer using the 100-point rubric without an active session"
)
def evaluate_standalone_answer(
    request: EvaluateAnswerStandaloneRequest,
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(check_ai_rate_limit)
):
    """
    Directly evaluates a candidate's answer against a question using the 100-point rubric.
    Protected endpoint returning full rubric breakdown, strengths, gaps, and suggestions.
    """
    try:
        return evaluation_service.evaluate_answer_rubric(
            question_text=request.question_text,
            topic=request.topic,
            difficulty=request.difficulty,
            target_level=request.target_level,
            interview_type=request.interview_type,
            candidate_answer=request.candidate_answer,
            current_follow_up_count=0,
            max_follow_ups=1
        )
    except ai_service.AIConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message)
    except ai_service.AIRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=exc.message)
    except ai_service.AIServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.error(f"Error in standalone answer evaluation: {exc}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI evaluation error: {str(exc)}")


# ==============================================================================
# 4. Retrieve Next Question
# ==============================================================================

@router.get(
    "/sessions/{session_id}/next-question",
    response_model=CurrentQuestionResponse,
    summary="Get next unanswered question"
)
def get_next_question(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Convenience endpoint to retrieve the next pending question in session order."""
    return get_current_question(session_id=session_id, current_user=current_user, db=db)


# ==============================================================================
# 5. End Interview Session Early
# ==============================================================================

@router.post(
    "/sessions/{session_id}/end",
    response_model=EndSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="End an active interview session"
)
def end_interview_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Concludes the interview session, records completion timestamp, and tallies score."""
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this session.")

    completed_at = datetime.now(timezone.utc)
    db_service.update_session_status(db, session_id=session_id, status="completed", completed_at=completed_at)

    total_questions = db.query(Question).filter(Question.session_id == session_id).count()
    answered_count = (
        db.query(Answer)
        .join(Question, Answer.question_id == Question.id)
        .filter(Question.session_id == session_id)
        .count()
    )
    avg_score = db_service.calculate_session_average_score(db, session_id=session_id)

    summary_text = (
        f"Session completed on {session.topic}. Answered {answered_count} of {total_questions} questions "
        f"with average score {avg_score if avg_score is not None else 0.0}/100.0."
    )

    return EndSessionResponse(
        session_id=session.id,
        status="completed",
        completed_at=completed_at,
        questions_answered=answered_count,
        total_questions=total_questions,
        average_score=avg_score,
        summary=summary_text
    )


# ==============================================================================
# 6. Previous Sessions & Details
# ==============================================================================

@router.get(
    "/sessions",
    response_model=List[InterviewSessionResponse],
    summary="Get all interview sessions for current user"
)
def get_user_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves all viva/interview sessions for the authenticated candidate."""
    return db_service.get_user_interview_sessions(db, user_id=current_user.id)


@router.get(
    "/sessions/{session_id}",
    response_model=InterviewSessionResponse,
    summary="Get interview session details"
)
def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves an interview session by ID for the authenticated candidate."""
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    return session


@router.get(
    "/sessions/{session_id}/questions",
    response_model=List[QuestionResponse],
    summary="Get all questions for an interview session"
)
def get_session_questions(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves all questions generated for an interview session."""
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    return db_service.get_session_questions(db, session_id=session_id)


# ==============================================================================
# 7. Saved Detailed Evaluation & Model Answer Retrieval
# ==============================================================================

@router.get(
    "/sessions/{session_id}/questions/{question_id}/result",
    response_model=AnswerEvaluationResult,
    summary="Retrieve the saved detailed evaluation and model answer for a question"
)
def get_question_result(
    session_id: int,
    question_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves the saved evaluation, scoring rubric, and model answer for a specific question.
    - Strictly enforces user ownership (session.user_id == current_user.id).
    - Returns 403 Forbidden if accessed by another user.
    - Returns 404 Not Found if question was not answered yet or does not exist.
    """
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: you do not have permission to view results for this session.")

    question = db.query(Question).filter(Question.id == question_id, Question.session_id == session_id).first()
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question not found in this session.")

    answer = db_service.get_question_answer(db, question_id=question_id)
    if not answer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No answer or evaluation found for this question.")

    return db_service.format_answer_evaluation_result(answer, question=question)


@router.get(
    "/sessions/{session_id}/results",
    response_model=List[AnswerEvaluationResult],
    summary="Retrieve all saved evaluations and model answers for a session"
)
def get_session_results(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves all saved evaluations and model answers for all answered questions in the session.
    - Strictly enforces user ownership (session.user_id == current_user.id).
    """
    session = db_service.get_interview_session(db, session_id=session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: you do not have permission to view results for this session.")

    answers = (
        db.query(Answer)
        .join(Question, Answer.question_id == Question.id)
        .filter(Question.session_id == session_id)
        .order_by(Question.question_order.asc())
        .all()
    )
    return [db_service.format_answer_evaluation_result(a, question=a.question) for a in answers]
