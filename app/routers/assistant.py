"""Router for OpenAI GPT-powered Student Doubt-Clearing Assistant."""

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.models import User
from app.schemas import AskDoubtRequest, AskDoubtResponse, ExplainQuestionRequest
from app.security import get_current_user
from app.services.openai_service import (
    ask_gpt_doubt,
    OpenAIConfigurationError,
    OpenAIRateLimitError,
    OpenAIServiceError,
    OpenAITimeoutError,
)
from app.services.gemini_service import (
    ask_gemini_doubt,
    GeminiConfigurationError,
    GeminiRateLimitError,
    GeminiServiceError,
    GeminiTimeoutError,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/assistant", tags=["Student Doubt-Clearing Assistant"])


@router.post(
    "/chat",
    response_model=AskDoubtResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a technical or viva doubt to the OpenAI GPT AI Academic Tutor"
)
def chat_with_assistant(
    request: AskDoubtRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Receives a student's technical question or viva doubt and generates a structured,
    patient academic tutor explanation using OpenAI GPT.

    - Protected: Requires a valid JWT Bearer token.
    - Supports multi-turn conversation history for seamless follow-up clarifications.
    - Handles rate limits, timeouts, and API errors gracefully without fabricating answers.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question message cannot be empty or solely whitespace."
        )

    if len(request.message) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question message is too long (maximum 5,000 characters)."
        )

    # Limit history length to prevent context explosion
    sanitized_history = request.history[-10:] if request.history else []
    settings = get_settings()
    active_provider = (getattr(settings, "MODEL_ANSWER_PROVIDER", "openai") or "openai").lower().strip()

    try:
        if active_provider == "gemini":
            answer_text = ask_gemini_doubt(
                message=request.message,
                history=sanitized_history,
                context=request.context,
            )
            used_model = settings.GEMINI_MODEL or "gemini-1.5-flash"
        else:
            answer_text = ask_gpt_doubt(
                message=request.message,
                history=sanitized_history,
                context=request.context,
            )
            used_model = settings.OPENAI_MODEL or "gpt-4o-mini"

        return AskDoubtResponse(
            success=True,
            answer=answer_text,
            model=used_model,
            timestamp=datetime.now(timezone.utc),
        )

    except (OpenAIConfigurationError, GeminiConfigurationError) as e:
        logger.error(f"AI Assistant configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    except (OpenAIRateLimitError, GeminiRateLimitError) as e:
        logger.warning(f"AI Assistant rate limit error: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    except (OpenAITimeoutError, GeminiTimeoutError) as e:
        logger.warning(f"AI Assistant timeout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(e),
        )

    except (OpenAIServiceError, GeminiServiceError) as e:
        logger.error(f"AI Assistant service error: {e}")
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
        )

    except Exception as e:
        logger.exception(f"Unexpected error in assistant chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while communicating with the AI Assistant.",
        )


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Get AI assistant availability status"
)
def get_assistant_status(current_user: User = Depends(get_current_user)):
    """Returns configuration and availability status of the OpenAI GPT assistant."""
    settings = get_settings()
    active_provider = (getattr(settings, "MODEL_ANSWER_PROVIDER", "openai") or "openai").lower().strip()

    if active_provider == "gemini":
        api_key = (settings.GEMINI_API_KEY or "").strip()
        is_configured = bool(api_key)
        is_mock = api_key.lower() in ("mock", "demo", "test")
        return {
            "status": "ready" if is_configured else "not_configured",
            "provider": "Google Gemini",
            "model": settings.GEMINI_MODEL or "gemini-1.5-flash",
            "mode": "simulation" if is_mock else "live",
            "user_email": current_user.email,
        }
    else:
        api_key = (settings.OPENAI_API_KEY or "").strip()
        is_configured = bool(api_key)
        is_mock = api_key.lower() in ("mock", "demo", "test")
        return {
            "status": "ready" if is_configured else "not_configured",
            "provider": "OpenAI GPT-4o Mini",
            "model": settings.OPENAI_MODEL or "gpt-4o-mini",
            "mode": "simulation" if is_mock else "live",
            "user_email": current_user.email,
        }


@router.post(
    "/explain-question",
    response_model=AskDoubtResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask AI for further explanation regarding a specific viva question & model answer"
)
def explain_evaluated_question(
    request: ExplainQuestionRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Provides a tailored, patient explanation from OpenAI GPT on a specific question,
    candidate answer, and model answer.
    """
    if not request.question_text or not request.question_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question text cannot be empty."
        )

    if len(request.question_text) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question text is too long (maximum 5,000 characters)."
        )

    if request.doubt_query and len(request.doubt_query) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Doubt query is too long (maximum 5,000 characters)."
        )

    # Build rich context combining the original question and the model solution
    context_parts = [
        f"Original Question: {request.question_text.strip()}",
    ]
    if request.topic:
        context_parts.append(f"Topic: {request.topic.strip()}")
    if request.candidate_answer:
        context_parts.append(f"Candidate's Answer: {request.candidate_answer.strip()}")
    if request.correct_answer:
        context_parts.append(f"Model Solution: {request.correct_answer.strip()}")

    composed_context = "\n".join(context_parts)

    user_query = request.doubt_query.strip() if (request.doubt_query and request.doubt_query.strip()) else (
        "Please provide a deeper, crystal-clear explanation of this question and model solution. "
        "Explain the key concepts in simple terms with an analogy, break down any code line by line, "
        "and point out common viva pitfalls."
    )

    settings = get_settings()
    active_provider = (getattr(settings, "MODEL_ANSWER_PROVIDER", "openai") or "openai").lower().strip()

    try:
        if active_provider == "gemini":
            answer_text = ask_gemini_doubt(
                message=user_query,
                history=request.history[-6:] if request.history else [],
                context=composed_context,
            )
            used_model = settings.GEMINI_MODEL or "gemini-1.5-flash"
        else:
            answer_text = ask_gpt_doubt(
                message=user_query,
                history=request.history[-6:] if request.history else [],
                context=composed_context,
            )
            used_model = settings.OPENAI_MODEL or "gpt-4o-mini"

        return AskDoubtResponse(
            success=True,
            answer=answer_text,
            model=used_model,
            timestamp=datetime.now(timezone.utc),
        )

    except (OpenAIConfigurationError, GeminiConfigurationError) as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except (OpenAIRateLimitError, GeminiRateLimitError) as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except (OpenAITimeoutError, GeminiTimeoutError) as e:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e))
    except (OpenAIServiceError, GeminiServiceError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.exception(f"Unexpected error in explain-question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate additional explanation from AI Assistant.",
        )
