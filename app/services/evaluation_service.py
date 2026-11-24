"""Dedicated AI answer evaluation service using 100-point configurable scoring rubric."""

import json
import logging
import re
from typing import Optional, Any, Dict
import anthropic

from app.config import get_settings
from app.prompts.answer_evaluation import (
    DETAILED_EVALUATION_SYSTEM_PROMPT,
    build_detailed_evaluation_prompt,
)
from app.schemas import DetailedEvaluation, RubricBreakdown, AnswerEvaluationResult
from app.services.evaluation_rubric import DEFAULT_RUBRIC, ScoringRubric
from app.services import ai_service, gemini_service, openai_service, reference_answers
from app.services.ai_service import (
    AIServiceError,
    AIConfigurationError,
    AIRateLimitError,
    AIResponseParseError,
)

logger = logging.getLogger(__name__)


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps a floating point value between min_val and max_val."""
    return max(min_val, min(max_val, round(val, 1)))


def create_fallback_evaluation(
    candidate_answer: str,
    error_reason: str = "Fallback evaluation applied due to parsing exception.",
    question_text: str = "",
    topic: str = "",
    question_id: Optional[int] = None
) -> DetailedEvaluation:
    """
    Safely generates a fallback DetailedEvaluation if AI evaluation fails or times out.
    Enforces that failures never receive a fake passing or default score (awards 0.0).
    """
    topic_label = topic or "the requested topic"
    return DetailedEvaluation(
        question_id=question_id,
        numerical_score=0.0,
        score=0.0,
        correctness_status="incorrect",
        brief_explanation=f"Evaluation could not be completed by AI service: {error_reason}",
        evaluation_feedback=f"Evaluation could not be completed by AI service: {error_reason}",
        correct_concepts=[],
        correct_concepts_identified=[],
        incorrect_concepts=[],
        technical_inaccuracies=[],
        missing_concepts=["Evaluation incomplete due to service error"],
        correct_answer=f"A complete, authoritative answer for {topic_label} thoroughly details its core mechanism, operating architecture, and practical engineering trade-offs.",
        simple_explanation=f"In simple terms, {topic_label} provides a standardized, reliable method to perform necessary operations without system corruption.",
        detailed_explanation=f"A rigorous technical explanation of {topic_label} covers architectural interfaces, component state lifecycles, and failure recovery protocols.",
        example=f"In production software systems, {topic_label} is applied through modular libraries and defensive design patterns.",
        key_points_to_remember=[
            f"State the primary definition and architectural purpose of {topic_label}.",
            "Walk through internal execution mechanisms and component interactions.",
            "Always discuss real-world system trade-offs in technical viva examinations."
        ],
        improved_answer=f"\"To answer '{question_text or topic_label}' comprehensively in a viva: Begin with a direct definition, walk through the step-by-step mechanism, and conclude with key trade-offs and practical considerations.\"",
        candidate_answer=candidate_answer,
        specific_improvement_suggestions=["Please retry submitting your answer."],
        suggested_ideal_outline=[
            f"Define {topic_label} and its primary purpose",
            "Detail internal mechanics and data flow",
            "Explain system trade-offs and edge cases"
        ],
        next_difficulty_recommendation="same",
        rubric_breakdown=RubricBreakdown(
            technical_correctness=0.0,
            conceptual_understanding=0.0,
            completeness=0.0,
            clarity=0.0,
            relevance=0.0
        ),
        is_correct=False,
        should_follow_up=False,
        follow_up_reason=None,
        is_ai_estimate=True,
        model_answer_provider="Google Gemini",
        model_answer_error=f"Model answer generation unavailable: {error_reason}",
        model_answer_disclaimer="AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
    )


def evaluate_answer_rubric(
    question_text: str,
    topic: str,
    difficulty: str,
    target_level: str,
    interview_type: str,
    candidate_answer: str,
    current_follow_up_count: int = 0,
    max_follow_ups: int = 1,
    question_id: Optional[int] = None,
    custom_client: Optional[Any] = None,
    rubric: ScoringRubric = DEFAULT_RUBRIC
) -> DetailedEvaluation:
    """
    Evaluates candidate response using the 100-point configurable rubric.
    Generates both:
    1. Comprehensive evaluation of the candidate's answer.
    2. Detailed, technically accurate model answer with simple & detailed explanations,
       practical examples, and key points to remember.
    """
    clean_ans = candidate_answer.strip()
    clean_lower = clean_ans.lower()
    has_substantive_content = len(clean_ans) >= 40 and any(w in clean_lower for w in [
        "polymorphism", "virtual memory", "mutex", "acid", "generator", "indexing", "paging", "inheritance", "encapsulation", "many forms", "pages", "frames", "dynamic dispatch", "method"
    ])
    is_empty_or_idk = not has_substantive_content and (
        len(clean_ans) == 0
        or any(x in clean_lower for x in [
            "don't know", "dont know", "do not know", "not know", "idk",
            "no idea", "pass", "skip", "unfamiliar"
        ])
        or clean_lower in ["not sure", "unsure", "i am not sure", "i'm not sure"]
        or (len(clean_ans) < 35 and any(x in clean_lower for x in ["not sure", "unsure"]))
    )

    client = ai_service.get_anthropic_client(custom_client)
    settings = get_settings()
    model_name = settings.ANTHROPIC_MODEL

    eval_prompt = build_detailed_evaluation_prompt(
        question_text=question_text,
        topic=topic,
        difficulty=difficulty,
        target_level=target_level,
        interview_type=interview_type,
        candidate_answer=candidate_answer if not is_empty_or_idk else "[Candidate provided an empty answer or indicated unfamiliarity with the topic]",
        current_follow_up_count=current_follow_up_count,
        max_follow_ups=max_follow_ups
    )

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.2,
            system=DETAILED_EVALUATION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": eval_prompt}
            ]
        )
    except anthropic.AuthenticationError as exc:
        raise AIServiceError("Invalid Anthropic API key credentials.", status_code=502)
    except anthropic.RateLimitError:
        raise AIRateLimitError()
    except anthropic.APITimeoutError:
        logger.warning("Anthropic timeout during answer evaluation.")
        raise AIServiceError("AI evaluation request timed out. Please try again.", status_code=504)
    except anthropic.APIConnectionError:
        raise AIServiceError("Could not connect to Anthropic AI service.", status_code=502)
    except anthropic.APIStatusError as exc:
        raise AIServiceError(f"Anthropic API error: {exc.message}", status_code=502)
    except Exception as exc:
        logger.error(f"Unexpected error in Anthropic call: {exc}")
        raise AIServiceError(f"AI evaluation service error: {str(exc)}", status_code=502)

    content_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            content_text += block.text

    if not content_text.strip():
        logger.warning("Empty response from AI examiner.")
        raise AIResponseParseError("Empty response received from AI evaluation service.")

    try:
        raw_json = ai_service.extract_json_payload(content_text)
    except Exception as exc:
        logger.warning(f"Could not parse JSON from AI evaluation: {exc}.")
        raise AIResponseParseError(f"Failed to parse AI evaluation JSON: {str(exc)}")

    if not isinstance(raw_json, dict):
        logger.warning("AI evaluation response was not a dictionary.")
        raise AIResponseParseError("AI evaluation response was not a valid JSON dictionary.")

    # Parse and clamp rubric breakdown (defaulting to 0.0 so unaddressed dimensions do not award free marks)
    raw_breakdown = raw_json.get("rubric_breakdown") or {}
    tc = clamp(float(raw_breakdown.get("technical_correctness", 0.0)), 0.0, rubric.technical_correctness.max_score)
    cu = clamp(float(raw_breakdown.get("conceptual_understanding", 0.0)), 0.0, rubric.conceptual_understanding.max_score)
    comp = clamp(float(raw_breakdown.get("completeness", 0.0)), 0.0, rubric.completeness.max_score)
    clar = clamp(float(raw_breakdown.get("clarity", 0.0)), 0.0, rubric.clarity.max_score)
    rel = clamp(float(raw_breakdown.get("relevance", 0.0)), 0.0, rubric.relevance.max_score)

    is_correct = raw_json.get("is_correct")

    # Correctness Gating: In academic viva and technical interviews, an answer with 0 technical
    # correctness or empty input MUST NOT receive marks for clarity or relevance.
    # Note: If is_correct is False but tc > 0, the candidate provided partially correct concepts
    # and earned partial marks; only zero out if empty/idk or tc == 0.0 or explicitly 0-scored incorrect response.
    raw_score_val = raw_json.get("score") if raw_json.get("score") is not None else raw_json.get("numerical_score")
    is_truly_zero = (raw_score_val is not None and float(raw_score_val) == 0.0) or (tc == 0.0 and cu == 0.0)

    if is_empty_or_idk or (tc == 0.0 and (is_correct is False or is_truly_zero)):
        tc = 0.0
        cu = 0.0
        comp = 0.0
        clar = 0.0
        rel = 0.0
        score = 0.0
        is_correct = False
    else:
        # Numerical score: either sum of breakdown, numerical_score, or score
        calculated_sum = round(tc + cu + comp + clar + rel, 1)
        raw_score = raw_json.get("score")
        if raw_score is None:
            raw_score = raw_json.get("numerical_score")
        if raw_score is not None:
            try:
                score = clamp(float(raw_score), 0.0, 100.0)
            except (ValueError, TypeError):
                score = calculated_sum
        else:
            score = calculated_sum
        if score == 0.0:
            is_correct = False
        elif is_correct is None or score >= 40.0:
            is_correct = (score >= 40.0)

    rubric_breakdown = RubricBreakdown(
        technical_correctness=tc,
        conceptual_understanding=cu,
        completeness=comp,
        clarity=clar,
        relevance=rel
    )

    # Next difficulty recommendation
    next_diff = str(raw_json.get("next_difficulty_recommendation", "")).lower().strip()
    if next_diff not in ["easier", "same", "harder"]:
        if score < 50.0:
            next_diff = "easier"
        elif score > 80.0:
            next_diff = "harder"
        else:
            next_diff = "same"

    # Lists helper
    def clean_str_list(items: Any) -> list:
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
        elif isinstance(items, str) and items.strip():
            return [items.strip()]
        return []

    correct_concepts = clean_str_list(raw_json.get("correct_concepts"))
    if not correct_concepts and raw_json.get("correct_concepts_identified"):
        correct_concepts = clean_str_list(raw_json.get("correct_concepts_identified"))
    if not correct_concepts and raw_json.get("strengths"):
        correct_concepts = clean_str_list(raw_json.get("strengths"))

    missing_concepts = clean_str_list(raw_json.get("missing_concepts"))
    if not missing_concepts and raw_json.get("improvement_areas"):
        missing_concepts = clean_str_list(raw_json.get("improvement_areas"))

    incorrect_concepts = clean_str_list(raw_json.get("incorrect_concepts"))
    if not incorrect_concepts and raw_json.get("technical_inaccuracies"):
        incorrect_concepts = clean_str_list(raw_json.get("technical_inaccuracies"))

    tech_inaccuracies = incorrect_concepts
    improvements = clean_str_list(raw_json.get("specific_improvement_suggestions"))
    ideal_outline = clean_str_list(raw_json.get("suggested_ideal_outline"))

    explanation = str(raw_json.get("evaluation_feedback") or raw_json.get("brief_explanation") or raw_json.get("feedback") or "Candidate response evaluated.").strip()

    # ==============================================================================
    # Independent Model Answer Generation (Delegated Provider Design)
    # ==============================================================================
    designated_model_provider = (getattr(settings, "MODEL_ANSWER_PROVIDER", "openai") or "openai").lower().strip()
    if designated_model_provider in ("openai", "gpt"):
        model_answer_provider_name = "OpenAI GPT"
    elif designated_model_provider == "gemini":
        model_answer_provider_name = "Google Gemini"
    else:
        model_answer_provider_name = "Anthropic Claude"

    model_answer_error = None
    model_answer_disclaimer = "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."

    if designated_model_provider in ("openai", "gpt"):
        try:
            gpt_res = openai_service.generate_gpt_model_answer(
                question_text=question_text,
                subject=topic,
                topic=topic,
                difficulty=difficulty,
                target_level=target_level,
                candidate_answer=candidate_answer,
                custom_client=custom_client
            )
            correct_ans = gpt_res["correct_answer"]
            simple_exp = gpt_res["simple_explanation"]
            detailed_exp = gpt_res["detailed_explanation"]
            example_text = gpt_res["example"]
            key_points = gpt_res["key_points_to_remember"]
            model_answer_disclaimer = gpt_res.get("disclaimer") or model_answer_disclaimer
        except Exception as gpt_exc:
            logger.warning(f"Failed to generate OpenAI GPT model answer: {gpt_exc}")
            model_answer_error = f"OpenAI GPT was unable to generate a model answer: {str(gpt_exc)}. No fabricated answer was provided."
            correct_ans = ""
            simple_exp = ""
            detailed_exp = ""
            example_text = None
            key_points = []
    elif designated_model_provider == "gemini":
        try:
            gemini_res = gemini_service.generate_gemini_model_answer(
                question_text=question_text,
                subject=topic,
                topic=topic,
                difficulty=difficulty,
                target_level=target_level,
                candidate_answer=candidate_answer,
                custom_client=custom_client
            )
            correct_ans = gemini_res["correct_answer"]
            simple_exp = gemini_res["simple_explanation"]
            detailed_exp = gemini_res["detailed_explanation"]
            example_text = gemini_res["example"]
            key_points = gemini_res["key_points_to_remember"]
            model_answer_disclaimer = gemini_res.get("disclaimer") or model_answer_disclaimer
        except Exception as gem_exc:
            logger.warning(f"Failed to generate Gemini model answer: {gem_exc}")
            # Do NOT silently switch providers or fabricate answers on failure
            model_answer_error = f"Gemini AI was unable to generate a model answer: {str(gem_exc)}. No fabricated answer was provided."
            correct_ans = ""
            simple_exp = ""
            detailed_exp = ""
            example_text = None
            key_points = []
    else:
        # Fallback to Claude's model answer
        correct_ans = str(raw_json.get("correct_answer") or raw_json.get("model_answer") or "").strip()
        if not correct_ans:
            correct_ans = f"A technically accurate model answer for '{question_text}' explains the underlying mechanism, operational flow, and key system trade-offs."

        simple_exp = str(raw_json.get("simple_explanation") or raw_json.get("plain_explanation") or "").strip()
        if not simple_exp:
            simple_exp = f"In simple terms, {topic} provides a structured and reliable approach to handling system operations predictably."

        detailed_exp = str(raw_json.get("detailed_explanation") or raw_json.get("deep_explanation") or "").strip()
        if not detailed_exp:
            detailed_exp = correct_ans

        example_text = str(raw_json.get("example") or raw_json.get("practical_example") or "").strip() or None

        key_points = clean_str_list(raw_json.get("key_points_to_remember") or raw_json.get("key_points") or raw_json.get("key_takeaways"))
        if not key_points:
            key_points = [
                f"State the primary definition and purpose of {topic}.",
                "Detail internal components, workflows, and edge case exceptions.",
                "Articulate performance trade-offs during technical vivas and interviews."
            ]

    # Correctness status determination
    correctness_status = str(raw_json.get("correctness_status") or "").lower().strip()
    if correctness_status not in ["correct", "partially_correct", "incorrect", "empty"]:
        if is_empty_or_idk or (score == 0.0 and len(clean_ans) == 0):
            correctness_status = "empty"
        elif score >= 75.0:
            correctness_status = "correct"
        elif score >= 40.0:
            correctness_status = "partially_correct"
        else:
            correctness_status = "incorrect"

    # Follow-up determination
    should_follow_up = bool(raw_json.get("should_follow_up", False))
    if current_follow_up_count >= max_follow_ups:
        should_follow_up = False

    follow_up_reason = raw_json.get("follow_up_reason")
    follow_up_q = raw_json.get("follow_up_question")
    if not should_follow_up or not isinstance(follow_up_q, dict) or not follow_up_q.get("question_text"):
        should_follow_up = False
        follow_up_q = None

    # Determine improved viva answer
    improved_ans = raw_json.get("improved_answer")
    if not improved_ans and designated_model_provider == "gemini" and not model_answer_error:
        improved_ans = gemini_res.get("improved_answer") if "gemini_res" in locals() and isinstance(gemini_res, dict) else None
    if not improved_ans and not model_answer_error:
        ref_sol = reference_answers.get_authoritative_reference_solution(
            question_text=question_text,
            topic=topic,
            difficulty=difficulty
        )
        improved_ans = ref_sol.get("improved_answer")

    # Construct Pydantic model for validation
    evaluation_obj = DetailedEvaluation(
        question_id=question_id,
        numerical_score=score,
        score=score,
        correctness_status=correctness_status,
        brief_explanation=explanation,
        evaluation_feedback=explanation,
        correct_concepts=correct_concepts,
        correct_concepts_identified=correct_concepts,
        incorrect_concepts=incorrect_concepts,
        technical_inaccuracies=tech_inaccuracies,
        missing_concepts=missing_concepts,
        correct_answer=correct_ans,
        simple_explanation=simple_exp,
        detailed_explanation=detailed_exp,
        example=example_text,
        key_points_to_remember=key_points,
        improved_answer=improved_ans,
        candidate_answer=candidate_answer,
        specific_improvement_suggestions=improvements,
        suggested_ideal_outline=ideal_outline,
        next_difficulty_recommendation=next_diff,
        rubric_breakdown=rubric_breakdown,
        is_correct=is_correct,
        should_follow_up=should_follow_up,
        follow_up_reason=follow_up_reason,
        follow_up_question=follow_up_q,
        is_ai_estimate=True,
        model_answer_provider=model_answer_provider_name,
        model_answer_error=model_answer_error,
        model_answer_disclaimer=model_answer_disclaimer
    )

    return evaluation_obj


def detailed_to_answer_result(
    evaluation: DetailedEvaluation,
    follow_up_question: Optional[dict] = None,
    candidate_answer: Optional[str] = None
) -> AnswerEvaluationResult:
    """Converts a validated DetailedEvaluation into an AnswerEvaluationResult."""
    strengths_text = "\n".join(f"• {c}" for c in evaluation.correct_concepts) if evaluation.correct_concepts else None
    improvements_text = "\n".join(f"• {m}" for m in evaluation.missing_concepts) if evaluation.missing_concepts else None

    return AnswerEvaluationResult(
        question_id=evaluation.question_id,
        score=evaluation.numerical_score,
        correctness_status=evaluation.correctness_status,
        evaluation_feedback=evaluation.evaluation_feedback or evaluation.brief_explanation,
        correct_concepts=evaluation.correct_concepts,
        incorrect_concepts=evaluation.incorrect_concepts,
        missing_concepts=evaluation.missing_concepts,
        correct_answer=evaluation.correct_answer,
        simple_explanation=evaluation.simple_explanation,
        detailed_explanation=evaluation.detailed_explanation,
        example=evaluation.example,
        key_points_to_remember=evaluation.key_points_to_remember,
        improved_answer=evaluation.improved_answer,
        feedback=evaluation.brief_explanation,
        candidate_answer=candidate_answer or evaluation.candidate_answer,
        is_correct=evaluation.is_correct,
        strengths=strengths_text,
        improvement_areas=improvements_text,
        technical_inaccuracies=evaluation.incorrect_concepts,
        improvement_suggestions=evaluation.specific_improvement_suggestions,
        suggested_ideal_outline=evaluation.suggested_ideal_outline,
        next_difficulty=evaluation.next_difficulty_recommendation,
        rubric_breakdown=evaluation.rubric_breakdown,
        should_follow_up=evaluation.should_follow_up,
        follow_up_reason=evaluation.follow_up_reason,
        is_ai_estimate=True,
        model_answer_provider=evaluation.model_answer_provider,
        model_answer_error=evaluation.model_answer_error,
        model_answer_disclaimer=evaluation.model_answer_disclaimer
    )
