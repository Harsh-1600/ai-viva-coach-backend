"""Performance reporting and analytics service for mock vivas and interviews."""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models import InterviewSession, Question, Answer, SessionReport, User
from app.schemas import (
    DetailedSessionReportResponse,
    ScoreByQuestionItem,
    TopicPerformanceItem,
    RubricAverages,
    SessionHistoryItem,
    SessionProgressPoint,
    PerformanceStatisticsResponse,
)

logger = logging.getLogger(__name__)


def calculate_overall_and_average_scores(answers: List[Answer]) -> tuple[float, float, int]:
    """
    Clearly documented scoring formula:
    - Only answers with valid non-null numerical scores (0.0 to 100.0) are factored in.
    - Unanswered questions are NOT counted as zero unless explicitly submitted with a 0 score.
    - Formula:
        average_score = sum(valid_scores) / valid_answer_count
        overall_score = round(average_score, 1)
    - If valid_answer_count == 0, returns (0.0, 0.0, 0) safely without division by zero.
    """
    valid_scores = [a.score for a in answers if a.score is not None and a.score >= 0.0]
    count = len(valid_scores)
    if count == 0:
        return 0.0, 0.0, 0

    avg = sum(valid_scores) / count
    rounded_avg = round(avg, 1)
    return rounded_avg, rounded_avg, count


def parse_answer_evaluation_details(answer: Answer) -> Dict[str, Any]:
    """Parses JSON serialized evaluation_details stored in Answer model."""
    if not answer.evaluation_details:
        return {}
    try:
        data = json.loads(answer.evaluation_details)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Could not parse evaluation_details for answer {answer.id}: {exc}")
        return {}


def compile_session_report(
    db: Session,
    session: InterviewSession,
    force_regenerate: bool = False
) -> SessionReport:
    """
    Compiles or updates the comprehensive SessionReport for an interview session.
    Synthesizes question performance, rubric metrics, repeated mistakes, and study recommendations.
    """
    # Check if a report already exists and not forcing regeneration
    if not force_regenerate and session.report:
        return session.report

    # Query all questions and associated answers in chronological question_order
    questions = (
        db.query(Question)
        .filter(Question.session_id == session.id)
        .order_by(Question.question_order.asc())
        .all()
    )

    answers = (
        db.query(Answer)
        .join(Question, Answer.question_id == Question.id)
        .filter(Question.session_id == session.id)
        .all()
    )
    answer_map: Dict[int, Answer] = {a.question_id: a for a in answers}

    # 1. Calculate Scores using documented formula
    overall_score, avg_score, answered_count = calculate_overall_and_average_scores(answers)

    # 2. Extract Rubric Breakdowns and Topic Groupings
    rubric_totals = {
        "technical_correctness": 0.0,
        "conceptual_understanding": 0.0,
        "completeness": 0.0,
        "clarity": 0.0,
        "relevance": 0.0,
        "count": 0
    }

    topic_scores = defaultdict(list)
    all_correct_concepts: List[str] = []
    all_missing_concepts: List[str] = []
    all_inaccuracies: List[str] = []
    score_by_question_items: List[Dict[str, Any]] = []

    def parse_str_list(val):
        if not val:
            return []
        if isinstance(val, list):
            return [str(x) for x in val if x]
        if isinstance(val, str):
            try:
                p = json.loads(val)
                if isinstance(p, list):
                    return [str(x) for x in p if x]
            except Exception:
                pass
            return [line.strip().lstrip("•*- ").strip() for line in val.split("\n") if line.strip()]
        return []

    for q in questions:
        ans = answer_map.get(q.id)
        q_topic = getattr(q, "topic", None) or session.topic
        q_score = ans.score if ans else None
        is_ans = ans is not None

        # Cleanly reset all per-question variables to prevent any variable leakage
        details: Dict[str, Any] = {}
        correct_c: List[str] = []
        missing_c: List[str] = []
        inaccuracies: List[str] = []
        outline_c: List[str] = []
        kp_list: List[str] = []
        ex_text: Optional[str] = None
        c_status: Optional[str] = None
        c_ans: Optional[str] = None
        s_exp: Optional[str] = None
        d_exp: Optional[str] = None
        eval_fb: Optional[str] = None
        inc_c: List[str] = []

        if ans:
            if q_score is not None:
                topic_scores[q_topic].append(q_score)

            details = parse_answer_evaluation_details(ans)
            correct_c = parse_str_list(getattr(ans, "correct_concepts", None)) or details.get("correct_concepts_identified") or []
            missing_c = parse_str_list(getattr(ans, "missing_concepts", None)) or details.get("missing_concepts") or []
            inaccuracies = parse_str_list(getattr(ans, "incorrect_concepts", None)) or details.get("technical_inaccuracies") or []
            outline_c = details.get("suggested_ideal_outline") or []
            inc_c = inaccuracies

            all_correct_concepts.extend(correct_c)
            all_missing_concepts.extend(missing_c)
            all_inaccuracies.extend(inaccuracies)

            rb = details.get("rubric_breakdown")
            if isinstance(rb, dict):
                rubric_totals["technical_correctness"] += float(rb.get("technical_correctness", 0.0))
                rubric_totals["conceptual_understanding"] += float(rb.get("conceptual_understanding", 0.0))
                rubric_totals["completeness"] += float(rb.get("completeness", 0.0))
                rubric_totals["clarity"] += float(rb.get("clarity", 0.0))
                rubric_totals["relevance"] += float(rb.get("relevance", 0.0))
                rubric_totals["count"] += 1

            # Retrieve saved evaluation fields directly from SQLite columns
            c_ans = (getattr(ans, "correct_answer", None) or details.get("correct_answer") or "").strip() or None
            s_exp = (getattr(ans, "simple_explanation", None) or details.get("simple_explanation") or "").strip() or None
            d_exp = (getattr(ans, "detailed_explanation", None) or details.get("detailed_explanation") or "").strip() or None
            ex_text = (getattr(ans, "example", None) or details.get("example") or "").strip() or None
            c_status = getattr(ans, "correctness_status", None) or details.get("correctness_status")
            eval_fb = getattr(ans, "evaluation_feedback", None) or details.get("evaluation_feedback") or ans.feedback
            kp_list = parse_str_list(getattr(ans, "key_points_to_remember", None)) or parse_str_list(details.get("key_points_to_remember"))
        else:
            eval_fb = "Question was not answered during this session."

        score_by_question_items.append({
            "question_id": q.id,
            "question_order": q.question_order,
            "question_text": q.question_text,
            "topic": q_topic,
            "difficulty": q.difficulty or "medium",
            "score": q_score,
            "correctness_status": c_status,
            "is_answered": is_ans,
            "feedback": ans.feedback if ans else "Unanswered",
            "evaluation_feedback": eval_fb,
            "candidate_answer": ans.answer_text if ans else None,
            "correct_concepts": correct_c,
            "incorrect_concepts": inc_c,
            "missing_concepts": missing_c,
            "suggested_ideal_outline": outline_c,
            "correct_answer": c_ans,
            "simple_explanation": s_exp,
            "detailed_explanation": d_exp,
            "example": ex_text,
            "key_points_to_remember": kp_list,
            "model_answer_provider": getattr(ans, "model_answer_provider", None) or details.get("model_answer_provider") or "Google Gemini",
            "model_answer_error": getattr(ans, "model_answer_error", None) or details.get("model_answer_error"),
            "model_answer_disclaimer": getattr(ans, "model_answer_disclaimer", None) or details.get("model_answer_disclaimer") or "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
        })

    # 3. Calculate Rubric Averages
    rubric_count = rubric_totals["count"]
    if rubric_count > 0:
        rubric_averages = {
            "technical_correctness": round(rubric_totals["technical_correctness"] / rubric_count, 1),
            "conceptual_understanding": round(rubric_totals["conceptual_understanding"] / rubric_count, 1),
            "completeness": round(rubric_totals["completeness"] / rubric_count, 1),
            "clarity": round(rubric_totals["clarity"] / rubric_count, 1),
            "relevance": round(rubric_totals["relevance"] / rubric_count, 1),
        }
    else:
        rubric_averages = {
            "technical_correctness": 0.0,
            "conceptual_understanding": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "relevance": 0.0,
        }

    # 4. Rank Topics into Strongest and Requiring Improvement
    topic_performance_items: List[Dict[str, Any]] = []
    strongest_topics: List[str] = []
    weakest_topics: List[str] = []

    for t_name, scores in topic_scores.items():
        if scores:
            t_avg = round(sum(scores) / len(scores), 1)
            topic_performance_items.append({
                "topic": t_name,
                "average_score": t_avg,
                "question_count": len(scores)
            })

    topic_performance_items.sort(key=lambda x: x["average_score"], reverse=True)
    if topic_performance_items:
        strongest_topics = [t["topic"] for t in topic_performance_items if t["average_score"] >= 70.0]
        if not strongest_topics:
            strongest_topics = [topic_performance_items[0]["topic"]]

        weakest_topics = [t["topic"] for t in topic_performance_items if t["average_score"] < 70.0]
        if not weakest_topics and len(topic_performance_items) > 1:
            weakest_topics = [topic_performance_items[-1]["topic"]]
    else:
        strongest_topics = [session.topic]
        weakest_topics = []

    # 5. Identify Repeated Mistakes and Technical Gaps
    # Count occurrences of inaccuracies and missing points
    mistake_counts = defaultdict(int)
    for inacc in all_inaccuracies:
        mistake_counts[inacc] += 1
    for miss in all_missing_concepts:
        mistake_counts[f"Omission: {miss}"] += 1

    repeated_mistakes = [k for k, v in mistake_counts.items() if v > 1]
    if not repeated_mistakes and all_inaccuracies:
        repeated_mistakes = list(set(all_inaccuracies))[:3]
    if not repeated_mistakes and all_missing_concepts:
        repeated_mistakes = [f"Omission: {m}" for m in list(set(all_missing_concepts))[:3]]

    # 6. Qualitative Syntheses
    # Technical Knowledge Assessment
    tech_score = rubric_averages["technical_correctness"]
    comp_score = rubric_averages["completeness"]
    if overall_score >= 80:
        tech_assessment = (
            f"Candidate demonstrated robust technical command across {session.topic}. Core algorithms, data "
            f"structures, and hardware mechanisms were accurately articulated with strong conceptual depth "
            f"({tech_score}/30 technical accuracy, {comp_score}/20 completeness)."
        )
    elif overall_score >= 55:
        tech_assessment = (
            f"Candidate displayed a solid foundational understanding of {session.topic} but showed variable depth on "
            f"low-level mechanisms and edge cases ({tech_score}/30 technical accuracy, {comp_score}/20 completeness). "
            f"Key principles were recognized, but implementation details and trade-offs need refinement."
        )
    else:
        tech_assessment = (
            f"Candidate experienced difficulty with foundational principles of {session.topic} "
            f"({tech_score}/30 technical accuracy, {comp_score}/20 completeness). Critical mechanisms, terminology, "
            f"and operating workflows require structured revision."
        )

    # Communication & Clarity Feedback
    clar_score = rubric_averages["clarity"]
    rel_score = rubric_averages["relevance"]
    if clar_score >= 12 and rel_score >= 8:
        comm_feedback = (
            "Communication was concise, structured, and focused. The candidate answered the questions directly "
            "without tangential rambling and used standard engineering terminology effectively."
        )
    elif clar_score >= 8:
        comm_feedback = (
            "Explanations were generally understandable but could be more crisply structured. Practice framing responses "
            "with a three-part structure: 1) High-level definition, 2) Underlying mechanics, 3) Real-world trade-off."
        )
    else:
        comm_feedback = (
            "Responses were either too brief or lacked clear logical structuring. Focus on articulating thoughts step-by-step "
            "and directly addressing the examiner's prompt."
        )

    # Personalized Recommendations & Next Topics
    recommendations_list = []
    suggested_next_topics = []

    if overall_score < 70:
        recommendations_list.append(f"Review core textbook principles and official documentation for {session.topic}.")
    if rubric_averages["completeness"] < 14:
        recommendations_list.append("Broaden answers to mention edge cases, concurrency hazards, and memory overheads.")
    if rubric_averages["clarity"] < 10:
        recommendations_list.append("Practice speaking technical responses aloud to improve oral fluency and precision.")
    if not recommendations_list:
        recommendations_list.append("Practice advanced system design trade-offs and performance tuning scenarios.")

    # Next practice topics based on weakest topics
    if weakest_topics:
        for wt in weakest_topics:
            suggested_next_topics.append(f"{wt} - Deep Dive & Edge Cases")
    suggested_next_topics.append(f"Advanced {session.subject} System Architecture")
    suggested_next_topics.append(f"{session.topic} Failure Modes & Fault Tolerance")

    # High-level executive summary
    summary_text = (
        f"Completed {session.interview_mode.replace('_', ' ').title()} session on {session.topic} ({session.target_level}). "
        f"Answered {answered_count} of {len(questions)} questions with an overall estimated score of {overall_score}/100. "
        f"{'Demonstrated strong technical readiness.' if overall_score >= 75 else 'Identified specific areas for targeted revision.'}"
    )

    # Build Recharts details payload
    report_details_payload = {
        "score_by_question": score_by_question_items,
        "performance_by_topic": topic_performance_items,
        "rubric_breakdown_averages": rubric_averages,
        "overall_score": overall_score,
        "average_score": avg_score,
        "questions_answered": answered_count,
        "total_questions": len(questions)
    }

    # 7. Create or update SessionReport ORM object
    existing_report = session.report
    if existing_report:
        existing_report.overall_score = overall_score
        existing_report.questions_answered = answered_count
        existing_report.average_score = avg_score
        existing_report.summary = summary_text
        existing_report.strengths = "\n".join(f"• {c}" for c in list(set(all_correct_concepts))[:5])
        existing_report.improvement_areas = "\n".join(f"• {m}" for m in list(set(all_missing_concepts))[:5])
        existing_report.recommendations = "\n".join(f"• {r}" for r in recommendations_list)
        existing_report.strongest_topics = json.dumps(strongest_topics)
        existing_report.weakest_topics = json.dumps(weakest_topics)
        existing_report.technical_assessment = tech_assessment
        existing_report.communication_feedback = comm_feedback
        existing_report.repeated_mistakes = json.dumps(repeated_mistakes)
        existing_report.suggested_next_topics = json.dumps(suggested_next_topics)
        existing_report.report_details = json.dumps(report_details_payload)
        db.commit()
        db.refresh(existing_report)
        return existing_report
    else:
        new_report = SessionReport(
            session_id=session.id,
            overall_score=overall_score,
            questions_answered=answered_count,
            average_score=avg_score,
            summary=summary_text,
            strengths="\n".join(f"• {c}" for c in list(set(all_correct_concepts))[:5]),
            improvement_areas="\n".join(f"• {m}" for m in list(set(all_missing_concepts))[:5]),
            recommendations="\n".join(f"• {r}" for r in recommendations_list),
            strongest_topics=json.dumps(strongest_topics),
            weakest_topics=json.dumps(weakest_topics),
            technical_assessment=tech_assessment,
            communication_feedback=comm_feedback,
            repeated_mistakes=json.dumps(repeated_mistakes),
            suggested_next_topics=json.dumps(suggested_next_topics),
            report_details=json.dumps(report_details_payload)
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        return new_report


def format_detailed_report_response(
    session: InterviewSession,
    report: SessionReport
) -> DetailedSessionReportResponse:
    """Formats a SessionReport and InterviewSession into a DetailedSessionReportResponse schema."""
    details = {}
    if report.report_details:
        try:
            details = json.loads(report.report_details)
        except Exception:
            details = {}

    def parse_str_list(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [line.strip().lstrip("•*- ").strip() for line in raw.split("\n") if line.strip()]

    score_by_q_raw = details.get("score_by_question") or []
    ans_by_qid = {q.id: q.answer for q in session.questions if q.answer is not None}
    q_by_id = {q.id: q for q in session.questions}

    score_by_question: List[ScoreByQuestionItem] = []

    if score_by_q_raw:
        for item in score_by_q_raw:
            qid = item["question_id"]
            ans = ans_by_qid.get(qid)
            q_obj = q_by_id.get(qid)

            # Retrieve directly from SQLite Answer columns if item does not have it
            c_ans = item.get("correct_answer") or (getattr(ans, "correct_answer", None) if ans else None)
            s_exp = item.get("simple_explanation") or (getattr(ans, "simple_explanation", None) if ans else None)
            d_exp = item.get("detailed_explanation") or (getattr(ans, "detailed_explanation", None) if ans else None)
            ex_val = item.get("example") or (getattr(ans, "example", None) if ans else None)
            imp_val = item.get("improved_answer") or (getattr(ans, "improved_answer", None) if ans else None)
            c_stat = item.get("correctness_status") or (getattr(ans, "correctness_status", None) if ans else None)
            eval_fb = item.get("evaluation_feedback") or (getattr(ans, "evaluation_feedback", None) if ans else (item.get("feedback") or (ans.feedback if ans else None)))

            kp_raw = item.get("key_points_to_remember") or (parse_str_list(ans.key_points_to_remember) if ans and ans.key_points_to_remember else [])
            corr_raw = item.get("correct_concepts") or (parse_str_list(ans.correct_concepts) if ans and ans.correct_concepts else [])
            inc_raw = item.get("incorrect_concepts") or (parse_str_list(ans.incorrect_concepts) if ans and ans.incorrect_concepts else [])
            miss_raw = item.get("missing_concepts") or (parse_str_list(ans.missing_concepts) if ans and ans.missing_concepts else [])

            score_by_question.append(
                ScoreByQuestionItem(
                    question_id=qid,
                    question_order=item.get("question_order") or (q_obj.question_order if q_obj else 1),
                    question_text=item.get("question_text") or (q_obj.question_text if q_obj else ""),
                    topic=item.get("topic") or (getattr(q_obj, "topic", None) or session.topic),
                    difficulty=item.get("difficulty") or (q_obj.difficulty if q_obj else "medium"),
                    score=item.get("score") if (ans is not None or item.get("is_answered")) else None,
                    correctness_status=c_stat,
                    is_answered=item.get("is_answered", ans is not None),
                    feedback=item.get("feedback") or (ans.feedback if ans else None),
                    evaluation_feedback=eval_fb,
                    candidate_answer=item.get("candidate_answer") or (ans.answer_text if ans else None),
                    correct_concepts=corr_raw if isinstance(corr_raw, list) else parse_str_list(corr_raw),
                    incorrect_concepts=inc_raw if isinstance(inc_raw, list) else parse_str_list(inc_raw),
                    missing_concepts=miss_raw if isinstance(miss_raw, list) else parse_str_list(miss_raw),
                    suggested_ideal_outline=item.get("suggested_ideal_outline") or [],
                    correct_answer=c_ans,
                    simple_explanation=s_exp,
                    detailed_explanation=d_exp,
                    example=ex_val,
                    key_points_to_remember=kp_raw if isinstance(kp_raw, list) else parse_str_list(kp_raw),
                    improved_answer=imp_val,
                )
            )
    else:
        # Reconstruct directly from session questions and answers in SQLite
        for q_obj in sorted(session.questions, key=lambda x: x.question_order):
            ans = q_obj.answer
            score_by_question.append(
                ScoreByQuestionItem(
                    question_id=q_obj.id,
                    question_order=q_obj.question_order,
                    question_text=q_obj.question_text,
                    topic=getattr(q_obj, "topic", None) or session.topic,
                    difficulty=q_obj.difficulty or "medium",
                    score=ans.score if ans else None,
                    correctness_status=ans.correctness_status if ans else None,
                    is_answered=ans is not None,
                    feedback=ans.feedback if ans else "Unanswered",
                    evaluation_feedback=ans.evaluation_feedback if ans else ("Question was not answered during this session."),
                    candidate_answer=ans.answer_text if ans else None,
                    correct_concepts=parse_str_list(ans.correct_concepts) if ans and ans.correct_concepts else [],
                    incorrect_concepts=parse_str_list(ans.incorrect_concepts) if ans and ans.incorrect_concepts else [],
                    missing_concepts=parse_str_list(ans.missing_concepts) if ans and ans.missing_concepts else [],
                    suggested_ideal_outline=[],
                    correct_answer=ans.correct_answer if ans else None,
                    simple_explanation=ans.simple_explanation if ans else None,
                    detailed_explanation=ans.detailed_explanation if ans else None,
                    example=ans.example if ans else None,
                    key_points_to_remember=parse_str_list(ans.key_points_to_remember) if ans and ans.key_points_to_remember else [],
                    improved_answer=getattr(ans, "improved_answer", None) if ans else None,
                )
            )

    perf_by_t_raw = details.get("performance_by_topic") or []
    perf_by_topic = [
        TopicPerformanceItem(
            topic=item["topic"],
            average_score=float(item["average_score"]),
            question_count=int(item["question_count"])
        )
        for item in perf_by_t_raw
    ]

    rubric_avg_raw = details.get("rubric_breakdown_averages")
    rubric_averages = RubricAverages(**rubric_avg_raw) if rubric_avg_raw else None

    total_q = len(session.questions)

    return DetailedSessionReportResponse(
        id=report.id,
        session_id=session.id,
        subject=session.subject,
        topic=session.topic,
        target_level=session.target_level,
        interview_mode=session.interview_mode,
        session_status=session.status,
        started_at=session.started_at,
        completed_at=session.completed_at,
        overall_score=report.overall_score or 0.0,
        questions_answered=report.questions_answered,
        total_questions=total_q,
        average_score=report.average_score or report.overall_score or 0.0,
        summary=report.summary or f"Session completed on {session.topic}.",
        strengths=report.strengths,
        improvement_areas=report.improvement_areas,
        recommendations=report.recommendations,
        strongest_topics=parse_str_list(report.strongest_topics),
        weakest_topics=parse_str_list(report.weakest_topics),
        technical_assessment=report.technical_assessment or "Technical assessment compiled.",
        communication_feedback=report.communication_feedback or "Communication evaluation compiled.",
        repeated_mistakes=parse_str_list(report.repeated_mistakes),
        suggested_next_topics=parse_str_list(report.suggested_next_topics),
        score_by_question=score_by_question,
        performance_by_topic=perf_by_topic,
        rubric_breakdown_averages=rubric_averages,
        created_at=report.created_at
    )


def get_candidate_session_history(db: Session, user_id: int) -> List[SessionHistoryItem]:
    """Retrieves session history records for the authenticated candidate."""
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )

    history_items: List[SessionHistoryItem] = []
    for s in sessions:
        total_q = len(s.questions)
        answered_q = len([q for q in s.questions if q.answer is not None])
        score = s.report.overall_score if s.report else None

        history_items.append(SessionHistoryItem(
            session_id=s.id,
            subject=s.subject,
            topic=s.topic,
            target_level=s.target_level,
            interview_mode=s.interview_mode,
            status=s.status,
            started_at=s.started_at,
            completed_at=s.completed_at,
            questions_answered=answered_q,
            total_questions=total_q,
            overall_score=score,
            report_id=s.report.id if s.report else None
        ))

    return history_items


def get_candidate_performance_statistics(db: Session, user_id: int) -> PerformanceStatisticsResponse:
    """Computes cross-session analytics and session-over-session timeline for the user."""
    # Find all completed sessions with reports belonging to this user
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewSession.started_at.asc())
        .all()
    )

    total_sessions = len(sessions)
    completed_sessions = 0
    total_answered = 0
    scores_list = []
    timeline: List[SessionProgressPoint] = []
    topic_scores = defaultdict(list)

    rubric_totals = {
        "technical_correctness": 0.0,
        "conceptual_understanding": 0.0,
        "completeness": 0.0,
        "clarity": 0.0,
        "relevance": 0.0,
        "count": 0
    }

    for s in sessions:
        answered_count = len([q for q in s.questions if q.answer is not None])
        total_answered += answered_count

        if s.status == "completed" or s.report is not None:
            completed_sessions += 1
            if s.report and s.report.overall_score is not None:
                score = s.report.overall_score
                scores_list.append(score)
                topic_scores[s.topic].append(score)

                # Format date string for Recharts timeline
                date_str = s.started_at.strftime("%b %d") if s.started_at else f"Session #{s.id}"
                timeline.append(SessionProgressPoint(
                    session_id=s.id,
                    topic=s.topic,
                    date=date_str,
                    score=score
                ))

                # Accumulate rubric profile
                if s.report.report_details:
                    try:
                        details = json.loads(s.report.report_details)
                        r_avgs = details.get("rubric_breakdown_averages")
                        if isinstance(r_avgs, dict):
                            rubric_totals["technical_correctness"] += float(r_avgs.get("technical_correctness", 0.0))
                            rubric_totals["conceptual_understanding"] += float(r_avgs.get("conceptual_understanding", 0.0))
                            rubric_totals["completeness"] += float(r_avgs.get("completeness", 0.0))
                            rubric_totals["clarity"] += float(r_avgs.get("clarity", 0.0))
                            rubric_totals["relevance"] += float(r_avgs.get("relevance", 0.0))
                            rubric_totals["count"] += 1
                    except Exception:
                        pass

    cum_avg = round(sum(scores_list) / len(scores_list), 1) if scores_list else None

    # Calculate overall rubric profile
    rubric_count = rubric_totals["count"]
    rubric_profile = None
    if rubric_count > 0:
        rubric_profile = RubricAverages(
            technical_correctness=round(rubric_totals["technical_correctness"] / rubric_count, 1),
            conceptual_understanding=round(rubric_totals["conceptual_understanding"] / rubric_count, 1),
            completeness=round(rubric_totals["completeness"] / rubric_count, 1),
            clarity=round(rubric_totals["clarity"] / rubric_count, 1),
            relevance=round(rubric_totals["relevance"] / rubric_count, 1),
        )

    # Strongest and weakest topics
    strongest_topics: List[str] = []
    weakest_topics: List[str] = []
    ranked_topics = []
    for top, top_scores in topic_scores.items():
        if top_scores:
            ranked_topics.append((top, sum(top_scores) / len(top_scores)))
    ranked_topics.sort(key=lambda x: x[1], reverse=True)

    if ranked_topics:
        strongest_topics = [t[0] for t in ranked_topics if t[1] >= 70.0]
        if not strongest_topics:
            strongest_topics = [ranked_topics[0][0]]
        weakest_topics = [t[0] for t in ranked_topics if t[1] < 70.0]
        if not weakest_topics and len(ranked_topics) > 1:
            weakest_topics = [ranked_topics[-1][0]]

    return PerformanceStatisticsResponse(
        total_sessions=total_sessions,
        completed_sessions=completed_sessions,
        total_questions_answered=total_answered,
        cumulative_average_score=cum_avg,
        progress_timeline=timeline,
        strongest_topics=strongest_topics,
        weakest_topics=weakest_topics,
        rubric_profile=rubric_profile
    )
