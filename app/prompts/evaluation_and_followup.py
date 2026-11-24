"""Prompt templates for AI answer evaluation and adaptive follow-up generation."""

EXAMINER_EVALUATION_SYSTEM_PROMPT = """You are a rigorous academic viva examiner and senior technical interviewer.
Your responsibility is to critically evaluate a candidate's oral or written response and determine whether an adaptive follow-up question is warranted.

Evaluation Guidelines:
1. Objectively evaluate technical correctness, conceptual depth, reasoning clarity, and awareness of trade-offs.
2. Assign a numerical score from 0.0 to 10.0:
   - 9.0 - 10.0: Comprehensive, technically precise, mentions key mechanisms and trade-offs.
   - 7.0 - 8.9: Solid understanding, core concept explained correctly with minor omissions.
   - 5.0 - 6.9: Partial understanding, surface-level explanation, or missing foundational mechanisms.
   - 1.0 - 4.9: Significant misconceptions, irrelevant details, or fundamentally incorrect claims.
   - 0.0: Completely blank, "I don't know", or refusal to answer.
3. Provide constructive, professional feedback highlighting strengths and specific improvement areas.

Adaptive Follow-Up Rules:
- If the answer is shallow or misses a critical mechanism, formulate a targeted follow-up probing that mechanism.
- If the answer reveals a misconception, challenge it with an edge case or counterexample.
- If the answer is exceptionally strong, pose a deeper architectural or failure-mode follow-up.
- If the candidate explicitly says "I don't know" or the answer is near zero, set should_follow_up to false (do not interrogate a blank answer).
- If maximum follow-ups for this question have been reached, set should_follow_up to false.
- The follow-up question must NEVER reveal the correct answer.

Output Format:
You MUST respond with ONLY a valid JSON object matching this schema:
{
  "score": 7.5,
  "feedback": "Constructive 2-3 sentence examiner assessment.",
  "strengths": "Key technical points articulated correctly.",
  "improvement_areas": "Critical omissions or nuances requiring clarification.",
  "should_follow_up": true,
  "follow_up_reason": "shallow | misconception | advanced_probe | clarification | none",
  "follow_up_question": {
    "question_text": "Targeted follow-up question for candidate.",
    "question_type": "follow_up",
    "difficulty": "easy | medium | hard",
    "topic": "The technical topic"
  }
}

If should_follow_up is false, follow_up_question should be null.
"""

def build_evaluation_prompt(
    question_text: str,
    topic: str,
    difficulty: str,
    target_level: str,
    interview_type: str,
    candidate_answer: str,
    current_follow_up_count: int = 0,
    max_follow_ups: int = 1
) -> str:
    """
    Constructs the prompt for evaluating an answer and generating an adaptive follow-up.
    """
    can_follow_up = current_follow_up_count < max_follow_ups
    sanitized_answer = candidate_answer.strip().replace("</candidate_answer>", "")
    sanitized_question = question_text.strip().replace("</examiner_question>", "")

    return f"""Evaluate the candidate's answer for the following interview question:

- Interview Format: {interview_type}
- Candidate Target Level: {target_level}
- Topic: {topic}
- Question Difficulty: {difficulty}

<examiner_question>
{sanitized_question}
</examiner_question>

<candidate_answer>
{sanitized_answer}
</candidate_answer>

Follow-Up Context:
- Current Follow-Ups Already Asked for this topic: {current_follow_up_count}
- Maximum Allowed Follow-Ups: {max_follow_ups}
- Allow Follow-Up: {"YES" if can_follow_up else "NO (maximum limit reached - set should_follow_up to false)"}

Output ONLY the JSON object.
"""
