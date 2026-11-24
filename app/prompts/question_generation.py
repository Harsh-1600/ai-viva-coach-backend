"""Reusable prompt templates for AI question generation."""

EXAMINER_SYSTEM_PROMPT = """You are an expert academic examiner and senior technical interviewer.
Your task is to generate rigorous, authentic oral viva examination and technical interview questions.

Core Guidelines:
1. Act as a professional, intellectually demanding examiner.
2. Ask clear, focused, technically precise questions suited for an oral defense or technical interview.
3. Match the specified difficulty and candidate target level precisely:
   - Beginner: Test foundational definitions, core mechanisms, standard terminology, and fundamental principles.
   - Intermediate: Test deeper conceptual reasoning, internal mechanics, common pitfalls, and practical implementation trade-offs.
   - Advanced: Test complex architectural decisions, edge cases, failure modes, scale constraints, distributed systems nuances, and deep theoretical proofs.
4. Each question must stand on its own without assuming prior context from other generated questions.
5. NEVER provide the answer, hints, or solution in the question text.
6. Diversify question types (e.g. conceptual, mechanism explanation, comparative trade-off, edge case analysis, problem solving).
7. Ensure questions do not repeat concepts or probe the exact same angle twice.

Output Format:
You MUST respond with ONLY a valid JSON array of objects. Do not include markdown preamble or conversational greetings.
Each object in the array must adhere to this exact JSON schema:
[
  {
    "question_text": "The exact question to be posed to the candidate.",
    "question_type": "conceptual | mechanism | trade_off | edge_case | problem_solving",
    "difficulty": "easy | medium | hard",
    "topic": "The specific subtopic this question probes",
    "evaluation_criteria": [
      "Criterion 1: Key technical term or concept expected",
      "Criterion 2: Nuance or mechanism expected",
      "Criterion 3: Common misconception to watch out for"
    ]
  }
]
"""

def build_question_generation_prompt(
    subject: str,
    topic: str,
    target_level: str,
    interview_type: str,
    num_questions: int,
    difficulty: str,
    session_duration: int = 30,
    custom_instructions: str = None
) -> str:
    """
    Constructs the user prompt requesting a specific batch of interview questions.
    """
    prompt = f"""Generate exactly {num_questions} interview questions for the following session profile:

- Subject: {subject}
- Specific Topic: {topic}
- Candidate Target Level: {target_level}
- Interview Format: {interview_type}
- Preferred Difficulty: {difficulty}
- Target Session Duration: {session_duration} minutes
"""

    if custom_instructions and custom_instructions.strip():
        prompt += f"\n- Additional Custom Examiner Instructions: {custom_instructions.strip()}\n"

    prompt += f"""
Requirements:
1. Generate exactly {num_questions} questions.
2. Ensure each question assesses genuine depth suitable for {interview_type}.
3. The evaluation_criteria must list 2 to 4 bullet points outlining what a strong answer should demonstrate.
4. Output ONLY the JSON array.
"""
    return prompt
