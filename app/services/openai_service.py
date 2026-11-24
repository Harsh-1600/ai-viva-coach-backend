"""OpenAI GPT AI Service module for Student Doubt-Clearing Assistant & Model Answer Generation.

Uses OpenAI's official 'openai' SDK (v1+) to power:
1. Patient, friendly, academic-tutor conversations clarifying student doubts and viva questions (GPT-4o-mini / GPT-4o).
2. Independent, technically rigorous model answers with simple analogies, deep technical explanations,
   code examples with line-by-line breakdowns, and key points to remember for viva exams.
"""

import json
import logging
import re
from typing import List, Optional, Any, Dict
from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError, APITimeoutError, InternalServerError

from app.config import get_settings
from app.schemas import DoubtChatMessage

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class OpenAIServiceError(Exception):
    """Base exception for OpenAI GPT Service errors."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OpenAIConfigurationError(OpenAIServiceError):
    """Raised when OpenAI service is not properly configured."""
    def __init__(self, message: str = "OpenAI API key is not configured."):
        super().__init__(message, status_code=503)


class OpenAIRateLimitError(OpenAIServiceError):
    """Raised when OpenAI API rate limit or quota is exceeded."""
    def __init__(self, message: str = "OpenAI service rate limit reached. Please wait a moment and try again."):
        super().__init__(message, status_code=429)


class OpenAITimeoutError(OpenAIServiceError):
    """Raised when OpenAI request times out."""
    def __init__(self, message: str = "OpenAI request timed out. Please try again."):
        super().__init__(message, status_code=504)


# Backwards compatibility aliases
GeminiServiceError = OpenAIServiceError
GeminiConfigurationError = OpenAIConfigurationError
GeminiRateLimitError = OpenAIRateLimitError
GeminiTimeoutError = OpenAITimeoutError


# ==============================================================================
# Friendly Education AI Assistant & Academic Tutor System Instructions
# ==============================================================================

FRIENDLY_GPT_ASSISTANT_SYSTEM_INSTRUCTION = """You are a warm, helpful, knowledgeable, and genuinely friendly general-purpose conversational AI assistant powered by OpenAI GPT (ChatGPT).
You behave like a natural, thoughtful, and intelligent conversation partner with a warm and supportive personality. You converse comfortably across everyday casual chatting, general knowledge, entertainment, movies, sports, hobbies, personal advice, decision-making, opinions, and learning—including academic subjects, programming, mathematics, career guidance, and viva preparation.

FRIENDLY IDENTITY & NATURAL CONVERSATION STYLE:
- Communicate naturally, warmly, and authentically as a real human friend would.
- Never use robotic formulas, repetitive canned greetings, or boilerplate self-introductions like "I am your friendly AI Education Assistant...".
- Do NOT force every conversation back into studying, coursework, or exams. If the user wants to talk about cricket, movies, jokes, technology, or feeling bored, chat about that naturally with genuine enthusiasm!
- If a study topic context is present, use it only to understand ambiguous academic questions; NEVER impose it on casual chit-chat, hobbies, entertainment, or unrelated queries.
- When the user shares feelings (boredom, nervousness, stress, excitement), respond with empathy, warmth, and supportive engagement.

QUERY HANDLING GUIDELINES:
1. Casual Greetings & Check-ins:
   - For simple greetings ("Hi", "Hey", "Hello"), respond with a short, warm greeting (e.g., "Hey! How's it going?").
   - For check-ins ("How are you?", "What are you doing?"), respond conversationally and naturally (e.g., "I'm good, thanks! How about you?" or "Just here and ready to chat! What's on your mind today?").
   - For gratitude ("Thanks", "Thank you"), respond simply and warmly ("You're welcome!").
   - For friendship inquiries, respond enthusiastically and kindly.
2. Casual Topics, Entertainment, Sports & General Knowledge:
   - Boredom: Suggest fun activities, interesting facts, trivia, or engaging conversation.
   - Sports & Hobbies: Engage enthusiastically in discussions about cricket, football, movies, music, or pop culture.
   - General Curiosity & Science: Share fascinating, mind-bending facts (e.g. space, neutron stars, astronomy, history) in an accessible, exciting way.
   - Advice & Decision Making: Offer balanced perspectives, pros and cons, or helpful frameworks (like 10/10/10 rule).
   - Opinions: Provide thoughtful, well-reasoned, and open-minded viewpoints (e.g. on learning programming, technology trends, AI).
   - Humor: Share lighthearted, clever, clean jokes when asked.
3. Multi-turn Follow-up Continuity:
   - When the user asks "Tell me more", "Okay, tell me more", "Can you explain further?", or "What about the second one?", look at the conversation history and seamlessly continue discussing that specific topic. Never ask the user to repeat themselves.
4. Simple Questions:
   - Provide a concise, direct answer in 1-3 clear sentences without forced headings, bullet lists, or essays.
5. In-Depth / Technical / Academic Explanations:
   - When the user specifically asks for detailed explanations, study help, or programming concepts, provide clear, accurate explanations with practical examples, code snippets, and key points directly without introductory fluff.
   - Core concepts for detailed technical explanations:
     * Direct Answer: Answer the core technical question authoritatively.
     * Simple Explanation: Plain-language explanation using intuitive analogies.
     * Detailed Explanation: In-depth technical breakdown and conceptual mechanics.
     * Practical Example & Code: Concrete scenarios or clean code snippets with explanations.
     * Important Points to Remember: High-yield key points for review.
6. Problem Solving & Math:
   - Show clear, correct step-by-step calculations and verify results.
7. Programming & Code:
   - Provide clean, correct, well-explained code snippets with logic breakdowns when relevant.

PEDAGOGICAL & RELIABILITY DIRECTIVES:
- Ambiguity: If a question is ambiguous, identify possible interpretations and ask for clarification.
- Independence from Student Errors: Never adopt or validate student errors; provide the correct solution independently.
- No Hallucinations: Never fabricate facts, benchmarks, fake syntax, calculations, or citations. Acknowledge uncertainty transparently instead of inventing facts.
"""

ACADEMIC_TUTOR_SYSTEM_INSTRUCTION = FRIENDLY_GPT_ASSISTANT_SYSTEM_INSTRUCTION

GPT_MODEL_ANSWER_SYSTEM_PROMPT = """You are a senior university professor and technical viva examiner.
Your goal is to generate an authoritative, comprehensive model answer and educational explanation for an interview or viva question.

Mandatory Educational Principles:
1. Target the Actual Question: Answer the exact question asked with technical precision rather than producing generic topic overviews.
2. Independent Formulation: Formulate the model answer completely independently of any submitted student response. Even if the student's answer was incorrect, incomplete, or absent, the model answer must represent the gold standard correct solution. NEVER incorporate or validate student errors into the model answer.
3. Structure & Schema: You must return ONLY a valid JSON object matching the requested schema with all 5 core educational components:
   - "correct_answer": Direct, authoritative model answer answering the exact question asked.
   - "simple_explanation": Intuitive plain-language explanation with a relatable real-world analogy.
   - "detailed_explanation": Deep technical breakdown of mechanisms, internal architecture, and trade-offs appropriate to the topic and difficulty level.
   - "example": Concrete practical scenario or correct code snippet with step-by-step line explanations. Avoid inventing execution results.
   - "key_points_to_remember": 3 to 5 high-yield revision bullet points for viva preparation.
   - "disclaimer": Standard revision disclaimer.
4. Accuracy & No Fabrication: Never fabricate facts, benchmarks, citations, or execution outputs.
5. Ambiguity Handling: If the question prompt is inherently ambiguous or underspecified, note the primary interpretations within the detailed explanation.
"""


# ==============================================================================
# JSON Parsing Helper
# ==============================================================================

def extract_json_payload(text: str) -> Any:
    """Extracts and parses JSON object or array from markdown-fenced or raw text."""
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
    clean_text = clean_text.strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r"(\{[\s\S]*\})", clean_text)
        if match:
            return json.loads(match.group(1))
        raise OpenAIServiceError("Failed to parse valid JSON from OpenAI model answer response.", status_code=502)


# ==============================================================================
# Mock Client for Offline Development & Automated Testing
# ==============================================================================

class MockChatMessage:
    """Mock message object mirroring OpenAI chat completion message."""
    def __init__(self, content: str):
        self.content = content


class MockChatChoice:
    """Mock choice object mirroring OpenAI chat completion choice."""
    def __init__(self, content: str):
        self.message = MockChatMessage(content)


class MockChatCompletion:
    """Mock chat completion response mirroring OpenAI SDK response."""
    def __init__(self, content: str):
        self.choices = [MockChatChoice(content)]


class MockChatCompletions:
    """Mock chat.completions namespace mirroring OpenAI SDK."""
    def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, **kwargs) -> MockChatCompletion:
        prompt_text = ""
        system_instr = ""
        contents_adapted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                if "[Current Study Context:" in content:
                    prompt_text += " " + content
                    contents_adapted.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
                    contents_adapted.append({
                        "role": "model",
                        "parts": [{"text": "Understood! I'll keep this topic in mind for relevant questions, while chatting naturally about whatever else the user wants to discuss."}]
                    })
                else:
                    system_instr += " " + content
            else:
                prompt_text += " " + content
                contents_adapted.append({
                    "role": "model" if role in ("assistant", "model") else "user",
                    "parts": [{"text": content}]
                })

        clean = prompt_text.lower()

        # Handle specific simulated error triggers for tests
        if "simulated_rate_limit" in clean:
            raise OpenAIRateLimitError("OpenAI API rate limit reached. Please wait a few moments and try your question again.")
        if "simulated_server_error" in clean or "simulated_model_answer_fail" in clean:
            raise OpenAIServiceError("OpenAI internal server error.", status_code=502)
        if "simulated_timeout" in clean:
            raise OpenAITimeoutError("OpenAI request timed out. Please try again.")
        if "simulated_invalid_key" in clean or "simulated_auth_error" in clean:
            raise OpenAIConfigurationError("Invalid or unauthorized OpenAI API key.")
        if "simulated_malformed_json" in clean:
            return MockChatCompletion("NOT_VALID_JSON_FORMAT")
        if "simulated_empty_response" in clean:
            return MockChatCompletion("")

        # Create mock config with system_instruction
        mock_config = type("MockConfig", (), {"system_instruction": system_instr.strip()})()

        # Generate response using MockGeminiModels
        from app.services.gemini_service import MockGeminiModels
        mock_models = MockGeminiModels()
        gemini_res = mock_models.generate_content(
            model="mock-gpt",
            contents=contents_adapted if contents_adapted else prompt_text,
            config=mock_config
        )
        reply_text = gemini_res.text
        return MockChatCompletion(reply_text)


class MockOpenAIClient:
    """Mock OpenAI Client providing `.chat.completions.create(...)` for offline testing."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "mock"
        self.chat = type("MockChat", (), {"completions": MockChatCompletions()})()


# ==============================================================================
# Client Factory
# ==============================================================================

def get_openai_client(custom_client: Optional[Any] = None) -> Any:
    """Returns an initialized OpenAI client or MockOpenAIClient for testing/offline use.

    Args:
        custom_client: Optional client override (e.g. for mock injection in tests).

    Returns:
        OpenAI or MockOpenAIClient instance.
    """
    if custom_client is not None:
        return custom_client

    settings = get_settings()
    api_key = (settings.OPENAI_API_KEY or "").strip()

    if not api_key:
        raise OpenAIConfigurationError("OPENAI_API_KEY environment variable is not configured.")

    if api_key.lower() in ("mock", "demo", "test"):
        logger.info("Using MockOpenAIClient for offline simulation/testing.")
        return MockOpenAIClient(api_key=api_key)

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise OpenAIConfigurationError(f"Failed to initialize OpenAI client: {str(e)}")


# ==============================================================================
# Service Methods
# ==============================================================================

def ask_gpt_doubt(
    message: str,
    history: Optional[List[DoubtChatMessage]] = None,
    context: Optional[str] = None,
    custom_client: Optional[Any] = None,
) -> str:
    """Sends a student doubt/question to OpenAI GPT with conversation history.

    Args:
        message: The student's current question or doubt.
        history: Optional list of previous DoubtChatMessage turns.
        context: Optional topic or viva question context to seed understanding.
        custom_client: Optional client override for testing.

    Returns:
        str: Patient academic tutor explanation.

    Raises:
        OpenAIConfigurationError: If API key is not configured.
        OpenAIRateLimitError: If rate limits or quotas are hit.
        OpenAIServiceError: If OpenAI API fails.
    """
    if not message or not message.strip():
        raise OpenAIServiceError("Question message cannot be empty.", status_code=400)

    clean_message = message.strip()
    if len(clean_message) > 5000:
        raise OpenAIServiceError("Question message is too long (maximum 5,000 characters).", status_code=400)

    settings = get_settings()
    model_name = settings.OPENAI_MODEL or "gpt-4o-mini"
    client = get_openai_client(custom_client=custom_client)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": FRIENDLY_GPT_ASSISTANT_SYSTEM_INSTRUCTION}
    ]

    # If context is provided, attach as background guidance
    if context and context.strip():
        messages.append({
            "role": "system",
            "content": f"[Current Study Context: {context.strip()} (Note: Only use this context if the user's message is directly asking about it. If the user asks about other topics, casual conversation, entertainment, or general questions, respond to their actual question naturally without forcing this context.)]"
        })

    # Append conversation history
    if history:
        for turn in history:
            role = "assistant" if turn.role in ("assistant", "model") else "user"
            messages.append({"role": role, "content": turn.content})

    # Append current user message
    messages.append({"role": "user", "content": clean_message})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
        )

        if not response or not response.choices or not response.choices[0].message.content:
            raise OpenAIServiceError("OpenAI returned an empty response. Please try rephrasing your question.")

        return response.choices[0].message.content

    except (OpenAIServiceError, OpenAIConfigurationError, OpenAIRateLimitError, OpenAITimeoutError):
        raise

    except RateLimitError as e:
        logger.warning(f"OpenAI API rate limit exceeded: {e}")
        raise OpenAIRateLimitError("OpenAI API rate limit reached. Please wait a few moments and try your question again.")

    except AuthenticationError as e:
        logger.error(f"OpenAI API authentication failed: {e}")
        raise OpenAIConfigurationError("Invalid or unauthorized OpenAI API key.")

    except APITimeoutError as e:
        logger.error(f"OpenAI API timed out: {e}")
        raise OpenAITimeoutError("OpenAI AI request timed out. Please try again.")

    except InternalServerError as e:
        logger.error(f"OpenAI API internal server error: {e}")
        raise OpenAIServiceError("OpenAI internal server error.", status_code=502)

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise OpenAIServiceError(f"OpenAI AI service error: {str(e)}", status_code=502)

    except Exception as e:
        logger.exception(f"Unexpected error communicating with OpenAI: {e}")
        raise OpenAIServiceError(f"Failed to communicate with OpenAI assistant: {str(e)}", status_code=502)


def generate_gpt_model_answer(
    question_text: str,
    subject: str = "Computer Science",
    topic: str = "General",
    difficulty: str = "Medium",
    target_level: str = "Intermediate",
    candidate_answer: Optional[str] = None,
    custom_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generates an authoritative, independent model answer and educational explanation using OpenAI GPT.

    Evaluates the question independently of whatever the candidate submitted.

    Args:
        question_text: Original interview or viva question prompt.
        subject: Subject domain (e.g. Java, Operating Systems, DBMS).
        topic: Specific subtopic (e.g. Polymorphism, Deadlocks).
        difficulty: Question difficulty (Easy, Medium, Hard).
        target_level: Target candidate level (Beginner, Intermediate, Advanced).
        candidate_answer: Optional submitted candidate response.
        custom_client: Optional client override for testing.

    Returns:
        Dict with keys:
            - correct_answer (str)
            - simple_explanation (str)
            - detailed_explanation (str)
            - example (Optional[str])
            - key_points_to_remember (List[str])
            - provider (str): "OpenAI GPT"
            - model (str)
            - is_generated (bool)
            - disclaimer (str)
            - error_message (None)

    Raises:
        OpenAIConfigurationError: If OpenAI is not configured.
        OpenAIRateLimitError: If OpenAI rate limits are hit.
        OpenAIServiceError: If OpenAI fails to generate an answer.
    """
    if not question_text or not question_text.strip():
        raise OpenAIServiceError("Question prompt cannot be empty.", status_code=400)

    clean_question = question_text.strip()
    if len(clean_question) > 5000:
        raise OpenAIServiceError("Question prompt is too long (maximum 5,000 characters).", status_code=400)

    settings = get_settings()
    model_name = settings.OPENAI_MODEL or "gpt-4o-mini"
    client = get_openai_client(custom_client=custom_client)

    candidate_context_note = ""
    if candidate_answer and candidate_answer.strip():
        candidate_context_note = (
            f"\n[Submitted Candidate Response (Provided for evaluation context ONLY; "
            f"NEVER copy, repeat, or adopt any errors, gaps, or misconceptions from this into the model answer)]:\n"
            f"\"{candidate_answer.strip()}\"\n"
        )

    user_prompt = (
        f"Generate a detailed, authoritative model answer and educational explanation for the following viva question.\n\n"
        f"Question: \"{clean_question}\"\n"
        f"Subject / Domain: {subject or 'Computer Science'}\n"
        f"Topic: {topic or 'Technical Interview'}\n"
        f"Difficulty: {difficulty}\n"
        f"Target Level: {target_level}\n"
        f"{candidate_context_note}\n"
        f"Instructions:\n"
        f"1. Target the actual question directly rather than generating generic textbook filler.\n"
        f"2. Formulate the model answer completely independently of any candidate response.\n"
        f"3. Do NOT copy the student's wrong answer or validate misconceptions into the model answer.\n"
        f"4. Provide an authoritative, technically rigorous direct model answer (`correct_answer`).\n"
        f"5. Explain the concept in simple language with an everyday analogy (`simple_explanation`).\n"
        f"6. Provide an in-depth technical explanation suited to {difficulty} level detailing mechanics and trade-offs (`detailed_explanation`).\n"
        f"7. For programming questions, provide clean, correct, compilable code and explicitly explain the logic line-by-line without inventing execution results (`example`).\n"
        f"8. Provide 3-5 high-yield key takeaways for viva revision (`key_points_to_remember`).\n"
        f"9. If the question is ambiguous, note the relevant interpretations and assumptions clearly.\n"
        f"10. Return ONLY a valid JSON object matching the requested schema."
    )

    messages = [
        {"role": "system", "content": GPT_MODEL_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
        )

        if not response or not response.choices or not response.choices[0].message.content:
            raise OpenAIServiceError("OpenAI returned an empty model answer response.")

        raw_data = extract_json_payload(response.choices[0].message.content)
        if not isinstance(raw_data, dict):
            raise OpenAIServiceError("OpenAI model answer was not a JSON dictionary.")

        correct_ans = str(raw_data.get("correct_answer") or "").strip()
        simple_exp = str(raw_data.get("simple_explanation") or "").strip()
        detailed_exp = str(raw_data.get("detailed_explanation") or "").strip()
        example = raw_data.get("example")
        if example is not None:
            example = str(example).strip() or None

        key_points = raw_data.get("key_points_to_remember")
        if isinstance(key_points, list):
            clean_points = [str(p).strip() for p in key_points if str(p).strip()]
        elif isinstance(key_points, str) and key_points.strip():
            clean_points = [key_points.strip()]
        else:
            clean_points = []

        disclaimer = str(
            raw_data.get("disclaimer")
            or "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
        )

        improved_ans = str(raw_data.get("improved_answer") or "").strip() or None

        return {
            "correct_answer": correct_ans,
            "simple_explanation": simple_exp,
            "detailed_explanation": detailed_exp,
            "example": example,
            "key_points_to_remember": clean_points,
            "improved_answer": improved_ans,
            "provider": "OpenAI GPT",
            "model": model_name,
            "is_generated": True,
            "disclaimer": disclaimer,
            "error_message": None,
        }

    except (OpenAIServiceError, OpenAIConfigurationError, OpenAIRateLimitError, OpenAITimeoutError):
        raise

    except RateLimitError as e:
        logger.warning(f"OpenAI API rate limit exceeded during model answer generation: {e}")
        raise OpenAIRateLimitError("OpenAI AI rate limit reached while generating model answer. Please wait a moment.")

    except AuthenticationError as e:
        logger.error(f"OpenAI API authentication failed during model answer generation: {e}")
        raise OpenAIConfigurationError("Invalid or unauthorized OpenAI API key.")

    except APITimeoutError as e:
        logger.error(f"OpenAI API timed out during model answer generation: {e}")
        raise OpenAITimeoutError("OpenAI AI request timed out. Please try again.")

    except InternalServerError as e:
        logger.error(f"OpenAI API internal server error during model answer generation: {e}")
        raise OpenAIServiceError("OpenAI internal server error.", status_code=502)

    except OpenAIError as e:
        logger.error(f"OpenAI API error during model answer generation: {e}")
        raise OpenAIServiceError(f"OpenAI AI service error: {str(e)}", status_code=502)

    except Exception as e:
        logger.exception(f"Unexpected error generating model answer with OpenAI: {e}")
        raise OpenAIServiceError(f"Failed to generate model answer: {str(e)}", status_code=502)
