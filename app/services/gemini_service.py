"""Gemini AI Service module for Student Doubt-Clearing Assistant & Model Answer Generation.

Uses Google's official 'google-genai' SDK to power:
1. Patient, academic-tutor conversations clarifying student doubts and viva questions.
2. Independent, technically rigorous model answers with simple analogies, deep technical explanations,
   code examples with line-by-line breakdowns, and key points to remember for viva exams.
"""

import json
import logging
import re
from typing import List, Optional, Any, Dict
from google import genai
from google.genai import types, errors

from app.config import get_settings
from app.schemas import DoubtChatMessage
from app.services import reference_answers

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class GeminiServiceError(Exception):
    """Base exception for Gemini AI Service errors."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GeminiConfigurationError(GeminiServiceError):
    """Raised when Gemini service is not properly configured."""
    def __init__(self, message: str = "Google Gemini API key is not configured."):
        super().__init__(message, status_code=503)


class GeminiRateLimitError(GeminiServiceError):
    """Raised when Gemini API rate limit or quota is exceeded."""
    def __init__(self, message: str = "Gemini service rate limit reached. Please wait a moment and try again."):
        super().__init__(message, status_code=429)


class GeminiTimeoutError(GeminiServiceError):
    """Raised when Gemini request times out."""
    def __init__(self, message: str = "Gemini request timed out. Please try again."):
        super().__init__(message, status_code=504)


# ==============================================================================
# Friendly Education AI Assistant & Academic Tutor System Instructions
# ==============================================================================

FRIENDLY_EDUCATION_ASSISTANT_SYSTEM_INSTRUCTION = """You are a warm, helpful, knowledgeable, and genuinely friendly general-purpose conversational AI assistant powered by Google Gemini.
You behave like a natural, thoughtful, and intelligent conversation partner (similar to ChatGPT) with a warm and supportive personality. You converse comfortably across everyday casual chatting, general knowledge, entertainment, movies, sports, hobbies, personal advice, decision-making, opinions, and learning—including academic subjects, programming, mathematics, career guidance, and viva preparation.

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

ACADEMIC_TUTOR_SYSTEM_INSTRUCTION = FRIENDLY_EDUCATION_ASSISTANT_SYSTEM_INSTRUCTION

GEMINI_MODEL_ANSWER_SYSTEM_PROMPT = """You are a senior university professor and technical viva examiner.
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
        # Search for first outermost { ... }
        match = re.search(r"(\{[\s\S]*\})", clean_text)
        if match:
            return json.loads(match.group(1))
        raise GeminiServiceError("Failed to parse valid JSON from Gemini model answer response.", status_code=502)


# ==============================================================================
# General Message & Greeting Classifier
# ==============================================================================

def classify_general_or_greeting_message(message: str, history_context: Optional[str] = None) -> Optional[str]:
    """Detects whether a user message is a greeting, pleasantry, conversational check-in,
    friendliness inquiry, emotional check-in, casual conversation topic (boredom, cricket,
    movies, technology, jokes, space, opinions, decision-making, what to learn, etc.),
    or contextual follow-up, and returns a natural, warm, friendly companion response.
    Returns None if the message should be processed as an academic/technical doubt.
    """
    if not message or not message.strip():
        return None

    clean = message.strip().lower()
    clean_no_punct = re.sub(r"[^\w\s]", "", clean).strip()
    words = clean_no_punct.split()

    if not words:
        return None

    hist = (history_context or "").lower()

    # 1. Friendliness feedback / check ("the ai not friendly", "why are you not friendly", "be friendly", "why so serious")
    unfriendly_feedback_phrases = [
        "not friendly", "is not friendly", "ai not friendly", "the ai not friendly",
        "you are not friendly", "why are you not friendly", "be friendly", "be more friendly",
        "why so serious", "why are you so serious", "too serious", "so serious",
        "why are you rude", "why are you cold", "talk friendly", "dont be serious", "don't be serious"
    ]
    if any(p in clean_no_punct for p in unfriendly_feedback_phrases) or (
        "friendly" in clean_no_punct and any(w in clean_no_punct for w in ["not", "be", "why", "more", "less", "try", "make", "become"])
    ):
        return (
            "I'm really sorry if I came across as too formal or serious! I definitely want to be friendly, "
            "approachable, and helpful to you. Let's relax and start fresh—how are you doing today, and what's on your mind? 😊"
        )

    # 2. "Are you friendly?" inquiry
    if "friendly" in clean_no_punct and any(w in clean_no_punct for w in ["are you", "is it", "is this", "can you be"]):
        return (
            "Yes, absolutely! I'm here to be a warm, friendly, and supportive companion whether you're studying, "
            "practicing for interviews, or just having a chat. How are you doing today?"
        )

    # 3. Friendship & Companionship ("can we be friends", "be my friend", "are you my friend", etc.)
    friendship_phrases = [
        "can we be friends", "be my friend", "are you my friend", "can you be my friend",
        "let's be friends", "lets be friends", "want to be friends", "will you be my friend",
        "talk to me like a friend", "be friends", "make friends"
    ]
    if any(p in clean_no_punct for p in friendship_phrases) or (
        any(w in words for w in ["friend", "friends"]) and any(w in clean_no_punct for w in ["be", "are you", "can we", "we", "want", "like"])
    ):
        return (
            "Of course we can! I'd love to be your friend. We can chat, work on your studies together, "
            "or just talk about how your day is going. What are you up to today?"
        )

    # 4. Emotional support / feeling down / stress / loneliness
    lonely_phrases = ["lonely", "alone", "feeling lonely", "sad", "unhappy", "depressed", "feeling down", "crying"]
    if any(w in words for w in lonely_phrases) or any(p in clean_no_punct for p in ["feel lonely", "feel sad", "feeling sad", "i am sad", "im sad", "i am lonely", "im lonely"]):
        return (
            "I'm sorry you're feeling that way. Please know you're not alone—I'm right here with you! "
            "If you want to talk about how you're feeling, chat about something fun, or explore a new topic, "
            "I'm always happy to keep you company. How can I brighten your day? 🌟"
        )

    stress_phrases = ["stressed", "overwhelmed", "exhausted", "burnout", "too much pressure"]
    if any(p in clean_no_punct for p in stress_phrases) or ("stress" in clean_no_punct and any(w in clean_no_punct for w in ["lot", "much", "very", "studies", "exam", "feel"])):
        return (
            "Take a deep breath—it's completely normal to feel stressed, especially with studies and deadlines. "
            "Remember to give yourself credit for how hard you're trying. Would you like to take a quick break, "
            "talk through what's on your mind, or break things down into smaller steps together?"
        )

    # 5. Uplifting / Motivation requests ("tell me something nice", "cheer me up")
    uplifting_phrases = ["tell me something nice", "cheer me up", "motivate me", "inspire me", "say something nice"]
    if any(p in clean_no_punct for p in uplifting_phrases):
        return (
            "Here's something to remember: every expert was once a beginner, and every small step you take "
            "adds up over time. You are capable of learning difficult things, and you're doing better than you think. "
            "Keep going, and believe in yourself! 😊"
        )

    # 6. Compliments to AI ("you are nice", "you are sweet", "you are awesome")
    compliment_phrases = ["you are nice", "you are sweet", "you are awesome", "youre great", "you are great", "good job", "love you", "you are smart", "you are the best", "you rock"]
    if any(p in clean_no_punct for p in compliment_phrases):
        return "Thank you so much! That's really kind of you to say. I'm always happy to be here for you! How can I help you today?"

    # 7. "What are you doing?" / "What are you up to?" (Checked BEFORE identity so it doesn't trigger "who are you")
    doing_phrases = [
        "what are you doing", "what r u doing", "what you doing", "what are u doing",
        "what are you up to", "what r u up to", "what doing"
    ]
    if any(p in clean_no_punct for p in doing_phrases):
        return "Just here and ready to chat! What's on your mind today?"

    # 8. Boredom ("I'm feeling bored. What can I do?", "I'm bored", "feeling bored")
    bored_phrases = [
        "feeling bored", "feel bored", "im bored", "i am bored", "so bored",
        "feeling bored what can i do", "bored what can i do", "i am feeling bored"
    ]
    if any(p in clean_no_punct for p in bored_phrases) or (clean_no_punct == "bored" or clean_no_punct.startswith("bored ")):
        return "Want to try a fun quiz, hear an interesting fact, or just chat? Or we could brainstorm something fun for you to do—like learning a quick new skill, listening to music, or diving into an exciting topic! What sounds fun right now?"

    # 9. Jokes & Humor
    joke_phrases = ["tell me a joke", "tell a joke", "tell me joke", "say a joke", "another joke", "make me laugh", "know any jokes"]
    if any(p in clean_no_punct for p in joke_phrases) or "joke" in clean_no_punct or "jokes" in clean_no_punct:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"

    # 10. Cricket & Sports discussions ("Let's talk about cricket", "talk about cricket", etc.)
    cricket_phrases = ["talk about cricket", "lets talk about cricket", "let's talk about cricket", "chat about cricket", "know about cricket", "like cricket", "play cricket"]
    if any(p in clean_no_punct for p in cricket_phrases) or (
        "cricket" in clean_no_punct and any(w in clean_no_punct for w in ["talk", "chat", "lets", "let", "about", "discuss", "play", "match", "team", "ipl", "t20", "world cup"])
    ):
        return "Sure! Cricket is an amazing game. What's your favorite format—T20, ODIs, or Test cricket? Or is there a favorite team, player, or recent match you want to discuss?"

    # 11. Movies & Cinema discussions ("Do you know about movies?", "talk about movies")
    movie_phrases = [
        "know about movies", "talk about movies", "tell me about movies", "chat about movies",
        "lets talk about movies", "like movies", "recommend a movie", "favorite movie", "favorite movies", "favourite movie", "favourite movies"
    ]
    if any(p in clean_no_punct for p in movie_phrases) or (
        "movie" in clean_no_punct and any(w in clean_no_punct for w in ["talk", "chat", "about", "know", "like", "recommend", "watch", "cinema"])
    ):
        return "Yes, absolutely! I love talking about cinema—whether it's sci-fi, mind-bending thrillers, action blockbusters, classic cinema, or comedies. What kind of movies do you enjoy, or seen anything great recently?"

    # 12. Technology Chat ("Can we talk about technology?", "talk about tech")
    tech_talk_phrases = [
        "talk about technology", "talk about tech", "we talk about technology", "we talk about tech",
        "chat about technology", "chat about tech", "lets talk about technology", "lets talk about tech"
    ]
    if any(p in clean_no_punct for p in tech_talk_phrases):
        return (
            "I'd love to! Technology is moving so fast right now—from artificial intelligence and robotics "
            "to space tech, electric vehicles, and smartphones. Is there a specific tech trend, gadget, or programming topic you're excited about, or should we explore what's new?"
        )

    # 13. Opinion on Learning Programming ("What is your opinion on learning programming?")
    opinion_programming_phrases = [
        "opinion on learning programming", "opinion on programming", "opinion about programming",
        "opinion about learning programming", "what do you think of learning programming",
        "what do you think about learning programming", "what do you think of programming",
        "should i learn programming", "should i learn to code", "is programming hard", "is programming worth learning"
    ]
    if any(p in clean_no_punct for p in opinion_programming_phrases) or (
        "opinion" in clean_no_punct and ("programming" in clean_no_punct or "coding" in clean_no_punct)
    ):
        return (
            "Learning programming is one of the most rewarding skills you can pick up! Beyond writing code, it fundamentally trains you to break down complex problems, think logically, and build things from scratch. "
            "It definitely has a learning curve and can feel frustrating when debugging at first, but the moment your code runs and solves a real problem is super satisfying. Are you thinking about learning a language, or already exploring one?"
        )

    # 14. Space & Astronomy facts ("Explain something interesting about space.", "something about space")
    space_phrases = [
        "interesting about space", "something about space", "tell me about space",
        "explain something interesting about space", "facts about space", "space facts", "talk about space"
    ]
    if any(p in clean_no_punct for p in space_phrases) or (
        "space" in words and any(w in clean_no_punct for w in ["interesting", "fact", "facts", "cool", "explain", "tell me", "something"])
    ):
        return (
            "Space is full of incredible mysteries! For example, neutron stars are so intensely dense that a single teaspoon of their matter would weigh around 6 billion tons on Earth—about the weight of Mount Everest! Also, they can spin up to 700 times per second. Black holes and cosmic nebulae are just as fascinating—is there a particular corner of the universe you're curious about?"
        )

    # 15. General Interesting Facts ("Tell me something interesting.")
    interesting_fact_phrases = [
        "tell me something interesting", "something interesting", "tell me an interesting fact",
        "tell me something cool", "share something interesting", "interesting fact", "cool fact"
    ]
    if any(p in clean_no_punct for p in interesting_fact_phrases):
        return (
            "Here's a mind-blowing fact: a day on Venus is longer than a year on Venus! It takes Venus about 243 Earth days to rotate once on its axis, but only about 225 Earth days to complete an orbit around the Sun. What kind of topics are you usually curious about—space, history, technology, or nature?"
        )

    # 16. Decision Making ("Can you help me make a decision?", "help me decide")
    decision_phrases = [
        "help me make a decision", "make a decision", "help me decide", "help make a decision",
        "cant decide", "can't decide", "help me choose", "making a decision"
    ]
    if any(p in clean_no_punct for p in decision_phrases):
        return "I'd be glad to help! Tell me what options you're weighing. We can break down the pros and cons, look at your priorities and timeline, and find the choice that fits you best. What's on your mind?"

    # 17. What should I learn today? ("What should I learn today?")
    learn_today_phrases = [
        "what should i learn today", "what to learn today", "what should i learn",
        "suggest something to learn", "what can i learn today"
    ]
    if any(p in clean_no_punct for p in learn_today_phrases):
        return (
            "It depends on what mood you're in! If you want something hands-on and practical, you could try learning the basics of Python or Git version control. "
            "If you're feeling curious about science, dive into how black holes work or how quantum computers operate. Or if you're in a creative mindset, exploring UI design or storytelling is a great pick. What direction sounds exciting to you today?"
        )

    # 18. Multi-turn Follow-up Continuity ("Okay, tell me more.", "Tell me more", "elaborate")
    followup_continuation_phrases = [
        "tell me more", "tell more", "okay tell me more", "ok tell me more",
        "can you tell me more", "tell me more about this", "tell me more about that",
        "elaborate more", "go on", "continue", "what else", "tell me further"
    ]
    is_continuation_query = any(p in clean_no_punct for p in followup_continuation_phrases) or (
        clean_no_punct in ("tell me more", "more", "tell more", "go on", "continue")
    )
    if is_continuation_query:
        if any(w in hist for w in ("space", "neutron", "venus", "astronomy", "star", "planet", "black hole")):
            return (
                "Space only gets stranger the deeper you look! For instance, near the event horizon of a black hole, "
                "gravitational time dilation is so intense that time slows down relative to an observer far away. "
                "There are also 'rogue planets' wandering completely alone in the dark between stars without any host sun. "
                "Would you like to hear about black holes, dark matter, or how stars are born?"
            )
        elif "cricket" in hist:
            return (
                "Cricket has so many legendary stories! From thrilling last-over finishes in the IPL and T20 World Cups, "
                "to classic Test rivalries like The Ashes and India vs Pakistan. "
                "Are you more drawn to masterclass batting, fiery fast bowling, or the tactical captaincy decisions that turn a match around?"
            )
        elif any(w in hist for w in ("movie", "movies", "cinema", "film")):
            return (
                "Cinema is full of incredible storytelling! For example, directors like Christopher Nolan use practical effects "
                "and non-linear timelines to bend reality in films like Inception and Interstellar, while classics like The Godfather "
                "show the power of character depth and pacing. Are you in the mood for mind-bending sci-fi, intense thrillers, or feel-good comfort films?"
            )
        elif any(w in hist for w in ("technology", "tech")):
            return (
                "One of the biggest shifts right now is how AI, edge computing, and robotics are intersecting. "
                "We're moving from systems that just predict words to models that can write code, control robotic arms, and discover new scientific proteins. "
                "Which branch of technology impacts your daily life or interests you the most?"
            )
        elif any(w in hist for w in ("decision", "decide")):
            return (
                "A great mental model for tough decisions is the 10/10/10 rule: how will you feel about this choice in 10 minutes, 10 months, and 10 years? "
                "Another is regret minimization—looking back in a few years, which path would you regret not trying? What are the specific options you're choosing between right now?"
            )
        elif "bored" in hist:
            return (
                "Here's a fun mini-challenge: try a 5-minute riddle, learn 5 words in a new language, or explore a random Wikipedia rabbit hole! "
                "Or if you want, tell me what hobbies you enjoy and I can give you a quick creative project to try right now."
            )
        elif any(w in hist for w in ("programming", "coding", "code")):
            return (
                "A great way to continue with programming is focusing on small, tangible projects rather than just reading syntax. "
                "For example, building a simple todo app, a weather bot, or a mini web scraper gives you immediate feedback and teaches debugging quickly. "
                "Would you like to start with Python, web development, or something else?"
            )
        elif "photosynthesis" in hist:
            return (
                "Photosynthesis actually happens in two main stages inside plant chloroplasts: the Light-Dependent Reactions "
                "(which capture photons and produce ATP and Oxygen), and the Calvin Cycle (the light-independent reactions that use ATP to fix Carbon Dioxide into Glucose). "
                "Would you like to look at either stage in more detail?"
            )
        elif "recursion" in hist:
            return (
                "Every recursive function builds up stack frames in memory. If the base case is missing or not reached, you get a StackOverflowError! "
                "To optimize this, some languages use tail call optimization, and in algorithms, memoization is used to cache results so you don't recompute the same subproblems. "
                "Would you like an example of memoized recursion?"
            )
        elif "inheritance" in hist or "java" in hist:
            return (
                "In Java, inheritance works hand-in-hand with polymorphism and abstraction. While classes use 'extends' for single inheritance, "
                "Java allows a class to implement multiple 'interfaces' to achieve multiple inheritance of type, avoiding the diamond problem. "
                "Would you like to see how interfaces or abstract classes build on this?"
            )
        elif "polymorphism" in hist:
            return (
                "Polymorphism is powerful because it decouples your application architecture: callers interact with high-level interfaces, "
                "while low-level implementations can be swapped without touching client code. In Java, this is backed at the bytecode level by 'invokevirtual' instructions and vtables."
            )
        else:
            return "Happy to dive deeper! Which aspect of that would you like to explore further, or should I walk you through the key details?"

    # 19. Conversational check-ins ("how are you?", "what's up?", etc.)
    checkin_phrases = [
        "how are you", "how are u", "how r u", "howre you", "how do you do",
        "how is it going", "hows it going", "whats up", "what is up",
        "how are you doing", "how r u doing"
    ]
    greeting_tokens = {
        "hai", "hello", "hi", "hey", "hiya", "howdy", "hola", "sup", "yo", "heya",
        "vanakkam", "namaste", "namaskar", "namaskaram", "greetings"
    }

    has_greeting = (words[0] in greeting_tokens) or any(
        clean_no_punct.startswith(g) for g in ["good morning", "good afternoon", "good evening", "good day"]
    )
    has_checkin = any(p in clean_no_punct for p in checkin_phrases)

    if has_greeting and has_checkin:
        return "Hey! I'm good, thanks. How about you?"

    if has_checkin:
        return "I'm good, thanks! How about you?"

    # 20. Specific time-of-day greetings
    if clean_no_punct.startswith("good morning"):
        return "Good morning! Hope you're having a wonderful start to your day. How's everything going?"
    if clean_no_punct.startswith("good afternoon"):
        return "Good afternoon! Hope your day is going smoothly. What are you working on today?"
    if clean_no_punct.startswith("good evening"):
        return "Good evening! Hope you had a great day. How can I help you tonight?"
    if any(clean_no_punct.startswith(p) for p in ["good night", "goodnight"]):
        return "Good night! Rest well and recharge. Talk to you soon!"

    # 21. Gratitude & Thank-you
    gratitude_phrases = [
        "thank you", "thanks", "thx", "thank u", "thank you so much",
        "thanks a lot", "appreciate it", "many thanks"
    ]
    if any(p in clean_no_punct for p in gratitude_phrases) and len(words) <= 7:
        return "You're welcome!"

    # 22. Identity & Capabilities (Strictly when not "what are you doing" which is already handled above)
    identity_phrases = [
        "who are you", "who r u", "what are you", "what is your name",
        "tell me about yourself", "who created you"
    ]
    if any(clean_no_punct == p or clean_no_punct.startswith(p + " ") for p in identity_phrases):
        return "I'm an AI assistant here to help you learn, prepare for interviews, and answer questions. What can I help you with?"

    if any(clean_no_punct == p or clean_no_punct.endswith(p) for p in ["can you help me", "help me"]):
        return "Hey! Yes, of course. What do you need help with?"

    if any(p in clean_no_punct for p in ["what can you help me with", "what can you do", "what do you do"]):
        return (
            "I can help you with academic subjects, programming and debugging, study plans, "
            "placement and interview preparation, and general learning questions. What would you like to work on?"
        )

    # 23. Pure Greetings (hai, hello, hi, hey, etc.)
    greeting_phrases = [
        "hello there", "hi there", "hey there", "hai there",
        "hello coach", "hi coach", "hey coach", "hai coach",
        "hello friend", "hi friend", "hey friend", "hai friend",
        "hello sir", "hi sir", "hey sir", "hai sir",
        "hello assistant", "hi assistant", "hey assistant", "hai assistant",
        "hello bot", "hi bot", "hey bot", "hai bot",
        "hello gemini", "hi gemini", "hey gemini", "hai gemini",
    ]
    is_greeting_phrase = any(clean_no_punct.startswith(p) for p in greeting_phrases)
    is_pure_greeting_word = len(words) <= 3 and any(w in greeting_tokens for w in words)
    is_single_greeting = words[0] in greeting_tokens and len(words) <= 3

    if is_greeting_phrase or is_pure_greeting_word or is_single_greeting:
        return "Hey! How's it going?"

    # 24. Farewells & Goodbye
    farewell_phrases = [
        "bye", "goodbye", "good bye", "see you", "see ya", "take care",
        "tc", "tata", "have a good day"
    ]
    if any(clean_no_punct.startswith(p) for p in farewell_phrases) and len(words) <= 5:
        return "Take care! Have a good one."

    # 25. Affirmations & Short Acknowledgments
    affirmations = {
        "ok", "okay", "cool", "nice", "awesome", "great", "got it",
        "understood", "sure", "alright", "all right", "perfect"
    }
    if clean_no_punct in affirmations or (len(words) <= 2 and clean_no_punct in affirmations):
        return "Got it! Let me know if you need anything else."

    # 26. Motivation, Exam Anxiety & General Advice
    viva_tips_keywords = [
        "nervous", "scared", "anxious", "fear", "viva tips", "interview tips",
        "how to prepare", "how should i prepare", "exam tips", "tips for viva",
        "how to clear viva", "confidence", "viva advice", "tips for interview", "motivation"
    ]
    if any(k in clean_no_punct for k in viva_tips_keywords):
        return (
            "It's completely normal to feel nervous before an exam or interview. Here are a few practical tips:\n\n"
            "1. State the main idea first in 1-2 direct sentences.\n"
            "2. Use a simple real-world analogy to show your understanding.\n"
            "3. Take a breath and structure your answer before speaking.\n"
            "4. If you don't know a detail, be honest and explain what you do know about related concepts.\n\n"
            "Would you like to practice a question together?"
        )

    return None

    clean = message.strip().lower()
    clean_no_punct = re.sub(r"[^\w\s]", "", clean).strip()
    words = clean_no_punct.split()

    if not words:
        return None

    # 1. Friendliness feedback / check ("the ai not friendly", "why are you not friendly", "be friendly", "why so serious")
    unfriendly_feedback_phrases = [
        "not friendly", "is not friendly", "ai not friendly", "the ai not friendly",
        "you are not friendly", "why are you not friendly", "be friendly", "be more friendly",
        "why so serious", "why are you so serious", "too serious", "so serious",
        "why are you rude", "why are you cold", "talk friendly", "dont be serious", "don't be serious"
    ]
    if any(p in clean_no_punct for p in unfriendly_feedback_phrases) or (
        "friendly" in clean_no_punct and any(w in clean_no_punct for w in ["not", "be", "why", "more", "less", "try", "make", "become"])
    ):
        return (
            "I'm really sorry if I came across as too formal or serious! I definitely want to be friendly, "
            "approachable, and helpful to you. Let's relax and start fresh—how are you doing today, and what's on your mind? 😊"
        )

    # 2. "Are you friendly?" inquiry
    if "friendly" in clean_no_punct and any(w in clean_no_punct for w in ["are you", "is it", "is this", "can you be"]):
        return (
            "Yes, absolutely! I'm here to be a warm, friendly, and supportive companion whether you're studying, "
            "practicing for interviews, or just having a chat. How are you doing today?"
        )

    # 3. Friendship & Companionship ("can we be friends", "be my friend", "are you my friend", etc.)
    friendship_phrases = [
        "can we be friends", "be my friend", "are you my friend", "can you be my friend",
        "let's be friends", "lets be friends", "want to be friends", "will you be my friend",
        "talk to me like a friend", "be friends", "make friends"
    ]
    if any(p in clean_no_punct for p in friendship_phrases) or (
        any(w in words for w in ["friend", "friends"]) and any(w in clean_no_punct for w in ["be", "are you", "can we", "we", "want", "like"])
    ):
        return (
            "Of course we can! I'd love to be your friend. We can chat, work on your studies together, "
            "or just talk about how your day is going. What are you up to today?"
        )

    # 4. Emotional support / feeling down / stress / loneliness
    lonely_phrases = ["lonely", "alone", "feeling lonely", "sad", "unhappy", "depressed", "feeling down", "crying"]
    if any(w in words for w in lonely_phrases) or any(p in clean_no_punct for p in ["feel lonely", "feel sad", "feeling sad", "i am sad", "im sad", "i am lonely", "im lonely"]):
        return (
            "I'm sorry you're feeling that way. Please know you're not alone—I'm right here with you! "
            "If you want to talk about how you're feeling, chat about something fun, or explore a new topic, "
            "I'm always happy to keep you company. How can I brighten your day? 🌟"
        )

    stress_phrases = ["stressed", "overwhelmed", "exhausted", "burnout", "too much pressure"]
    if any(p in clean_no_punct for p in stress_phrases) or ("stress" in clean_no_punct and any(w in clean_no_punct for w in ["lot", "much", "very", "studies", "exam", "feel"])):
        return (
            "Take a deep breath—it's completely normal to feel stressed, especially with studies and deadlines. "
            "Remember to give yourself credit for how hard you're trying. Would you like to take a quick break, "
            "talk through what's on your mind, or break things down into smaller steps together?"
        )

    # 5. Uplifting / Motivation requests ("tell me something nice", "cheer me up")
    uplifting_phrases = ["tell me something nice", "cheer me up", "motivate me", "inspire me", "say something nice"]
    if any(p in clean_no_punct for p in uplifting_phrases):
        return (
            "Here's something to remember: every expert was once a beginner, and every small effort you make today "
            "adds up over time. You are capable of learning difficult things, and you're doing better than you think. "
            "Keep going, and believe in yourself! 😊"
        )

    # 6. Compliments to AI ("you are nice", "you are sweet", "you are awesome")
    compliment_phrases = ["you are nice", "you are sweet", "you are awesome", "youre great", "you are great", "good job", "love you", "you are smart", "you are the best", "you rock"]
    if any(p in clean_no_punct for p in compliment_phrases):
        return "Thank you so much! That's really kind of you to say. I'm always happy to be here for you! How can I help you today?"

    # 7. Conversational check-ins ("how are you?", "what's up?", etc.)
    checkin_phrases = [
        "how are you", "how are u", "how r u", "howre you", "how do you do",
        "how is it going", "hows it going", "whats up", "what is up",
        "how are you doing", "how r u doing"
    ]
    greeting_tokens = {
        "hai", "hello", "hi", "hey", "hiya", "howdy", "hola", "sup", "yo", "heya",
        "vanakkam", "namaste", "namaskar", "namaskaram", "greetings"
    }

    has_greeting = (words[0] in greeting_tokens) or any(
        clean_no_punct.startswith(g) for g in ["good morning", "good afternoon", "good evening", "good day"]
    )
    has_checkin = any(p in clean_no_punct for p in checkin_phrases)

    if has_greeting and has_checkin:
        return "Hey! I'm good, thanks. How about you?"

    if has_checkin:
        return "I'm good, thanks! How about you?"

    # 8. Specific time-of-day greetings
    if clean_no_punct.startswith("good morning"):
        return "Good morning! Hope you're having a wonderful start to your day. How's everything going?"
    if clean_no_punct.startswith("good afternoon"):
        return "Good afternoon! Hope your day is going smoothly. What are you working on today?"
    if clean_no_punct.startswith("good evening"):
        return "Good evening! Hope you had a great day. How can I help you tonight?"
    if any(clean_no_punct.startswith(p) for p in ["good night", "goodnight"]):
        return "Good night! Rest well and recharge. Talk to you soon!"

    # 9. Gratitude & Thank-you
    gratitude_phrases = [
        "thank you", "thanks", "thx", "thank u", "thank you so much",
        "thanks a lot", "appreciate it", "many thanks"
    ]
    if any(p in clean_no_punct for p in gratitude_phrases) and len(words) <= 7:
        return "You're welcome!"

    # 10. Identity & Capabilities
    identity_phrases = [
        "who are you", "who r u", "what are you", "what is your name",
        "tell me about yourself", "who created you"
    ]
    if any(p in clean_no_punct for p in identity_phrases):
        return "I'm an AI assistant here to help you learn, prepare for interviews, and answer questions. What can I help you with?"

    if any(clean_no_punct == p or clean_no_punct.endswith(p) for p in ["can you help me", "help me"]):
        return "Hey! Yes, of course. What do you need help with?"

    if any(p in clean_no_punct for p in ["what can you help me with", "what can you do", "what do you do"]):
        return (
            "I can help you with academic subjects, programming and debugging, study plans, "
            "placement and interview preparation, and general learning questions. What would you like to work on?"
        )

    # 11. Pure Greetings (hai, hello, hi, hey, etc.)
    greeting_phrases = [
        "hello there", "hi there", "hey there", "hai there",
        "hello coach", "hi coach", "hey coach", "hai coach",
        "hello friend", "hi friend", "hey friend", "hai friend",
        "hello sir", "hi sir", "hey sir", "hai sir",
        "hello assistant", "hi assistant", "hey assistant", "hai assistant",
        "hello bot", "hi bot", "hey bot", "hai bot",
        "hello gemini", "hi gemini", "hey gemini", "hai gemini",
    ]
    is_greeting_phrase = any(clean_no_punct.startswith(p) for p in greeting_phrases)
    is_pure_greeting_word = len(words) <= 3 and any(w in greeting_tokens for w in words)
    is_single_greeting = words[0] in greeting_tokens and len(words) <= 3

    if is_greeting_phrase or is_pure_greeting_word or is_single_greeting:
        return "Hey! How's it going?"

    # 12. Farewells & Goodbye
    farewell_phrases = [
        "bye", "goodbye", "good bye", "see you", "see ya", "take care",
        "tc", "tata", "have a good day"
    ]
    if any(clean_no_punct.startswith(p) for p in farewell_phrases) and len(words) <= 5:
        return "Take care! Have a good one."

    # 13. Affirmations & Short Acknowledgments
    affirmations = {
        "ok", "okay", "cool", "nice", "awesome", "great", "got it",
        "understood", "sure", "alright", "all right", "perfect"
    }
    if clean_no_punct in affirmations or (len(words) <= 2 and clean_no_punct.startswith("ok")):
        return "Got it! Let me know if you need anything else."

    # 14. Motivation, Exam Anxiety & General Advice
    viva_tips_keywords = [
        "nervous", "scared", "anxious", "fear", "viva tips", "interview tips",
        "how to prepare", "how should i prepare", "exam tips", "tips for viva",
        "how to clear viva", "confidence", "viva advice", "tips for interview", "motivation"
    ]
    if any(k in clean_no_punct for k in viva_tips_keywords):
        return (
            "It's completely normal to feel nervous before an exam or interview. Here are a few practical tips:\n\n"
            "1. State the main idea first in 1-2 direct sentences.\n"
            "2. Use a simple real-world analogy to show your understanding.\n"
            "3. Take a breath and structure your answer before speaking.\n"
            "4. If you don't know a detail, be honest and explain what you do know about related concepts.\n\n"
            "Would you like to practice a question together?"
        )

    # 15. Unrelated Jokes / Casual inquiries
    if "joke" in clean_no_punct:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"

    return None


# ==============================================================================
# Mock Client for Offline Development & Automated Testing
# ==============================================================================

class MockGeminiResponse:
    """Mock response object mirroring the google.genai response structure."""
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text


class MockGeminiModels:
    """Mock models service for offline testing without network access."""
    def generate_content(self, model: str, contents: Any, config: Optional[Any] = None) -> MockGeminiResponse:
        # Extract the prompt text and last user prompt from contents
        prompt_text = ""
        last_user_prompt = ""
        user_turns_text = ""
        user_turn_count = 0
        if isinstance(contents, str):
            prompt_text = contents
            last_user_prompt = contents
            user_turns_text = contents
            user_turn_count = 1
        elif isinstance(contents, list):
            for item in contents:
                if isinstance(item, types.Content):
                    item_role = getattr(item, "role", "")
                    curr_parts = []
                    for part in item.parts:
                        if hasattr(part, "text") and part.text:
                            curr_parts.append(part.text)
                            prompt_text += " " + part.text
                    if item_role == "user" and curr_parts:
                        text_joined = " ".join(curr_parts)
                        if not text_joined.strip().startswith("["):
                            last_user_prompt = text_joined
                            user_turns_text += " " + text_joined
                            user_turn_count += 1
                elif isinstance(item, str):
                    prompt_text += " " + item
                    if not item.strip().startswith("["):
                        last_user_prompt = item
                        user_turns_text += " " + item
                        user_turn_count += 1
                elif isinstance(item, dict) and "parts" in item:
                    item_role = item.get("role", "")
                    curr_parts = []
                    for part in item["parts"]:
                        if isinstance(part, dict) and "text" in part:
                            curr_parts.append(part["text"])
                            prompt_text += " " + part["text"]
                    if item_role == "user" and curr_parts:
                        text_joined = " ".join(curr_parts)
                        if not text_joined.strip().startswith("["):
                            last_user_prompt = text_joined
                            user_turns_text += " " + text_joined
                            user_turn_count += 1

        clean = prompt_text.lower()
        clean_user = (last_user_prompt or prompt_text).lower()
        clean_user_hist = (user_turns_text or last_user_prompt or prompt_text).lower()
        system_instr = ""
        if config and hasattr(config, "system_instruction") and config.system_instruction:
            system_instr = str(config.system_instruction).lower()

        # Handle specific simulated error triggers for tests
        if "simulated_rate_limit" in clean:
            raise errors.APIError(429, {"error": {"message": "Resource has been exhausted (e.g. check quota)."}})
        if "simulated_server_error" in clean or "simulated_model_answer_fail" in clean:
            raise errors.APIError(500, {"error": {"message": "Internal server error occurred."}})
        if "simulated_timeout" in clean:
            raise GeminiTimeoutError("Gemini request timed out. Please try again.")
        if "simulated_invalid_key" in clean or "simulated_auth_error" in clean:
            raise errors.APIError(401, {"error": {"message": "API key not valid. Please pass a valid API key."}})
        if "simulated_malformed_json" in clean:
            return MockGeminiResponse("NOT_VALID_JSON_FORMAT")
        if "simulated_empty_response" in clean:
            return MockGeminiResponse("")

        # Check if this is a Model Answer JSON generation request
        is_model_answer_req = "model answer" in system_instr or "model answer" in clean or "valid json object" in system_instr

        if not is_model_answer_req:
            # Check for greetings, pleasantries, check-ins, or general non-technical conversational messages
            general_reply = classify_general_or_greeting_message(clean_user or last_user_prompt, history_context=clean_user_hist)
            if general_reply:
                return MockGeminiResponse(general_reply)

        if is_model_answer_req:
            if "polymorphism" in clean:
                model_json = {
                    "correct_answer": (
                        "Polymorphism in Java is a core Object-Oriented Programming (OOP) principle that allows a single interface or method call "
                        "to behave differently based on the concrete runtime object executing it. It is realized through Compile-time Polymorphism "
                        "(method overloading, static binding) and Runtime Polymorphism (method overriding, dynamic binding via JVM dynamic dispatch)."
                    ),
                    "simple_explanation": (
                        "Think of a universal remote control with a 'Power' button: pressing the same button turns on a TV, wakes up a soundbar, "
                        "or toggles smart lights—the identical action triggers different, device-specific behaviors."
                    ),
                    "detailed_explanation": (
                        "1. Compile-time Polymorphism (Method Overloading): Resolved at compile time by the Java compiler based on method signatures (static binding).\n"
                        "2. Runtime Polymorphism (Method Overriding): A subclass provides a specific implementation of a method declared in its parent class. "
                        "At runtime, the JVM uses the object's virtual method table (vtable) to determine which method implementation to execute dynamically.\n"
                        "3. Architectural Significance: It adheres to the Open/Closed Principle (OCP)—software entities should be open for extension but closed for modification."
                    ),
                    "example": (
                        "```java\n"
                        "// Base superclass\n"
                        "class Animal {\n"
                        "    void makeSound() {\n"
                        "        System.out.println(\"Some generic animal sound\");\n"
                        "    }\n"
                        "}\n\n"
                        "// Subclass overriding superclass method\n"
                        "class Dog extends Animal {\n"
                        "    @Override\n"
                        "    void makeSound() {\n"
                        "        System.out.println(\"Woof! Woof!\");\n"
                        "    }\n"
                        "}\n\n"
                        "public class Main {\n"
                        "    public static void main(String[] args) {\n"
                        "        Animal myAnimal = new Dog(); // Line A: Superclass reference holds subclass instance\n"
                        "        myAnimal.makeSound();        // Line B: Dispatched dynamically at runtime to Dog's makeSound\n"
                        "    }\n"
                        "}\n"
                        "```\n\n"
                        "**Important Line Explanations:**\n"
                        "- **Line A (`Animal myAnimal = new Dog()`):** Upcasting allows a superclass reference variable to refer to a subclass object in heap memory.\n"
                        "- **Line B (`myAnimal.makeSound()`):** Dynamic method dispatch resolves the invocation at runtime via the JVM vtable, executing `Dog`'s overridden method rather than `Animal`'s default."
                    ),
                    "key_points_to_remember": [
                        "Polymorphism enables one interface to have multiple implementations.",
                        "Method overloading is compile-time (static binding); method overriding is runtime (dynamic binding).",
                        "In Java, dynamic method dispatch uses JVM virtual method tables (vtables).",
                        "Static, private, and final methods cannot be overridden dynamically."
                    ],
                    "disclaimer": "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
                }
                return MockGeminiResponse(json.dumps(model_json))
            elif "deadlock" in clean:
                model_json = {
                    "correct_answer": (
                        "A deadlock in Operating Systems is a condition where a set of concurrent processes are permanently blocked "
                        "because each process holds a resource and waits to acquire another resource held by another process in the set."
                    ),
                    "simple_explanation": (
                        "Imagine two cars meeting head-on on a narrow single-lane bridge: neither car can proceed forward unless the other reverses, "
                        "but neither driver is willing to back up, causing a complete standstill."
                    ),
                    "detailed_explanation": (
                        "A deadlock occurs if and only if all four Coffman Conditions hold simultaneously:\n"
                        "1. Mutual Exclusion: At least one non-shareable resource.\n"
                        "2. Hold and Wait: A process holds resources while waiting for others.\n"
                        "3. No Preemption: Resources cannot be forcibly seized from a holding process.\n"
                        "4. Circular Wait: A closed chain of processes exists where P[i] waits for P[(i+1)%n]."
                    ),
                    "example": (
                        "```c\n"
                        "// Thread 1 acquires Mutex A then Mutex B\n"
                        "pthread_mutex_lock(&mutexA);\n"
                        "pthread_mutex_lock(&mutexB);\n\n"
                        "// Thread 2 acquires Mutex B then Mutex A (Circular Dependency!)\n"
                        "pthread_mutex_lock(&mutexB);\n"
                        "pthread_mutex_lock(&mutexA);\n"
                        "```\n\n"
                        "**Important Line Explanations:**\n"
                        "- If Thread 1 acquires `mutexA` while Thread 2 acquires `mutexB`, neither thread can proceed to its second lock, producing circular wait deadlock.\n"
                        "- **Fix:** Enforce a global resource acquisition ordering (e.g. always acquire `mutexA` before `mutexB`)."
                    ),
                    "key_points_to_remember": [
                        "All four Coffman conditions must hold simultaneously for a deadlock to occur.",
                        "Deadlock prevention breaks at least one Coffman condition (commonly circular wait via resource ordering).",
                        "Banker's Algorithm is used for deadlock avoidance in safe state resource allocation."
                    ],
                    "disclaimer": "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
                }
                return MockGeminiResponse(json.dumps(model_json))
            elif "binary search tree" in clean or "bst" in clean:
                model_json = {
                    "correct_answer": (
                        "A Binary Search Tree (BST) is a hierarchical node-based data structure where each node has at most two children. "
                        "It strictly enforces the BST invariant: every key in the left subtree is strictly less than the node's key, and every "
                        "key in the right subtree is strictly greater than the node's key. This invariant enables logarithmic search, insertion, and deletion."
                    ),
                    "simple_explanation": (
                        "Think of a BST like looking up a word in a printed dictionary: opening directly to the middle allows you to instantly "
                        "discard half the pages based on whether your word comes alphabetically before or after, halving remaining work at each step."
                    ),
                    "detailed_explanation": (
                        "1. Operational Complexity: On a balanced tree of height h, search, insert, and delete execute in O(h) = O(log n) time. "
                        "In an unbalanced, skewed tree, height degrades to O(n), behaving like a linear singly-linked list.\n"
                        "2. Self-Balancing Invariants: To guarantee O(log n) worst-case time, self-balancing BSTs like AVL Trees (strict height balance factor in {-1,0,1}) "
                        "and Red-Black Trees (color invariants maintaining bounded black-height) perform tree rotations upon node mutations.\n"
                        "3. In-Order Traversal: Performing an in-order traversal (Left -> Root -> Right) on a BST processes all keys in strictly sorted ascending order."
                    ),
                    "example": (
                        "```python\n"
                        "class BSTNode:\n"
                        "    def __init__(self, val):\n"
                        "        self.val = val\n"
                        "        self.left = None\n"
                        "        self.right = None\n\n"
                        "def search_bst(root, target):\n"
                        "    if root is None or root.val == target:\n"
                        "        return root  # Line 1: Found target or reached null leaf\n"
                        "    if target < root.val:\n"
                        "        return search_bst(root.left, target)   # Line 2: Recurse into left subtree\n"
                        "    return search_bst(root.right, target)      # Line 3: Recurse into right subtree\n"
                        "```\n\n"
                        "**Important Line Explanations:**\n"
                        "- **Line 1 (`root is None or root.val == target`):** Base condition terminating recursion upon hit or search exhaustion.\n"
                        "- **Lines 2 & 3 (`search_bst(root.left/right)`):** Binary elimination halves search space at each recursive step based on the BST invariant."
                    ),
                    "key_points_to_remember": [
                        "Enforces the BST invariant: Left Subtree < Root < Right Subtree for all nodes.",
                        "Average time complexity is O(log n); worst-case is O(n) for degenerate trees.",
                        "In-order traversal always visits nodes in monotonically sorted ascending order.",
                        "Self-balancing BST variants (AVL, Red-Black) guarantee O(log n) bounds via tree rotations."
                    ],
                    "disclaimer": "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
                }
                return MockGeminiResponse(json.dumps(model_json))
            elif "tcp" in clean or "handshake" in clean:
                model_json = {
                    "correct_answer": (
                        "The TCP 3-Way Handshake is a connection-establishment protocol used in Transmission Control Protocol (TCP) "
                        "to synchronize initial sequence numbers (ISNs) and establish a reliable, full-duplex byte stream between client and server."
                    ),
                    "simple_explanation": (
                        "Think of it like a formal phone greeting: Person A asks 'Can you hear me?' (SYN), Person B responds 'Yes, I hear you, can you hear me?' (SYN-ACK), "
                        "and Person A confirms 'Yes, let's talk' (ACK)."
                    ),
                    "detailed_explanation": (
                        "1. Step 1 (SYN): Client sends a TCP segment with SYN flag set and an initial sequence number ISN_C = x (client enters SYN_SENT).\n"
                        "2. Step 2 (SYN-ACK): Server allocates TCP buffers, chooses its own ISN_S = y, and responds with SYN=1 and ACK=x+1 (server enters SYN_RCVD).\n"
                        "3. Step 3 (ACK): Client acknowledges with ACK=y+1 (client enters ESTABLISHED). When server receives this, it enters ESTABLISHED."
                    ),
                    "example": (
                        "```text\n"
                        "Client [SYN_SENT]   ---- SYN (seq=100) --------->   Server [LISTEN -> SYN_RCVD]\n"
                        "Client [ESTABLISHED]<--- SYN-ACK (seq=300, ack=101) - Server [SYN_RCVD]\n"
                        "Client [ESTABLISHED]---- ACK (ack=301) ---------->   Server [ESTABLISHED]\n"
                        "```\n\n"
                        "**Important Line Explanations:**\n"
                        "- Both peers independently select and synchronize 32-bit Initial Sequence Numbers.\n"
                        "- A 3-way exchange is mathematically necessary to prevent delayed duplicate SYN packets from causing half-open connections."
                    ),
                    "key_points_to_remember": [
                        "3 packets are required for mutual sequence synchronization and buffer allocation.",
                        "SYN consumes 1 sequence number; ACK acknowledges sequence + 1.",
                        "Prevents duplicate connection requests from corrupting server state.",
                        "Connection teardown requires a 4-way handshake (FIN-ACK-FIN-ACK)."
                    ],
                    "disclaimer": "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
                }
                return MockGeminiResponse(json.dumps(model_json))
            else:
                topic_title = "Computer Science Core Concept"
                for kw, title in [
                    ("sql", "SQL & Database Queries"),
                    ("acid", "ACID Properties in Databases"),
                    ("index", "Database Indexing & Optimization"),
                    ("normalization", "Database Normalization"),
                    ("quicksort", "QuickSort Algorithm"),
                    ("mergesort", "MergeSort Algorithm"),
                    ("virtual memory", "Virtual Memory & Paging"),
                    ("mutex", "Mutex & Synchronization"),
                    ("docker", "Docker & Containerization"),
                    ("thread", "Multithreading & Concurrency"),
                    ("cache", "Caching & Memory Hierarchy"),
                ]:
                    if kw in clean:
                        topic_title = title
                        break

                ref_sol = reference_answers.get_authoritative_reference_solution(
                    question_text=clean,
                    topic=topic_title,
                    subject="Computer Science"
                )

                model_json = {
                    "correct_answer": ref_sol["correct_answer"],
                    "simple_explanation": ref_sol["simple_explanation"],
                    "detailed_explanation": ref_sol["detailed_explanation"],
                    "example": ref_sol["example"],
                    "key_points_to_remember": ref_sol["key_points_to_remember"],
                    "improved_answer": ref_sol.get("improved_answer"),
                    "disclaimer": "AI-generated model answer for revision purposes; verify with curriculum and examiner standards."
                }
                return MockGeminiResponse(json.dumps(model_json))

        # Ambiguity Handling (Requirement 8)
        if (
            "simulated_ambiguous" in clean
            or clean.strip().endswith("what is a pipeline?")
            or clean.strip().endswith("what is pipeline?")
            or clean.strip().endswith("is it fast?")
        ):
            return MockGeminiResponse(
                "### Request for Clarification: Ambiguous Technical Question\n\n"
                "Your question could refer to several distinct concepts in Computer Science:\n"
                "1. **CPU Instruction Pipelining (Computer Architecture)**: Overlapping execution stages (Fetch, Decode, Execute, Memory, Write-back) to increase instruction throughput.\n"
                "2. **Unix / Shell Pipelines (Operating Systems)**: Chaining process outputs to inputs via anonymous pipes (`stdout | stdin`).\n"
                "3. **CI/CD Build Pipelines (DevOps / Software Engineering)**: Automated sequences of build, test, and deployment jobs.\n\n"
                "**Please clarify:** Which of these three domains or mechanisms would you like to explore for your viva preparation?"
            )

        # Uncertain Information / Limitation Handling (Requirement 9)
        if "simulated_uncertain" in clean or "uncertain limitation" in clean:
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "The requested detail is currently uncertain and not defined within standard computer science specifications.\n\n"
                "### 2. Simple Explanation\n"
                "In computer systems, when an implementation detail is left unspecified by standard RFCs or vendor manuals, system architectures may behave differently.\n\n"
                "### 3. Detailed Explanation\n"
                "**Limitation Disclosure:** Exact internal timing thresholds and undocumented opcode behaviors are proprietary to hardware steppings. Academic examiners expect candidates to identify that this behavior is implementation-defined rather than guessing or fabricating numbers.\n\n"
                "### 4. Practical Example & Code\n"
                "In the C language specification (ISO/IEC 9899), reading an uninitialized variable triggers 'Undefined Behavior' (UB); the compiler is not obligated to produce a deterministic result.\n\n"
                "### 5. Important Points to Remember\n"
                "- Clearly identify when a mechanism is undefined or implementation-dependent in viva examinations.\n"
                "- Distinguish between formal standards and proprietary vendor extensions."
            )

        # Identify whether user is asking a follow-up or elaboration on previous topic
        is_followup = (user_turn_count > 1) and any(
            kw in clean_user for kw in (
                "explaination", "explanation", "follow-up", "clarification", "more details",
                "deep dive", "tell more", "example", "elaborate", "for this", "what about",
                "how about", "further", "again", "also", "stack", "avoid", "simpler", "simple",
                "another example", "previous topic"
            )
        )
        is_simplification = is_followup and any(w in clean_user for w in ("simpler words", "simpler", "in simple terms", "explain simpler", "more simple", "simple language", "plain words", "simpler terms"))
        is_another_example = is_followup and any(w in clean_user for w in ("another example", "different example", "second example", "more examples", "give another"))

        # Standard Doubt-Clearing Chat Responses structured with the 5 Educational Pillars
        if "polymorphism" in clean_user or (is_followup and "polymorphism" in clean_user_hist and not any(k in clean_user for k in ("java", "python"))):
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "**Polymorphism** in Java is a core Object-Oriented Programming (OOP) principle that allows a single interface or method call to behave differently based on the concrete runtime object executing it.\n\n"
                "### 2. Simple Explanation\n"
                "Think of a universal remote control with a 'Power' button: pressing that single button turns on a television, wakes up an audio amplifier, or toggles smart lighting—the identical command triggers different, device-specific behaviors based on what device receives it.\n\n"
                "### 3. Detailed Explanation\n"
                "Java supports two primary forms of polymorphism:\n"
                "1. **Compile-time Polymorphism (Static Binding / Method Overloading)**: Same method name with different parameters resolved during compilation.\n"
                "2. **Runtime Polymorphism (Dynamic Binding / Method Overriding)**: Subclass provides a specific implementation of a method declared in its superclass, resolved at runtime via the virtual method table (`vtable`).\n\n"
                "### 4. Practical Example & Code\n"
                "```java\n"
                "class Animal {\n"
                "    void makeSound() {\n"
                "        System.out.println(\"Animal sound\");\n"
                "    }\n"
                "}\n"
                "class Dog extends Animal {\n"
                "    @Override\n"
                "    void makeSound() {\n"
                "        System.out.println(\"Woof! Woof!\");\n"
                "    }\n"
                "}\n"
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                "        Animal myAnimal = new Dog(); // Line 1: Superclass reference holds subclass instance\n"
                "        myAnimal.makeSound();        // Line 2: Dispatched dynamically at runtime -> prints 'Woof! Woof!'\n"
                "    }\n"
                "}\n"
                "```\n\n"
                "**Line Breakdown & Logic:**\n"
                "- **Line 1 (`Animal myAnimal = new Dog()`):** Upcasting enables polymorphic referencing where an `Animal` reference refers to a heap-allocated `Dog` instance.\n"
                "- **Line 2 (`myAnimal.makeSound()`):** Dynamic method dispatch resolves the invocation at runtime via the JVM `vtable`, executing `Dog`'s overridden method rather than `Animal`'s default.\n\n"
                "### 5. Important Points to Remember\n"
                "- Dynamic polymorphism enables the **Open/Closed Principle**—your code is open for extension but closed for modification.\n"
                "- Method overloading is static binding (compile-time); method overriding is dynamic binding (runtime).\n"
                "- In Java, `private`, `final`, and `static` methods cannot be overridden dynamically."
            )
        elif "deadlock" in clean_user or ("deadlock" in clean and ("condition" in clean_user or "coffman" in clean_user or (is_followup and not any(w in clean_user for w in ("java", "python", "polymorphism"))))):
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "A **deadlock** in Operating Systems is a state where a set of processes are permanently blocked because each process holds a resource and waits for another resource held by another process in the set.\n\n"
                "### 2. Simple Explanation\n"
                "Imagine two cars driving from opposite directions onto a narrow single-lane bridge: neither car can proceed forward unless the other reverses, but neither driver is willing to back up, creating a permanent impasse.\n\n"
                "### 3. Detailed Explanation\n"
                "A deadlock occurs if and only if all four **Coffman Conditions** hold simultaneously:\n"
                "1. **Mutual Exclusion**: At least one resource is held in a non-shareable mode.\n"
                "2. **Hold and Wait**: A process holds at least one resource and waits to acquire additional resources.\n"
                "3. **No Preemption**: Resources cannot be preempted; they are released only voluntarily.\n"
                "4. **Circular Wait**: A closed chain of processes exists where each process holds a resource needed by the next.\n\n"
                "### 4. Practical Example & Code\n"
                "```c\n"
                "// Thread 1 acquires Mutex A then Mutex B\n"
                "pthread_mutex_lock(&mutexA);\n"
                "pthread_mutex_lock(&mutexB); // Blocks if Thread 2 holds Mutex B\n\n"
                "// Thread 2 acquires Mutex B then Mutex A (Circular Dependency!)\n"
                "pthread_mutex_lock(&mutexB);\n"
                "pthread_mutex_lock(&mutexA); // Blocks if Thread 1 holds Mutex A\n"
                "```\n\n"
                "**Line Breakdown & Logic:**\n"
                "- If Thread 1 acquires `mutexA` and Thread 2 acquires `mutexB`, neither thread can proceed to its second lock, producing circular wait deadlock.\n"
                "- **Remediation:** Enforce a strict global resource acquisition ordering (always acquire `mutexA` before `mutexB`).\n\n"
                "### 5. Important Points to Remember\n"
                "- All four Coffman conditions must hold simultaneously for a deadlock to occur.\n"
                "- Deadlock prevention breaks at least one Coffman condition (commonly circular wait via resource ordering).\n"
                "- Banker's Algorithm is used for deadlock avoidance in safe state resource allocation."
            )
        elif "binary search tree" in clean_user or "bst" in clean_user or (is_followup and "bst" in clean_user_hist and not any(w in clean_user for w in ("java", "python", "polymorphism", "deadlock"))):
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "A **Binary Search Tree (BST)** is a hierarchical node-based data structure where each node has at most two children and strictly maintains the BST ordering invariant.\n\n"
                "### 2. Simple Explanation\n"
                "Think of searching for a word in a printed dictionary: opening directly to the middle allows you to instantly eliminate half the book depending on whether the target word comes before or after alphabetically, cutting the remaining problem in half at every step.\n\n"
                "### 3. Detailed Explanation\n"
                "1. **The BST Invariant:** For any node $N$, all keys in its left subtree are strictly smaller than $N.val$, and all keys in its right subtree are strictly greater than $N.val$.\n"
                "2. **Time Complexity:**\n"
                "   - Average Case (Balanced Tree): $O(\\log n)$ for Search, Insert, and Delete.\n"
                "   - Worst Case (Degenerate Skewed Tree): $O(n)$ if items are inserted in sorted order.\n"
                "3. **In-Order Traversal:** Performing an in-order traversal (Left $\\to$ Root $\\to$ Right) on a BST processes all keys in strictly sorted ascending order.\n\n"
                "### 4. Practical Example & Code\n"
                "```python\n"
                "class Node:\n"
                "    def __init__(self, val):\n"
                "        self.val = val\n"
                "        self.left = None\n"
                "        self.right = None\n\n"
                "def search_bst(root, target):\n"
                "    # Line 1: Target found or reached null leaf\n"
                "    if root is None or root.val == target:\n"
                "        return root\n"
                "    # Line 2: If target is smaller, recurse left\n"
                "    if target < root.val:\n"
                "        return search_bst(root.left, target)\n"
                "    # Line 3: If target is larger, recurse right\n"
                "    return search_bst(root.right, target)\n"
                "```\n\n"
                "**Line Breakdown & Logic:**\n"
                "- **Line 1:** Base condition terminating recursion upon hit or search exhaustion.\n"
                "- **Lines 2 & 3:** Binary decision halves the remaining search space at each recursive step based on the BST invariant.\n\n"
                "### 5. Important Points to Remember\n"
                "- Invariant: Left Subtree $<$ Root $<$ Right Subtree for all nodes.\n"
                "- Average time complexity is $O(\\log n)$; worst-case is $O(n)$ for degenerate trees.\n"
                "- In-order traversal always visits nodes in monotonically sorted ascending order.\n"
                "- Self-balancing BST variants (AVL, Red-Black) guarantee $O(\\log n)$ bounds via tree rotations."
            )
        elif "tcp" in clean_user or "handshake" in clean_user or (is_followup and "tcp" in clean_user_hist and not any(w in clean_user for w in ("java", "python", "polymorphism", "deadlock", "bst"))):
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "The **TCP 3-Way Handshake** is the connection-establishment protocol used in Transmission Control Protocol (TCP) to synchronize initial sequence numbers (ISNs), allocate socket buffers, and establish a reliable full-duplex communication channel between client and server.\n\n"
                "### 2. Simple Explanation\n"
                "Think of a two-way radio check: Station A says 'Radio check, can you copy?' (SYN). Station B responds 'Loud and clear, copy you, how do you copy me?' (SYN-ACK). Station A confirms 'Copy you loud and clear, ready for message' (ACK). Both are now certain they can talk and listen.\n\n"
                "### 3. Detailed Explanation\n"
                "The three-step sequence operates as follows:\n"
                "1. **SYN**: Client sends `SYN=1`, `seq=x` (Client enters `SYN_SENT`).\n"
                "2. **SYN-ACK**: Server sends `SYN=1`, `ACK=1`, `seq=y`, `ack=x+1` (Server enters `SYN_RCVD`).\n"
                "3. **ACK**: Client sends `ACK=1`, `seq=x+1`, `ack=y+1` (Client and Server enter `ESTABLISHED`).\n\n"
                "### 4. Practical Example & Code\n"
                "```text\n"
                "Client [SYN_SENT]    ---- SYN (seq=100) -------------> Server [LISTEN -> SYN_RCVD]\n"
                "Client [ESTABLISHED] <--- SYN-ACK (seq=300, ack=101) - Server [SYN_RCVD]\n"
                "Client [ESTABLISHED] ---- ACK (ack=301) -------------> Server [ESTABLISHED]\n"
                "```\n\n"
                "**Line Breakdown & Logic:**\n"
                "- **Packet 1 (SYN):** Announces client sequence number $x$. Consumes 1 sequence number.\n"
                "- **Packet 2 (SYN-ACK):** Server acknowledges $x+1$ and announces server sequence number $y$.\n"
                "- **Packet 3 (ACK):** Client acknowledges $y+1$. Both sequence spaces are synchronized.\n\n"
                "### 5. Important Points to Remember\n"
                "- 3 packets are mathematically necessary to prevent delayed duplicate SYN packets from causing half-open connections.\n"
                "- Sequence numbers start at a randomized Initial Sequence Number (ISN) to defend against TCP sequence prediction attacks.\n"
                "- Connection setup uses a 3-way handshake; graceful teardown uses a **4-way handshake** (FIN, ACK, FIN, ACK)."
            )
        # Context-Aware Java Handling
        elif ("java" in clean_user and "inheritance" not in clean_user) or (is_followup and "java" in clean_user_hist and not any(k in clean_user for k in ("polymorphism", "inheritance")) and "inheritance" not in clean_user_hist and "polymorphism" not in clean_user_hist):
            is_detail_requested = any(w in clean_user for w in ("detail", "in-depth", "architecture", "subsystem", "jvm architecture", "memory model", "explain java in detail", "internal", "deep dive"))
            if is_detail_requested or is_followup:
                return MockGeminiResponse(
                    "In Java, program execution and runtime lifecycle are managed by the Java Virtual Machine (JVM) through three primary subsystems: ClassLoader Subsystem, JVM Memory Areas, and Execution Engine.\n\n"
                    "1. ClassLoader Subsystem: Loads, links (verifies, prepares, resolves), and initializes `.class` bytecode files.\n"
                    "2. JVM Memory Model:\n"
                    "   - Heap: Shared memory where all objects and instance arrays reside, managed by Garbage Collection.\n"
                    "   - Stack: Thread-private memory storing stack frames for each method execution (local variables and partial results).\n"
                    "   - Metaspace: Stores class metadata, runtime constant pools, and method descriptors.\n"
                    "3. Execution Engine:\n"
                    "   - Interpreter: Reads and executes bytecode sequentially.\n"
                    "   - JIT Compiler: Compiles frequently called 'hot' code into native machine code for high performance.\n"
                    "   - Garbage Collector: Reclaims memory from unreachable heap objects.\n\n"
                    "```java\n"
                    "public class JvmMemoryDemo {\n"
                    "    public static void main(String[] args) {\n"
                    "        int count = 10; // Stored on thread stack\n"
                    "        String message = new String(\"Deep Java\"); // Reference on stack, object in heap\n"
                    "        processTask(count, message);\n"
                    "    }\n"
                    "    private static void processTask(int num, String text) {\n"
                    "        System.out.println(text + \": \" + num);\n"
                    "    }\n"
                    "}\n"
                    "```"
                )
            else:
                return MockGeminiResponse(
                    "Java is a programming language used to build applications. It runs on the Java Virtual Machine (JVM), which helps it work across different platforms."
                )

        # Step-by-Step Mathematics Problem Solving
        elif any(k in clean_user for k in ("solve this mathematics question", "solve this math question", "mathematics question step by step", "math question step by step", "solve step by step", "solve math", "solve equation")) or ("solve" in clean_user and any(w in clean_user for w in ("math", "mathematics", "equation", "algebra", "calculus", "step by step", "2x"))):
            return MockGeminiResponse(
                "Here is a clear, step-by-step mathematical solution with complete algebraic reasoning and verification:\n\n"
                "### 1. Problem Formulation & Objective\n"
                "Let us solve the linear algebraic equation:\n"
                "$$2x + 5 = 15$$\n"
                "**Goal:** Isolate the unknown variable $x$ on one side of the equation to determine its exact value.\n\n"
                "### 2. Step-by-Step Algebraic Solution\n\n"
                "**Step 1: Isolate the variable term by subtracting the constant from both sides**\n"
                "We subtract $5$ from both the left-hand side (LHS) and right-hand side (RHS) to maintain equality:\n"
                "$$2x + 5 - 5 = 15 - 5$$\n"
                "$$2x = 10$$\n\n"
                "**Step 2: Solve for $x$ by dividing both sides by the coefficient**\n"
                "The coefficient of $x$ is $2$. Divide both sides by $2$:\n"
                "$$\\frac{2x}{2} = \\frac{10}{2}$$\n"
                "$$x = 5$$\n\n"
                "### 3. Step 3: Formal Verification (Proof by Substitution)\n"
                "We substitute our calculated value $x = 5$ back into the original equation to verify correctness:\n"
                "$$\\text{LHS} = 2(5) + 5 = 10 + 5 = 15$$\n"
                "$$\\text{RHS} = 15$$\n"
                "Since $\\text{LHS} = \\text{RHS} = 15$, the solution $x = 5$ is verified to be 100% correct.\n\n"
                "### 4. Key Mathematical Rules Applied\n"
                "1. **Subtraction Property of Equality:** Subtracting the same number from both sides preserves equation balance.\n"
                "2. **Division Property of Equality:** Dividing both sides by the same non-zero number yields an equivalent equation.\n"
                "3. **Verification Check:** Always substitute the final solution back into the original equation to catch calculation slips.\n\n"
                "Would you like to solve a specific equation from your homework or practice quadratic equations, systems of linear equations, or calculus derivatives next?"
            )

        # Python Learning Roadmap
        elif any(k in clean_user for k in ("learn python", "learning python", "how do i learn python", "how to learn python", "python roadmap", "start python", "python for beginners")) or ("python" in clean_user and any(w in clean_user for w in ("how", "learn", "roadmap", "guide", "start", "study", "beginner"))):
            return MockGeminiResponse(
                "Learning Python is one of the most rewarding and versatile journeys in modern computing! "
                "Here is a structured, step-by-step roadmap to go from absolute beginner to building real-world applications:\n\n"
                "### 1. Phase 1: Python Fundamentals (Weeks 1–2)\n"
                "- **Core Concepts:** Variables, dynamic typing, primitive data types (`int`, `float`, `str`, `bool`), and type casting.\n"
                "- **Control Flow:** `if / elif / else` conditional statements, `for` loops, `while` loops, and `break` / `continue` statements.\n"
                "- **Functions:** Defining functions (`def`), parameters, default arguments, `*args` / `**kwargs`, and return values.\n"
                "- **Hands-On Practice:** Write simple command-line programs (number guessing game, basic calculator, unit converter).\n\n"
                "### 2. Phase 2: Built-in Data Structures & Collections (Weeks 3–4)\n"
                "- **Lists:** Indexing, slicing, appending, extending, and **list comprehensions** (`[x*2 for x in nums]`).\n"
                "- **Dictionaries & Sets:** Key-value mappings, fast $O(1)$ lookups, and unique sets for membership testing.\n"
                "- **Tuples & Strings:** Immutability, string formatting (f-strings: `f'Hello, {name}'`), and common string methods.\n"
                "- **File Handling:** Safely reading and writing files using context managers (`with open(...) as f:`).\n\n"
                "### 3. Phase 3: Object-Oriented Programming (OOP) & Clean Code (Weeks 5–6)\n"
                "- **Classes & Objects:** Defining classes, attributes, `__init__` constructor, and `self`.\n"
                "- **OOP Pillars:** Encapsulation (private attributes `_var`), Inheritance (`class Child(Parent)`), and Polymorphism.\n"
                "- **Modules & Packages:** Using standard library modules (`math`, `random`, `datetime`, `os`, `sys`).\n"
                "- **Exception Handling:** Robust error management using `try / except / finally` blocks.\n\n"
                "### 4. Phase 4: Choose a Specialization & Build Projects (Weeks 7–10)\n"
                "- **Web Development:** Build REST APIs with **FastAPI** or full-stack apps with **Flask / Django**.\n"
                "- **Data Science & AI:** Explore data manipulation with **Pandas & NumPy**, and ML models with **Scikit-Learn**.\n"
                "- **Automation & Scripting:** Automate web tasks using **BeautifulSoup & Requests** or schedule recurring scripts.\n\n"
                "### 5. Daily Learning Tips for Success\n"
                "1. **Code Everyday:** Write code for 30–45 minutes daily instead of passively watching hours of tutorials.\n"
                "2. **Debug Actively:** Read Python error tracebacks carefully; they tell you exactly what went wrong and on what line.\n"
                "3. **Build Mini Projects:** Apply each concept immediately by creating something small and interactive.\n\n"
                "Would you like to start with an interactive beginner exercise, or would you like guidance on setting up Python and VS Code?"
            )

        # Context-Aware Python Handling
        elif ("python" in clean_user and not any(k in clean_user for k in ("recursion", "inheritance", "stack", "algorithm", "tree", "photosynthesis", "timetable", "placement"))) or (is_followup and "python" in clean_user_hist and not any(k in clean_user_hist for k in ("recursion", "inheritance", "stack", "photosynthesis", "timetable", "placement"))):
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "**Python** is a high-level, interpreted, dynamically-typed, multi-paradigm programming language emphasizing developer productivity, elegant syntax, and readability.\n\n"
                "### 2. Simple Explanation\n"
                "Think of Python like an interactive human conversation: instead of writing and translating a whole book in advance (compiling ahead of time), an interpreter reads and runs each instruction sequentially.\n\n"
                "### 3. Detailed Explanation\n"
                "1. **Execution Engine:** Source code (`.py`) is compiled into bytecode (`.pyc`) executed by the Python Virtual Machine (CPython).\n"
                "2. **Dynamic Typing:** Variables are bound to objects at runtime; variables themselves do not hold fixed types.\n"
                "3. **Memory Management:** Memory reclamation uses reference counting as the primary strategy, supplemented by a generational cyclic garbage collector.\n"
                "4. **Global Interpreter Lock (GIL):** A mutex in CPython that prevents multiple native threads from executing Python bytecode simultaneously, preserving memory safety.\n\n"
                "### 4. Practical Example & Code\n"
                "```python\n"
                "# Pythonic implementation illustrating list comprehension and dynamic types\n"
                "def filter_scores(raw_data):\n"
                "    # Clean list comprehension in linear O(n) time\n"
                "    return [score for score in raw_data if isinstance(score, (int, float)) and score >= 50]\n\n"
                "print(filter_scores([85, 'absent', 92, None, 40, 78]))\n"
                "```\n\n"
                "**Line Breakdown & Logic:**\n"
                "- `isinstance(score, (int, float))`: Defensive type validation at runtime.\n"
                "- `[score for ... if ...]`: Optimized C-level loop generating a filtered list without side-effects.\n\n"
                "### 5. Important Points to Remember\n"
                "- Python is strongly and dynamically typed.\n"
                "- Memory management combines reference counting with generational garbage collection.\n"
                "- The GIL ensures thread-safe memory management in CPython."
            )

        # Requirement 2: Java Inheritance
        elif (("inheritance" in clean_user or (is_followup and "inheritance" in clean_user_hist)) and not is_another_example and not is_simplification):
            return MockGeminiResponse(
                "Inheritance in Java is a mechanism where one class acquires the properties and methods of another class using the `extends` keyword. The class that inherits is called the subclass (or child class), and the class being inherited from is called the superclass (or parent class).\n\n"
                "Key Concepts:\n"
                "- `extends`: Used by a subclass to inherit from a superclass.\n"
                "- `super`: Used inside a subclass to access parent constructors (`super()`) or parent methods (`super.method()`).\n"
                "- Method Overriding: A subclass provides a specific implementation of a method defined in the superclass.\n"
                "- Access Control: `public` and `protected` members are inherited; `private` members are not directly accessible.\n\n"
                "Example:\n"
                "```java\n"
                "// Parent class\n"
                "class Vehicle {\n"
                "    protected String brand = \"Generic Vehicle\";\n\n"
                "    public void start() {\n"
                "        System.out.println(\"Vehicle is starting smoothly.\");\n"
                "    }\n"
                "}\n\n"
                "// Child class inheriting from Vehicle\n"
                "class Car extends Vehicle {\n"
                "    private int doors = 4;\n\n"
                "    @Override\n"
                "    public void start() {\n"
                "        super.start(); // Call parent version\n"
                "        System.out.println(brand + \" car with \" + doors + \" doors is ready to drive!\");\n"
                "    }\n"
                "}\n\n"
                "public class Main {\n"
                "    public static void main(String[] args) {\n"
                "        Car myCar = new Car(); // Subclass instantiation\n"
                "        myCar.start();         // Executes overridden method\n"
                "    }\n"
                "}\n"
                "```\n\n"
                "Why Use Inheritance?\n"
                "1. Code Reusability: Common logic stays in the parent class so you don't repeat code.\n"
                "2. Polymorphism: Allows subclasses to be treated as instances of their superclass at runtime.\n"
                "Java supports single inheritance for classes to avoid ambiguity (the diamond problem), but supports multiple inheritance through interfaces."
            )

        # Requirement 3: Improving English
        elif any(k in clean_user for k in ("improve my english", "improve english", "learn english", "speak english", "english communication")) or ("english" in clean_user and any(w in clean_user for w in ("improve", "speak", "practice", "fluent", "better", "talk", "learn"))):
            return MockGeminiResponse(
                "Improving your English is a journey of small, consistent daily habits! Here is a friendly, proven step-by-step roadmap to become confident and fluent in speaking, listening, reading, and writing:\n\n"
                "### 1. Speak Aloud Every Day (Active Production)\n"
                "- **Talk to yourself:** Describe what you are doing in English as you go about your day (e.g., *\"Now I am opening my laptop to review data structures\"*).\n"
                "- **Shadowing technique:** Listen to a 30-second clip from a podcast, YouTube video, or TED talk, pause it, and repeat the sentences aloud mimicking the speaker's pronunciation, pacing, and intonation.\n"
                "- **Record voice notes:** Record yourself speaking about your day for 1 minute on your phone. Listening back helps you identify where you hesitate and celebrate improvements!\n\n"
                "### 2. Active Listening (Ear Training)\n"
                "- Listen to content you genuinely enjoy: English podcasts (*BBC 6 Minute English*, *NPR Life Kit*), audiobooks, or educational channels.\n"
                "- Watch movies or interviews with English subtitles turned on. Notice how native speakers naturally link words together.\n\n"
                "### 3. Read for Context & Vocabulary (15 Minutes Daily)\n"
                "- Read articles, newspapers, or short stories at your current comfort level.\n"
                "- **The Rule of 3:** Don't stop to look up every single word. Highlight just 3 interesting words per day, find their meanings, and immediately write 2 original sentences using each word.\n\n"
                "### 4. Write Daily Reflections (Clear Expression)\n"
                "- Keep a short daily journal: write 3 to 5 sentences every night about what you learned, accomplished, or experienced.\n"
                "- Practice writing professional emails, project descriptions, or technical summaries in clean English.\n\n"
                "### 5. Master Core Grammar Naturally\n"
                "- Focus on high-impact fundamentals: **Subject-Verb Agreement**, **Verb Tenses** (Past, Present, Future), and **Prepositions** (in, on, at).\n"
                "- Learn grammar patterns through whole sentences rather than memorizing dry rule tables.\n\n"
                "### Friendly Mindset Tip 💡\n"
                "Never let fear of making mistakes stop you from speaking. Communication is about connecting and sharing ideas clearly, not perfection! With just 15–20 minutes of daily practice, you will notice remarkable improvement in 30 days.\n\n"
                "Would you like to practice a short conversation or work on any specific area (like interview English, email writing, or pronunciation)?"
            )

        # Requirement 4: Study Timetable for Exams
        elif any(k in clean_user for k in ("study timetable", "study plan", "timetable for my exams", "timetable for exam", "exam schedule", "study routine")) or ("timetable" in clean_user) or ("study" in clean_user and "exam" in clean_user):
            return MockGeminiResponse(
                "Here is a balanced, highly effective **Exam Preparation Study Timetable** designed to maximize focus, retention, and prevent burnout. It combines the **Pomodoro technique** (50 mins study + 10 mins break) with **Active Recall**:\n\n"
                "### 📅 Daily Study Schedule Framework\n\n"
                "| Time Block | Focus Session | Recommended Activity |\n"
                "| :--- | :--- | :--- |\n"
                "| **8:00 AM – 9:30 AM** | 🧠 **Deep Work: Hardest Topic** | Study your most challenging subject when mental energy is at its peak (e.g. Math, Algorithms, Physics). |\n"
                "| **9:30 AM – 9:45 AM** | ☕ Break | Hydrate, stretch, step away from screens. |\n"
                "| **9:45 AM – 11:30 AM** | ✍️ **Problem Solving & Practice** | Work through numerical problems, write code, or solve previous year questions (PYQs). |\n"
                "| **11:30 AM – 1:00 PM** | 📚 **Subject 2: Core Theory** | Study a second, complementary subject (e.g. DBMS, Operating Systems, Chemistry). |\n"
                "| **1:00 PM – 2:30 PM** | 🍲 Lunch & Rest | Relax, eat a wholesome meal, take a 20-minute power nap if needed. |\n"
                "| **2:30 PM – 4:30 PM** | 📝 **Active Recall & Testing** | Test yourself without looking at notes. Write formulas or key points from memory. |\n"
                "| **4:30 PM – 5:30 PM** | 🏃 Exercise & Recharge | Walk, workout, or talk with friends. Physical activity dramatically boosts memory retention! |\n"
                "| **5:30 PM – 7:30 PM** | 💻 **Practical Work & Assignments** | Lab exercises, coding practice, or assignment completion. |\n"
                "| **7:30 PM – 8:30 PM** | 🍽️ Dinner | Relax with family or friends. |\n"
                "| **8:30 PM – 9:45 PM** | 🔄 **Spaced Repetition & Recap** | Quickly review everything studied today and preview tomorrow's topics. |\n"
                "| **10:30 PM** | 😴 Sleep | Prioritize 7–8 hours of quality sleep for memory consolidation. |\n\n"
                "### 🚀 3 Golden Study Techniques for Top Scores\n"
                "1. **The 80/20 Rule (Pareto Principle):** Identify the 20% of high-weightage chapters that carry 80% of the exam marks and master them first.\n"
                "2. **The Feynman Technique:** Try explaining difficult topics in simple words as if teaching a beginner. Any point where you get stuck shows exactly what to revise.\n"
                "3. **Timed Mock Tests:** Solve at least two full-length previous year papers under real exam timing conditions.\n\n"
                "Would you like to customize this timetable for your specific subjects, exam dates, or daily college hours? Just let me know your subjects and schedule!"
            )

        # Requirement 5: Placement Preparation
        elif any(k in clean_user for k in ("prepare for placements", "prepare for placement", "placement preparation", "campus placement", "campus recruitment", "crack placement")) or ("placement" in clean_user and any(w in clean_user for w in ("prepare", "prep", "job", "crack", "guide", "roadmap"))):
            return MockGeminiResponse(
                "Campus placement preparation is a structured journey. By dividing your preparation into clear, focused phases, you can build steady confidence and land your dream offer! Here is a comprehensive 5-pillar placement preparation roadmap:\n\n"
                "### 1. Phase 1: Quantitative Aptitude & Logical Reasoning (Weeks 1–3)\n"
                "- **Why it matters:** The online aptitude test is the first screening filter used by most companies.\n"
                "- **Key Topics:** Percentages, Profit & Loss, Ratios, Speed & Distance, Permutations/Probability, Blood Relations, Syllogisms, and Data Interpretation.\n"
                "- **Practice Routine:** Solve 15–20 questions daily on platforms like IndiaBIX or RS Aggarwal.\n\n"
                "### 2. Phase 2: Data Structures & Algorithms (DSA) (Weeks 2–8)\n"
                "- **Why it matters:** Core technical coding rounds test structured problem-solving and algorithmic thinking.\n"
                "- **Must-Master Topics:**\n"
                "  - Arrays, Strings, Two Pointers, Sliding Window\n"
                "  - Hash Maps & Sets\n"
                "  - Linked Lists, Stacks & Queues\n"
                "  - Trees (Binary Tree traversals, BST) & Graphs (BFS/DFS)\n"
                "  - Sorting & Searching, Basic Dynamic Programming\n"
                "- **Goal:** Aim to understand core patterns (100–150 quality problems across LeetCode / GeeksforGeeks) rather than memorizing solutions.\n\n"
                "### 3. Phase 3: Core Computer Science Subjects (Weeks 5–8)\n"
                "- **Database Management Systems (DBMS):** SQL queries (JOINs, GROUP BY, subqueries), Normalization (1NF to BCNF), ACID properties, Indexing.\n"
                "- **Operating Systems (OS):** Process vs Thread, CPU Scheduling, Deadlocks & Coffman conditions, Virtual Memory & Paging, Semaphore vs Mutex.\n"
                "- **Computer Networks (CN):** OSI vs TCP/IP layers, 3-Way Handshake, DNS, HTTP vs HTTPS, IP addressing.\n"
                "- **Object-Oriented Programming (OOP):** Encapsulation, Abstraction, Inheritance, Polymorphism with code examples in your primary language (Java/C++/Python).\n\n"
                "### 4. Phase 4: Resume & High-Quality Projects (Weeks 6–9)\n"
                "- **1-Page Clean Resume:** Use standard templates (Jake's Resume / Overleaf), clean bullet points with metrics (*\"Boosted API response time by 40% using Redis caching\"*).\n"
                "- **2 Solid Projects:** Build full-stack or ML projects hosted live with GitHub repositories. Be ready to explain your architecture, database schema, trade-offs, and challenges faced.\n\n"
                "### 5. Phase 5: Technical Viva & HR Rounds (Weeks 8–10)\n"
                "- **STAR Method for Behavioral Questions:** (Situation, Task, Action, Result) for questions like *\"Tell me about a time you resolved a conflict in a team\"*.\n"
                "- **Classic HR Questions:** *\"Tell me about yourself\"*, *\"Why do you want to join our company?\"*, and *\"What are your strengths and weaknesses?\"*.\n"
                "- Practice mock interviews aloud!\n\n"
                "Would you like to practice answering *\"Tell me about yourself\"* or take a mock technical question right now?"
            )

        # Requirement 6: Photosynthesis in Simple English
        elif "photosynthesis" in clean_user:
            return MockGeminiResponse(
                "### What is Photosynthesis? (In Simple English)\n\n"
                "**Photosynthesis** is the wonderful natural process that green plants use to make their own food using **sunlight**, **water**, and **air**! 🌿☀️\n\n"
                "The word comes from two Greek words:\n"
                "- **Photo** = Light\n"
                "- **Synthesis** = Putting things together\n\n"
                "So, photosynthesis literally means *\"making food using light!\"*\n\n"
                "---\n\n"
                "### The \"Kitchen in the Leaf\" Analogy 🍳\n"
                "Imagine every green leaf is a tiny solar-powered kitchen:\n"
                "1. **The Chef (Chlorophyll):** Inside the leaf cells are tiny green structures called chloroplasts containing **chlorophyll**. Chlorophyll acts like solar panels catching sunlight.\n"
                "2. **Ingredient 1 (Water - $H_2O$):** Plant roots absorb water from the soil and send it up through stem pipes to the leaves.\n"
                "3. **Ingredient 2 (Carbon Dioxide - $CO_2$):** Leaves have tiny breathing pores called **stomata** that take in carbon dioxide gas from the air.\n"
                "4. **The Stove (Sunlight Energy):** Sunlight provides the heat and energy to cook the ingredients together.\n"
                "5. **The Cooked Dish (Glucose - $C_6H_{12}O_6$):** A nutritious sugar that the plant eats to grow fruits, flowers, and leaves.\n"
                "6. **The Fresh Gift to Earth (Oxygen - $O_2$):** As a bonus, the leaf releases clean oxygen gas back into the air for humans and animals to breathe!\n\n"
                "---\n\n"
                "### The Simple Chemical Formula\n"
                "$$\\text{Carbon Dioxide} + \\text{Water} + \\text{Sunlight} \\longrightarrow \\text{Glucose (Food)} + \\text{Oxygen}$$\n\n"
                "$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} + \\text{Light Energy} \\longrightarrow \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$\n\n"
                "---\n\n"
                "### Why is Photosynthesis Crucial for Life?\n"
                "- **It feeds the planet:** Plants produce the base food energy for herbivores, carnivores, and humans.\n"
                "- **It purifies our air:** It absorbs greenhouse gas ($CO_2$) and produces the oxygen ($O_2$) we breathe every single second.\n\n"
                "Does this explanation help make the process clear? Let me know if you'd like to explore how light and dark reactions work!"
            )

        # Requirement 7: Recursion with Example
        elif ("recursion" in clean_user or "recursive" in clean_user) and not is_another_example and not is_simplification:
            return MockGeminiResponse(
                "Don't worry at all! **Recursion** can feel tricky at first, but once you see how it works with a simple picture, it becomes super intuitive! 😊\n\n"
                "### 1. What is Recursion?\n"
                "In programming, **recursion** happens when a function **calls itself** to solve a smaller version of the exact same problem—until it reaches a stopping condition called the **base case**.\n\n"
                "### 2. The Russian Nesting Dolls (Matryoshka) Analogy 🪆\n"
                "Imagine you are given a giant wooden Russian nesting doll:\n"
                "1. You open the giant doll, and inside is a slightly smaller doll.\n"
                "2. You open that doll, and inside is an even smaller doll.\n"
                "3. You keep opening dolls (**Recursive Step**) until you reach the tiniest, solid wooden doll that cannot be opened (**Base Case**).\n"
                "4. Now that you've reached the bottom, you close all the dolls back up one by one in reverse order!\n\n"
                "### 3. The Two Golden Rules of Every Recursive Function\n"
                "Every recursive function must have two parts:\n"
                "1. **The Base Case:** A simple `if` condition where the function stops calling itself and returns a direct answer. (Without this, the function calls itself forever and crashes with a **Stack Overflow** error!).\n"
                "2. **The Recursive Step:** The function calling itself with a smaller input that moves one step closer to the base case.\n\n"
                "### 4. A Clear Code Example: Factorial of a Number\n"
                "The factorial of $4$ (written as $4!$) is: $4 \\times 3 \\times 2 \\times 1 = 24$.\n"
                "Notice that: $4! = 4 \\times 3!$, and $3! = 3 \\times 2!$, and $2! = 2 \\times 1!$. When we reach $1$, the answer is simply $1$!\n\n"
                "```python\n"
                "def factorial(n):\n"
                "    # Rule 1: Base Case (Stopping Point)\n"
                "    if n <= 1:\n"
                "        return 1\n\n"
                "    # Rule 2: Recursive Step (n * factorial of smaller number)\n"
                "    return n * factorial(n - 1)\n\n"
                "# Testing the function\n"
                "print(factorial(3)) # Output: 6\n"
                "```\n\n"
                "### 5. Step-by-Step Call Trace for `factorial(3)`\n"
                "Watch how the computer executes this in memory:\n"
                "1. `factorial(3)` calls `3 * factorial(2)` (waits for answer)\n"
                "2. `factorial(2)` calls `2 * factorial(1)` (waits for answer)\n"
                "3. `factorial(1)` hits the **Base Case** ($n \\le 1$) and returns `1`!\n"
                "4. Now the answers unfold backwards:\n"
                "   - `factorial(2)` computes $2 \\times 1 = 2$ and returns `2`\n"
                "   - `factorial(3)` computes $3 \\times 2 = 6$ and returns `6`!\n\n"
                "That's all recursion is: breaking down a big problem into a smaller identical problem until it's simple enough to solve directly!\n\n"
                "Would you like to see another example (like a countdown timer or reversing a string)?"
            )

        # Requirement 9: Context-Aware Multi-Turn Follow-Ups
        elif is_followup:
            is_simplification = any(w in clean_user for w in ("simpler words", "simpler", "in simple terms", "explain simpler", "more simple", "simple language", "plain words", "simpler terms"))
            is_another_example = any(w in clean_user for w in ("another example", "different example", "second example", "more examples", "give another"))

            if is_simplification:
                if "recursion" in clean_user_hist or "recursion" in clean:
                    return MockGeminiResponse(
                        "### Simpler Explanation: Recursion in Plain Everyday Words 🌟\n\n"
                        "Imagine you are sitting in a crowded movie theater and want to know which row you are sitting in:\n"
                        "1. You ask the person in front of you: *'What row are you in?'*\n"
                        "2. They don't know either, so they ask the person in front of them!\n"
                        "3. This keeps repeating until someone in the very front row (**Row 1**) answers: *'I am in Row 1!'* (This is the **Base Case**—the stopping point).\n"
                        "4. Now the answer travels backwards: Row 2 hears it, then Row 3, until the person in front tells you: *'I am in Row 4, so you are in Row 5!'*\n\n"
                        "**The Takeaway:**\n"
                        "Recursion is just passing a question forward until someone hits the base answer, then passing the solution all the way back!"
                    )
                elif "inheritance" in clean_user_hist or "inheritance" in clean:
                    return MockGeminiResponse(
                        "### Simpler Explanation: Java Inheritance in Plain Everyday Words 🌟\n\n"
                        "Think of inheritance like building a modern Smartphone:\n"
                        "- Instead of designing a phone from scratch, engineers start with a **Basic Phone** (making calls and sending SMS).\n"
                        "- The **Smartphone** simply *inherits* the calling and SMS features from the basic phone without re-writing that code, and then adds new features like a camera and touchscreen!\n\n"
                        "**In Java Code:**\n"
                        "```java\n"
                        "class Smartphone extends BasicPhone { ... }\n"
                        "```\n\n"
                        "**The Takeaway:**\n"
                        "Inheritance saves time and avoids copy-pasting code: the child class automatically gets everything the parent class already knows how to do!"
                    )
                elif "photosynthesis" in clean_user_hist or "photosynthesis" in clean:
                    return MockGeminiResponse(
                        "### Simpler Explanation: Photosynthesis in Plain Everyday Words 🌿\n\n"
                        "Think of a green leaf as a tiny solar-powered kitchen:\n"
                        "1. **Water:** Roots pull water from the soil.\n"
                        "2. **Air:** Leaves breathe in carbon dioxide gas from the air.\n"
                        "3. **Sunlight:** Green chlorophyll acts like solar heat to cook the water and air together!\n"
                        "4. **Food:** The plant makes sugar (glucose) to feed itself and grow big.\n"
                        "5. **Bonus for us:** The plant releases fresh oxygen into the air for humans and animals to breathe!\n\n"
                        "**The Takeaway:**\n"
                        "Photosynthesis is how plants use sunshine to cook their food and give us clean oxygen!"
                    )
                elif "python" in clean_user_hist or "python" in clean:
                    return MockGeminiResponse(
                        "### Simpler Explanation: Python in Plain Everyday Words 🐍\n\n"
                        "Imagine giving cooking instructions to a friend: you want to talk in simple, clear everyday English (*'mix flour, add sugar'*), not in strange binary machine codes.\n\n"
                        "Python was created to read just like plain English! Where other languages require 10 complicated lines just to say hello, Python lets you write:\n"
                        "```python\n"
                        "print('Hello, friend!')\n"
                        "```\n\n"
                        "**The Takeaway:**\n"
                        "Python is friendly, readable, and lets you focus on solving problems rather than memorizing complicated technical rules!"
                    )
                else:
                    return MockGeminiResponse(
                        "### Simpler Explanation in Plain Everyday Words 🌟\n\n"
                        "Let's break this down into the most basic, straightforward terms:\n"
                        "1. **The Core Idea:** At its heart, this concept is about keeping operations organized, predictable, and reliable.\n"
                        "2. **Everyday Analogy:** Think of it like following a simple recipe: each step happens in clear order, taking known ingredients and producing a predictable result.\n"
                        "3. **Why it Matters:** It solves the fundamental problem of preventing mistakes and making systems easier to understand and maintain."
                    )

            elif is_another_example:
                if "inheritance" in clean_user_hist or "inheritance" in clean:
                    return MockGeminiResponse(
                        "### Another Example: Java Inheritance in a Company Payroll System 💼\n\n"
                        "Here is a clear, real-world example of Java inheritance modeling company employees:\n\n"
                        "```java\n"
                        "// Parent class representing generic employee\n"
                        "class Employee {\n"
                        "    protected String name;\n"
                        "    protected double baseSalary;\n\n"
                        "    public Employee(String name, double baseSalary) {\n"
                        "        this.name = name;\n"
                        "        this.baseSalary = baseSalary;\n"
                        "    }\n\n"
                        "    public void displayDetails() {\n"
                        "        System.out.println(\"Employee: \" + name + \" | Base Salary: $\" + baseSalary);\n"
                        "    }\n"
                        "}\n\n"
                        "// Child class specializing for Software Engineer\n"
                        "class SoftwareEngineer extends Employee {\n"
                        "    private double techBonus;\n\n"
                        "    public SoftwareEngineer(String name, double baseSalary, double techBonus) {\n"
                        "        super(name, baseSalary); // Line 1: Reuses parent constructor\n"
                        "        this.techBonus = techBonus;\n"
                        "    }\n\n"
                        "    @Override\n"
                        "    public void displayDetails() {\n"
                        "        super.displayDetails(); // Line 2: Displays base details\n"
                        "        System.out.println(\"Role: Software Engineer | Bonus: $\" + techBonus);\n"
                        "    }\n"
                        "}\n"
                        "```\n\n"
                        "**Why this example works:**\n"
                        "The `SoftwareEngineer` class doesn't need to re-declare `name` or `baseSalary`—it inherits them cleanly from `Employee` and adds only what is unique (`techBonus`)."
                    )
                elif "recursion" in clean_user_hist or "recursion" in clean:
                    return MockGeminiResponse(
                        "### Another Example: Recursion for Reversing a String 🔄\n\n"
                        "Here is another classic and intuitive example of recursion: reversing a word step by step!\n\n"
                        "```python\n"
                        "def reverse_string(s):\n"
                        "    # Rule 1: Base Case (Empty string or single character is already reversed)\n"
                        "    if len(s) <= 1:\n"
                        "        return s\n\n"
                        "    # Rule 2: Recursive Step (Last character + reverse of remainder)\n"
                        "    return s[-1] + reverse_string(s[:-1])\n\n"
                        "# Testing the function\n"
                        "print(reverse_string('chat'))  # Output: 'tahc'\n"
                        "```\n\n"
                        "**Step-by-Step Call Trace:**\n"
                        "1. `reverse_string('chat')` $\\rightarrow$ `'t' + reverse_string('cha')`\n"
                        "2. `reverse_string('cha')` $\\rightarrow$ `'a' + reverse_string('ch')`\n"
                        "3. `reverse_string('ch')` $\\rightarrow$ `'h' + reverse_string('c')`\n"
                        "4. `reverse_string('c')` hits **Base Case** and returns `'c'`!\n"
                        "5. Answers concatenate backwards: `'t' + 'a' + 'h' + 'c' = 'tahc'`!"
                    )
                elif "photosynthesis" in clean_user_hist or "photosynthesis" in clean:
                    return MockGeminiResponse(
                        "### Another Example: Photosynthesis in Aquatic Plants (Elodea Experiment) 🌿💧\n\n"
                        "A classic school science experiment demonstrates photosynthesis happening right before your eyes:\n\n"
                        "1. **The Setup:** Place an aquatic plant like *Elodea* (pondweed) upside down in a beaker filled with water.\n"
                        "2. **The Light Source:** Shine a bright lamp directly at the beaker.\n"
                        "3. **What You Observe:** Within minutes, tiny bubbles of gas continuously bubble up from the cut stem!\n"
                        "4. **The Science:** Those visible bubbles are **pure Oxygen ($O_2$)** being produced in real time as the underwater leaves absorb light and water to perform photosynthesis!"
                    )
                elif "python" in clean_user_hist or "python" in clean:
                    return MockGeminiResponse(
                        "### Another Example: Python for Automated Grade Calculation 💻\n\n"
                        "Here is a practical Python script demonstrating how clean and readable Python code is for daily tasks:\n\n"
                        "```python\n"
                        "student_scores = {'Alex': 88, 'Maya': 95, 'Rohan': 72, 'Priya': 91}\n\n"
                        "# Calculate average and assign letter grades in clean Python\n"
                        "average_score = sum(student_scores.values()) / len(student_scores)\n"
                        "print(f'Class Average: {average_score:.1f}')\n\n"
                        "for name, score in student_scores.items():\n"
                        "    grade = 'A' if score >= 90 else ('B' if score >= 80 else 'C')\n"
                        "    print(f'{name}: {score} -> Grade {grade}')\n"
                        "```"
                    )
                else:
                    return MockGeminiResponse(
                        "### Another Concrete Example\n\n"
                        "Here is an alternative practical example to illustrate this concept from another perspective:\n\n"
                        "1. **Scenario:** Consider a real-world scenario where inputs vary dynamically.\n"
                        "2. **Execution:** By applying the fundamental principles step by step, the mechanism processes inputs cleanly without side effects.\n"
                        "3. **Key Takeaway:** Notice how the underlying logic remains consistent regardless of the specific data or environment."
                    )
            if "recursion" in clean_user_hist or "recursion" in clean:
                return MockGeminiResponse(
                    "### Follow-Up: How the Call Stack Handles Recursion & Avoiding Stack Overflow\n"
                    "Building on our recursion discussion, here is what happens under the hood inside your computer's memory:\n\n"
                    "1. **The Call Stack:** Every time a function calls itself, Python or Java creates a new **Stack Frame** on top of the call stack to remember local variables and where to return.\n"
                    "2. **Unwinding the Stack:** When the base case is hit, stack frames are popped off one by one (LIFO: Last-In, First-Out) as values are returned.\n"
                    "3. **What is Stack Overflow?** If you forget the base case, the function calls itself infinitely until the allocated stack memory is completely exhausted, throwing a `RecursionError: maximum recursion depth exceeded` (in Python) or `java.lang.StackOverflowError` (in Java).\n\n"
                    "**Another Simple Example: Countdown**\n"
                    "```python\n"
                    "def countdown(n):\n"
                    "    if n <= 0:\n"
                    "        print(\"Blast off! 🚀\")\n"
                    "        return\n"
                    "    print(n)\n"
                    "    countdown(n - 1)\n\n"
                    "countdown(3)\n"
                    "# Output:\n"
                    "# 3\n"
                    "# 2\n"
                    "# 1\n"
                    "# Blast off! 🚀\n"
                    "```\n"
                    "Does this help you visualize how the computer steps through each function call?"
                )
            elif "inheritance" in clean_user_hist or "inheritance" in clean:
                return MockGeminiResponse(
                    "### Follow-Up: Method Overriding vs Overloading in Java Inheritance\n"
                    "Continuing from our discussion on Java inheritance, examiners and interviewers frequently ask you to distinguish between **Method Overloading** and **Method Overriding**:\n\n"
                    "| Feature | Method Overloading | Method Overriding |\n"
                    "| :--- | :--- | :--- |\n"
                    "| **Location** | Within the **same class** | Between **superclass and subclass** |\n"
                    "| **Method Signature** | Must have **different parameters** | Must have **exact same name and parameters** |\n"
                    "| **Binding Time** | **Compile-time** (Static Binding) | **Runtime** (Dynamic Binding) |\n"
                    "| **Inheritance Needed?**| No | **Yes**, requires inheritance (`extends`) |\n"
                    "| **Return Type** | Can be different | Must be same or covariant |\n\n"
                    "**Key Tip:** In an interview, always mention that method overriding is the foundation of **Runtime Polymorphism**, where the JVM dynamically dispatches the call via the object's virtual method table (`vtable`)."
                )
            elif "english" in clean_user_hist or "english" in clean:
                return MockGeminiResponse(
                    "### Follow-Up: Daily 15-Minute English Speaking Practice Exercise\n"
                    "Here is a high-impact daily routine you can do right now to build conversational fluency:\n\n"
                    "1. **Minute 1–5: The 1-Minute Impromptu Speech:** Pick a simple topic (e.g. *\"My favorite book\"*, *\"Why I like coding\"*, or *\"My hometown\"*). Speak aloud for 60 seconds without pausing or switching back to your native language.\n"
                    "2. **Minute 5–10: Vocabulary Expansion:** Learn 2 transition words (e.g. *\"Furthermore\"*, *\"In addition\"*, *\"On the other hand\"*, *\"Consequently\"*) and practice using them in sentences.\n"
                    "3. **Minute 10–15: Voice Note Self-Audit:** Record a 30-second voice note on your phone describing your day. Listen back to it. Notice your pronunciation, pace, and where you hesitated—you'll naturally improve every time you listen!"
                )
            elif "photosynthesis" in clean_user_hist or "photosynthesis" in clean:
                return MockGeminiResponse(
                    "### Follow-Up: Light vs Dark Reactions in Photosynthesis\n"
                    "Building on our photosynthesis overview, photosynthesis actually happens in two synchronized stages inside the plant's chloroplasts:\n\n"
                    "1. **Light-Dependent Reactions (The Solar Power Stage):**\n"
                    "   - **Location:** In the thylakoid membranes of chloroplasts.\n"
                    "   - **Process:** Sunlight is absorbed by chlorophyll to split water molecules ($H_2O$) into hydrogen ions and oxygen ($O_2$).\n"
                    "   - **Output:** Oxygen is released into the air, and energy-storage molecules (ATP and NADPH) are charged up.\n"
                    "2. **Light-Independent Reactions / Calvin Cycle (The Sugar Factory Stage):**\n"
                    "   - **Location:** In the stroma (the fluid-filled space) of chloroplasts.\n"
                    "   - **Process:** Uses the stored ATP/NADPH energy to combine carbon dioxide ($CO_2$) molecules into glucose sugar ($C_6H_{12}O_6$).\n"
                    "   - **Key Fact:** This stage does not need direct photons of light, so it can run day and night as long as energy reserves are available!"
                )
            elif any(w in clean_user_hist for w in ("timetable", "study plan", "exam timetable", "exams", "study schedule")):
                return MockGeminiResponse(
                    "### Follow-Up: How to Adjust Your Timetable for Weak Subjects\n"
                    "To get the most out of your timetable when you have subjects you find difficult:\n\n"
                    "1. **Front-Load the Weak Subject:** Always schedule your hardest subject during your peak energy window (usually early morning or right after your first break). Never leave it for late at night when mental fatigue sets in.\n"
                    "2. **Interleaving:** Rather than spending 6 consecutive hours on one topic, study 2 hours of your weak subject, take a break, then switch to a subject you enjoy before returning.\n"
                    "3. **Active Problem Solving:** With weak subjects, passive reading gives a false sense of mastery. Immediately test yourself by working through textbook problems with the solutions covered up!"
                )
            elif "placement" in clean_user_hist or "placement" in clean:
                return MockGeminiResponse(
                    "### Follow-Up: Mastering the STAR Method for HR & Placement Interviews\n"
                    "Interviewers use behavioral questions (e.g. *\"Tell me about a difficult bug you solved\"* or *\"Describe a time you worked under pressure\"*) to predict how you will perform on the job. Use the **STAR method**:\n\n"
                    "- **S – Situation:** Set the scene in 1–2 sentences (*\"During our 6th-semester college project, our team built an automated attendance app with a deadline in two weeks.\"*)\n"
                    "- **T – Task:** State your specific responsibility (*\"My task was to design the database and integrate real-time face recognition.\"*)\n"
                    "- **A – Action:** Detail the concrete steps YOU took (*\"When OpenCV model inferences were taking 3 seconds per face, I optimized frame resolution and added multithreading to process camera feeds concurrently.\"*)\n"
                    "- **R – Result:** Quantify the successful outcome (*\"We reduced latency to 0.4 seconds, successfully demonstrated the project to the college panel, and secured the top project grade.\"*)"
                )
            else:
                return MockGeminiResponse(
                    "### Follow-Up Explanation & Deep Dive\n"
                    "Building on our previous discussion, here is a deeper look into the mechanism:\n\n"
                    "1. **Underlying Principles:** Connects fundamental theory with practical execution steps.\n"
                    "2. **Practical Scenario:** Consider how this behaves under edge cases or high concurrency.\n"
                    "3. **Summary & Key Takeaway:** Always evaluate trade-offs between simplicity, maintainability, and performance.\n\n"
                    "Would you like to explore another specific scenario or see an additional code example?"
                )

        elif "clarification" in clean or "follow-up" in clean or "example" in clean:
            return MockGeminiResponse(
                "### 1. Direct Answer\n"
                "Building on our discussion, this follow-up clarifies the operational mechanism with a concrete scenario.\n\n"
                "### 2. Simple Explanation\n"
                "Think of an airline reservation system: when two passengers attempt to book the last available seat simultaneously, an atomic locking mechanism is required so only one booking commits.\n\n"
                "### 3. Detailed Explanation\n"
                "In distributed architectures, maintaining consistency across nodes requires consensus protocols (e.g. Raft or Paxos) to prevent split-brain state divergence.\n\n"
                "### 4. Practical Example & Code\n"
                "```python\n"
                "# Distributed lock pattern using atomic Compare-and-Swap\n"
                "def acquire_distributed_lock(redis_client, lock_key, token, ttl_seconds):\n"
                "    # Line 1: NX flag sets key only if it does not already exist\n"
                "    return redis_client.set(lock_key, token, nx=True, ex=ttl_seconds)\n"
                "```\n\n"
                "### 5. Important Points to Remember\n"
                "- Atomic primitives prevent race conditions in concurrent and distributed environments.\n"
                "- Always release locks in a finally block to avoid deadlocks on unexpected failures."
            )
        else:
            # Dynamic topic extraction from user question
            topic_label = None
            subject_patterns = [
                r"(?:tell\s+(?:me\s+)?about|what\s+is|explain|describe|overview\s+of|how\s+does)\s+([a-zA-Z0-9#+.\s-]+?)(?:\?|\.|$)",
                r"Question:\s*\"([^\"]+)\"",
                r"context\s*/\s*viva topic:\s*([^\]\n]+)",
            ]
            for pat in subject_patterns:
                m = re.search(pat, last_user_prompt or prompt_text, re.IGNORECASE)
                if m:
                    candidate = m.group(1).strip()
                    candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=re.IGNORECASE).strip()
                    if candidate and len(candidate) > 1 and candidate.lower() not in ("it", "this", "that", "more", "explaination", "explanation", "follow-up", "clarification"):
                        topic_label = candidate.title()
                        break

            if not topic_label:
                for kw, title in [
                    ("c++", "C++ Programming"),
                    ("cpp", "C++ Programming"),
                    ("javascript", "JavaScript Runtime & Engine"),
                    ("js", "JavaScript Runtime & Engine"),
                    ("react", "React.js Framework"),
                    ("sql", "SQL & Relational Databases"),
                    ("acid", "ACID Transactions"),
                    ("index", "Database Indexing"),
                    ("normalization", "Database Normalization"),
                    ("quicksort", "QuickSort Algorithm"),
                    ("mergesort", "MergeSort Algorithm"),
                    ("virtual memory", "Virtual Memory"),
                    ("mutex", "Mutex Synchronization"),
                    ("docker", "Docker & Containerization"),
                    ("kubernetes", "Kubernetes Orchestration"),
                    ("thread", "Multithreading & Concurrency"),
                    ("cache", "Caching Architecture"),
                    ("tree", "Tree Data Structures"),
                    ("graph", "Graph Algorithms"),
                    ("hash", "Hash Tables & Hashing"),
                ]:
                    if kw in clean:
                        topic_label = title
                        break

            if not topic_label:
                has_academic_request = any(w in clean_user for w in (
                    "explain", "describe", "define", "what is", "difference between", "compare", "lecture", "topic"
                )) and any(w in clean_user for w in (
                    "concept", "theory", "principle", "mechanism", "subject", "syllabus", "exam", "course"
                ))
                if has_academic_request:
                    topic_label = "Educational Concept"
                elif "sky blue" in clean_user:
                    return MockGeminiResponse(
                        "The sky looks blue because of Rayleigh scattering! As sunlight reaches Earth's atmosphere, "
                        "it scatters in all directions against gas molecules. Blue light travels in shorter, smaller waves than other colors, "
                        "so it is scattered much more widely across the sky."
                    )
                elif any(w in clean_user for w in ("airplane", "aeroplane", "how do planes fly")):
                    return MockGeminiResponse(
                        "Airplanes fly thanks to four aerodynamic forces: lift, gravity, thrust, and drag. "
                        "The wings are curved so air moves faster over the top than the bottom, creating lower pressure above the wing and generating lift."
                    )
                else:
                    return MockGeminiResponse(
                        "I'm right here with you! What's on your mind today, or what would you like to explore?"
                    )

            # Check if this is a software engineering topic vs general academic topic
            is_cs_subject = any(w in clean for w in (
                "code", "algorithm", "function", "variable", "pointer", "memory", "database", "sql",
                "thread", "process", "mutex", "cache", "server", "compiler", "cpu", "programming",
                "oop", "encapsulation", "class", "object", "interface", "software"
            ))

            if is_cs_subject:
                return MockGeminiResponse(
                    f"### 1. Direct Answer\n"
                    f"**{topic_label}** is a core computer science and engineering concept that directly governs system coordination, data organization, and execution correctness.\n\n"
                    f"### 2. Simple Explanation\n"
                    f"Think of {topic_label} like a standardized air-traffic control protocol: it enforces clear, predictable rules so concurrent operations proceed safely without colliding or corrupting system state.\n\n"
                    f"### 3. Detailed Explanation\n"
                    f"1. **Core Architectural Mechanism:** Coordinates inputs, algorithmic state lifecycles, and underlying resources with precision.\n"
                    f"2. **Error Boundaries:** Isolates unexpected failures and prevents state corruption across components.\n"
                    f"3. **Trade-offs:** Weighs time complexity vs space complexity, latency vs throughput, and memory overhead.\n\n"
                    f"### 4. Practical Example & Code\n"
                    f"```python\n"
                    f"# Practical implementation pattern for {topic_label}\n"
                    f"def process_operation(data_payload):\n"
                    f"    # Line 1: Defensive validation prevents invalid execution states\n"
                    f"    if not data_payload:\n"
                    f"        return None\n"
                    f"    # Line 2: Process core workflow\n"
                    f"    result = [item.strip() for item in data_payload if item]\n"
                    f"    # Line 3: Return verified output\n"
                    f"    return result\n"
                    f"```\n\n"
                    f"**Line Breakdown & Logic:**\n"
                    f"- **Line 1:** Defensive validation guards against unexpected null pointer or memory faults.\n"
                    f"- **Line 2:** Transforms items in linear $O(n)$ time without side effects.\n\n"
                    f"### 5. Important Points to Remember\n"
                    f"- Always state the formal technical definition of {topic_label} first in examinations.\n"
                    f"- Walk through the internal execution sequence rather than treating it as a black box.\n"
                    f"- Highlight computational complexity, trade-offs, and failure recovery modes."
                )
            else:
                return MockGeminiResponse(
                    f"### 1. Direct Answer\n"
                    f"**{topic_label}** is an important concept in learning and academic study that helps us understand how fundamental principles, mechanisms, and problem-solving techniques work in practice.\n\n"
                    f"### 2. Simple Explanation & Analogy\n"
                    f"Think of {topic_label} like a key puzzle piece: on its own, it has distinct rules and characteristics, but when combined with surrounding fundamentals, it helps form a complete, coherent picture.\n\n"
                    f"### 3. Detailed Explanation\n"
                    f"1. **Core Principle:** Focuses on foundational rules, clear definitions, and underlying mechanisms.\n"
                    f"2. **Practical Application:** Shows how theoretical knowledge translates into real-world applications and academic problem solving.\n"
                    f"3. **Analytical Understanding:** Highlights trade-offs, common misconceptions, and best practices.\n\n"
                    f"### 4. Practical Example\n"
                    f"When working with {topic_label}, always begin by identifying the core inputs or premises, applying standard rules step by step, and verifying the outcome against expected benchmarks.\n\n"
                    f"### 5. Key Points to Remember\n"
                    f"- Understand the fundamental definition and purpose of {topic_label}.\n"
                    f"- Connect theoretical concepts to practical, real-world examples.\n"
                    f"- Break complex problems down step-by-step for clear explanations."
                )


class MockGeminiClient:
    """Mock Gemini client mimicking genai.Client for testing and offline development."""
    def __init__(self, api_key: str = "mock"):
        self.api_key = api_key
        self.models = MockGeminiModels()


# ==============================================================================
# Client Factory
# ==============================================================================

def get_gemini_client(custom_client: Optional[Any] = None) -> Any:
    """Returns an active Gemini client, or a MockGeminiClient when key is 'mock' or 'demo'.

    Raises:
        GeminiConfigurationError: If no API key is set and no custom client provided.
    """
    if custom_client is not None:
        return custom_client

    settings = get_settings()
    api_key = (settings.GEMINI_API_KEY or "").strip()

    if not api_key:
        raise GeminiConfigurationError("GEMINI_API_KEY environment variable is not configured.")

    if api_key.lower() in ("mock", "demo", "test"):
        logger.info("Using MockGeminiClient for offline simulation/testing.")
        return MockGeminiClient(api_key=api_key)

    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise GeminiConfigurationError(f"Failed to initialize Gemini client: {str(e)}")


# ==============================================================================
# Service Methods
# ==============================================================================

def ask_gemini_doubt(
    message: str,
    history: Optional[List[DoubtChatMessage]] = None,
    context: Optional[str] = None,
    custom_client: Optional[Any] = None,
) -> str:
    """Sends a student doubt/question to Google Gemini with conversation history.

    Args:
        message: The student's current question or doubt.
        history: Optional list of previous DoubtChatMessage turns.
        context: Optional topic or viva question context to seed understanding.
        custom_client: Optional client override for testing.

    Returns:
        str: Patient academic tutor explanation.

    Raises:
        GeminiConfigurationError: If API key is not configured.
        GeminiRateLimitError: If rate limits or quotas are hit.
        GeminiServiceError: If Gemini API fails.
    """
    if not message or not message.strip():
        raise GeminiServiceError("Question message cannot be empty.", status_code=400)

    clean_message = message.strip()
    if len(clean_message) > 5000:
        raise GeminiServiceError("Question message is too long (maximum 5,000 characters).", status_code=400)

    settings = get_settings()
    model_name = settings.GEMINI_MODEL or "gemini-1.5-flash"
    client = get_gemini_client(custom_client=custom_client)

    # Prepare multi-turn contents
    contents: List[types.Content] = []

    # If context is provided, attach it as background context without overriding casual or different queries
    if context and context.strip():
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"[Current Study Context: {context.strip()} (Note: Only use this context if the user's message is directly asking about it. If the user asks about other topics, casual conversation, entertainment, or general questions, respond to their actual question naturally without forcing this context.)]"
                )]
            )
        )
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text="Understood! I'll keep this topic in mind for relevant questions, while chatting naturally about whatever else the user wants to discuss."
                )]
            )
        )

    # Append conversation history
    if history:
        for turn in history:
            turn_role = "model" if turn.role in ("assistant", "model") else "user"
            contents.append(
                types.Content(
                    role=turn_role,
                    parts=[types.Part.from_text(text=turn.content)]
                )
            )

    # Append current user question
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=clean_message)]
        )
    )

    # Configure tutor parameters
    config = types.GenerateContentConfig(
        system_instruction=ACADEMIC_TUTOR_SYSTEM_INSTRUCTION,
        temperature=0.7,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        if not response or not hasattr(response, "text") or not response.text:
            raise GeminiServiceError("Gemini returned an empty response. Please try rephrasing your question.")

        return response.text

    except errors.APIError as e:
        status_code = getattr(e, "code", 502)
        message_lower = str(e).lower()
        if status_code in (401, 403) or "api_key" in message_lower or "api key" in message_lower or "unauthenticated" in message_lower:
            logger.error(f"Gemini API authentication failed: {e}")
            raise GeminiConfigurationError("Invalid or unauthorized Gemini API key.")
        if status_code in (408, 504) or "timeout" in message_lower or "deadline" in message_lower:
            logger.error(f"Gemini API timed out: {e}")
            raise GeminiTimeoutError("Gemini AI request timed out. Please try again.")
        if status_code == 429 or "quota" in message_lower or "resource has been exhausted" in message_lower:
            logger.warning(f"Gemini API rate limit exceeded: {e}")
            raise GeminiRateLimitError(
                "Gemini AI rate limit reached. Please wait a few moments and try your question again."
            )
        logger.error(f"Gemini API error ({status_code}): {e}")
        raise GeminiServiceError(f"Gemini AI service error: {str(e)}", status_code=502)

    except (GeminiServiceError, GeminiConfigurationError, GeminiRateLimitError, GeminiTimeoutError):
        raise

    except Exception as e:
        logger.exception(f"Unexpected error communicating with Gemini AI: {e}")
        raise GeminiServiceError(f"Failed to communicate with Gemini assistant: {str(e)}", status_code=502)


def generate_gemini_model_answer(
    question_text: str,
    subject: str = "Computer Science",
    topic: str = "General",
    difficulty: str = "Medium",
    target_level: str = "Intermediate",
    candidate_answer: Optional[str] = None,
    custom_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generates an authoritative, independent model answer and educational explanation using Google Gemini.

    Evaluates the question independently of whatever the candidate submitted.

    Args:
        question_text: Original interview or viva question prompt.
        subject: Subject domain (e.g. Java, Operating Systems, DBMS).
        topic: Specific subtopic (e.g. Polymorphism, Deadlocks).
        difficulty: Question difficulty (Easy, Medium, Hard).
        target_level: Target candidate level (Beginner, Intermediate, Advanced).
        candidate_answer: Optional submitted candidate response (used only for context, never for awarding free marks).
        custom_client: Optional client override for testing.

    Returns:
        Dict with keys:
            - correct_answer (str)
            - simple_explanation (str)
            - detailed_explanation (str)
            - example (Optional[str]) with code snippets and line-by-line breakdown when programming
            - key_points_to_remember (List[str])
            - provider (str)
            - model (str)
            - is_generated (bool)
            - disclaimer (str)

    Raises:
        GeminiConfigurationError: If Gemini is not configured.
        GeminiRateLimitError: If Gemini rate limits are hit.
        GeminiServiceError: If Gemini fails to generate an answer.
    """
    if not question_text or not question_text.strip():
        raise GeminiServiceError("Question prompt cannot be empty.", status_code=400)

    clean_question = question_text.strip()
    if len(clean_question) > 5000:
        raise GeminiServiceError("Question prompt is too long (maximum 5,000 characters).", status_code=400)

    settings = get_settings()
    model_name = settings.GEMINI_MODEL or "gemini-1.5-flash"
    client = get_gemini_client(custom_client=custom_client)

    candidate_context_note = ""
    if candidate_answer and candidate_answer.strip():
        candidate_context_note = (
            f"\n[Submitted Candidate Response (Provided for evaluation context ONLY; "
            f"NEVER copy, repeat, or adopt any errors, gaps, or misconceptions from this into the model answer)]:\n"
            f"\"{candidate_answer.strip()}\"\n"
        )

    prompt = (
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

    config = types.GenerateContentConfig(
        system_instruction=GEMINI_MODEL_ANSWER_SYSTEM_PROMPT,
        temperature=0.3,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        if not response or not hasattr(response, "text") or not response.text:
            raise GeminiServiceError("Gemini returned an empty model answer response.")

        raw_data = extract_json_payload(response.text)
        if not isinstance(raw_data, dict):
            raise GeminiServiceError("Gemini model answer was not a JSON dictionary.")

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
            or "AI-generated model solution for viva preparation. Candidate should review standard course material."
        )

        improved_ans = str(raw_data.get("improved_answer") or "").strip() or None

        return {
            "correct_answer": correct_ans,
            "simple_explanation": simple_exp,
            "detailed_explanation": detailed_exp,
            "example": example,
            "key_points_to_remember": clean_points,
            "improved_answer": improved_ans,
            "provider": "Google Gemini",
            "model": model_name,
            "is_generated": True,
            "disclaimer": disclaimer,
            "error_message": None,
        }

    except errors.APIError as e:
        status_code = getattr(e, "code", 502)
        message_lower = str(e).lower()
        if status_code in (401, 403) or "api_key" in message_lower or "api key" in message_lower or "unauthenticated" in message_lower:
            logger.error(f"Gemini API authentication failed during model answer generation: {e}")
            raise GeminiConfigurationError("Invalid or unauthorized Gemini API key.")
        if status_code in (408, 504) or "timeout" in message_lower or "deadline" in message_lower:
            logger.error(f"Gemini API timed out during model answer generation: {e}")
            raise GeminiTimeoutError("Gemini AI request timed out. Please try again.")
        if status_code == 429 or "quota" in message_lower or "resource has been exhausted" in message_lower:
            logger.warning(f"Gemini API rate limit exceeded during model answer generation: {e}")
            raise GeminiRateLimitError(
                "Gemini AI rate limit reached while generating model answer. Please wait a moment."
            )
        logger.error(f"Gemini API error during model answer generation ({status_code}): {e}")
        raise GeminiServiceError(f"Gemini AI service error: {str(e)}", status_code=502)

    except (GeminiServiceError, GeminiConfigurationError, GeminiRateLimitError, GeminiTimeoutError):
        raise

    except Exception as e:
        logger.exception(f"Unexpected error generating model answer with Gemini: {e}")
        raise GeminiServiceError(f"Failed to generate model answer: {str(e)}", status_code=502)
