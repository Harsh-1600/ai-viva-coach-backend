"""AI Service module for Anthropic Claude integration and question generation."""

import json
import logging
import re
from typing import List, Optional, Any
import anthropic

from app.config import get_settings
from app.prompts.question_generation import (
    EXAMINER_SYSTEM_PROMPT,
    build_question_generation_prompt,
)
from app.prompts.evaluation_and_followup import (
    EXAMINER_EVALUATION_SYSTEM_PROMPT,
    build_evaluation_prompt,
)
from app.schemas import GenerateQuestionsRequest, GeneratedQuestion
from app.services import reference_answers

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Base exception for AI Service errors."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AIConfigurationError(AIServiceError):
    """Raised when AI service is not properly configured (e.g., missing API key)."""
    def __init__(self, message: str = "Anthropic API key is not configured."):
        super().__init__(message, status_code=503)


class AIRateLimitError(AIServiceError):
    """Raised when Anthropic Claude API rate limit is exceeded."""
    def __init__(self, message: str = "AI service rate limit reached. Please try again shortly."):
        super().__init__(message, status_code=429)


class AIResponseParseError(AIServiceError):
    """Raised when AI response cannot be parsed or validated."""
    def __init__(self, message: str = "Failed to parse structured question output from AI."):
        super().__init__(message, status_code=502)


class MockBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class MockMessageResponse:
    def __init__(self, text: str):
        self.content = [MockBlock(text)]


class MockAnthropicClient:
    """Offline developer examiner used when ANTHROPIC_API_KEY is set to 'mock' or 'demo'."""
    class messages:
        @staticmethod
        def create(model, max_tokens, temperature, system, messages, **kwargs):
            prompt = messages[0]["content"] if messages else ""
            if "0-100 scoring rubric" in prompt or "0 to 100 points" in (system or ""):
                # 100-point Rubric Detailed Evaluation request
                answer = ""
                if "<candidate_answer>" in prompt:
                    try:
                        answer = prompt.split("<candidate_answer>")[1].split("</candidate_answer>")[0]
                    except Exception:
                        answer = ""
                elif 'Candidate\'s Answer:\n"""' in prompt:
                    try:
                        answer = prompt.split('Candidate\'s Answer:\n"""')[1].split('"""')[0]
                    except Exception:
                        answer = ""
                elif 'Candidate\'s Answer: """' in prompt:
                    try:
                        answer = prompt.split('Candidate\'s Answer: """')[1].split('"""')[0]
                    except Exception:
                        answer = ""

                clean_ans = answer.strip().lower()
                has_substantive_content = len(clean_ans) >= 40 and any(w in clean_ans for w in [
                    "polymorphism", "virtual memory", "mutex", "acid", "generator", "indexing", "paging", "inheritance", "encapsulation", "many forms", "pages", "frames", "dynamic dispatch", "method"
                ])
                is_empty_or_idk = not has_substantive_content and (
                    len(clean_ans) == 0
                    or any(x in clean_ans for x in [
                        "don't know", "dont know", "do not know", "not know", "idk",
                        "no idea", "pass", "skip", "empty answer", "unfamiliarity"
                    ])
                    or clean_ans in ["not sure", "unsure", "i am not sure", "i'm not sure"]
                    or (len(clean_ans) < 35 and any(x in clean_ans for x in ["not sure", "unsure"]))
                )

                # Extract question and topic from prompt
                q_match = re.search(r"<examiner_question>([\s\S]*?)</examiner_question>", prompt)
                q_text = q_match.group(1).strip().lower() if q_match else ""
                top_match = re.search(r"Topic:\s*([^\n\r]+)", prompt)
                topic_text = top_match.group(1).strip().lower() if top_match else ""

                stop_words = {
                    "the", "and", "for", "are", "can", "not", "all", "any", "but", "how", "why", "who",
                    "has", "had", "have", "explain", "mechanism", "what", "where", "when", "which", "with",
                    "your", "from", "about", "this", "that", "these", "those", "does", "operating", "system",
                    "systems", "primary", "describe", "compare", "contrast", "detail", "between", "during",
                    "state", "would", "could", "should", "work", "works", "function", "using", "into", "candidate"
                }

                combined_context = f"{topic_text} {q_text}"
                key_context_terms = {
                    t for t in re.findall(r"[a-z]{3,}", combined_context) if t not in stop_words
                }
                ans_terms = set(re.findall(r"[a-z]{3,}", clean_ans))

                is_polymorphism_context = any(k in combined_context for k in ["polymorphism", "java", "oop", "object-oriented"])
                is_db_misconception = is_polymorphism_context and any(w in clean_ans for w in ["database", "permanently", "hard drive", "sql", "store data"])

                # Detect explicit wrong/incorrect markers, contradictions or false domains
                is_explicitly_wrong = is_db_misconception or any(w in clean_ans for w in [
                    "wrong", "incorrect", "false", "potato", "garbage", "nonsense", "not true", 
                    "fake", "no clue", "bad answer", "random", "fail", "test wrong",
                    "gpu", "graphics card", "texture", "render 3d", "video game", "wifi", "router", "power supply"
                ])

                has_technical_overlap = len(key_context_terms.intersection(ans_terms)) > 0 if key_context_terms else True
                is_unrelated = (not has_technical_overlap and len(clean_ans) > 0 and not is_explicitly_wrong) or any(w in clean_ans for w in [
                    "unrelated", "random", "baking", "cake", "photosynthesis", "sunlight", "plants", "cricket", "weather"
                ])

                # Deep mechanics markers
                mechanics_markers = ["tlb", "fault", "trap", "swap", "cr3", "atomic", "cas", "scheduler", "critical section", "b-tree", "acid", "inorder", "handshake"]
                deep_count = sum(1 for m in mechanics_markers if m in clean_ans)

                # Domain classification
                is_db_context = any(k in combined_context for k in ["database", "sql", "acid", "transaction", "index", "normalization", "rdbms", "table"])
                is_ds_context = any(k in combined_context for k in ["binary search tree", "bst", "tree", "graph", "queue", "stack", "linked list", "sort", "algorithm", "data structure"])
                is_net_context = any(k in combined_context for k in ["tcp", "udp", "network", "packet", "protocol", "http", "handshake", "socket"])
                is_oop_context = any(k in combined_context for k in ["oop", "object-oriented", "encapsulation", "inheritance", "polymorphism", "abstraction", "interface", "class"])

                has_partial_flag = any(p in clean_ans for p in [
                    "partial", "partially", "limited", "incomplete", "some parts", "only part", "just basic", "not sure how", "not sure about"
                ])

                # Topic-specific strong matching
                is_poly_strong = is_polymorphism_context and not is_explicitly_wrong and not is_unrelated and not has_partial_flag and (
                    ("method" in clean_ans and "object" in clean_ans)
                    or any(w in clean_ans for w in ["dynamic dispatch", "method overriding", "runtime polymorphism", "vtable", "virtual method"])
                    or ("interface" in clean_ans and any(w in clean_ans for w in ["implement", "runtime", "dispatch", "subclass"]))
                    or ("override" in clean_ans and any(w in clean_ans for w in ["subclass", "parent", "method"]))
                )
                is_poly_partial = is_polymorphism_context and not is_empty_or_idk and not is_explicitly_wrong and not is_unrelated and (
                    "many forms" in clean_ans or "one thing" in clean_ans or "multiple forms" in clean_ans or "overloading" in clean_ans or has_partial_flag
                ) and not is_poly_strong

                # Semantic recognition for Virtual Memory & Operating Systems
                is_vm_context = any(w in combined_context for w in ["virtual memory", "paging", "mmu", "page table", "address translation"])
                vm_requires_fault = is_vm_context and any(w in q_text for w in ["fault", "page fault"])
                has_fault_coverage = any(w in clean_ans for w in ["fault", "trap", "disk", "missing", "not present", "not in physical", "invalid"])
                is_vm_shallow = is_vm_context and len(clean_ans) < 80 and not has_fault_coverage

                is_vm_strong = is_vm_context and not is_explicitly_wrong and not is_unrelated and not is_vm_shallow and (
                    (deep_count >= 2 and len(clean_ans) > 70)
                    or (any(w in clean_ans for w in ["map", "mapping", "translates", "translation", "decouple", "isolate", "illusion", "abstraction", "protect"])
                        and any(w in clean_ans for w in ["page table", "mmu", "physical", "ram", "frame", "disk", "secondary storage"])
                        and any(w in clean_ans for w in ["address", "virtual", "space", "memory"])
                        and (not vm_requires_fault or has_fault_coverage))
                )

                # Semantic recognition for Concurrency & Synchronization
                is_concurrency_strong = any(w in combined_context for w in ["mutex", "lock", "concurrency", "thread", "synchronization", "deadlock", "race condition"]) and not is_explicitly_wrong and not is_unrelated and not has_partial_flag and (
                    any(w in clean_ans for w in ["mutual exclusion", "critical section", "race condition", "synchroniz", "deadlock", "atomic"])
                    and any(w in clean_ans for w in ["thread", "process", "lock", "shared", "resource", "access", "mutex"])
                )

                # Databases & Transactions
                is_db_strong = is_db_context and not is_explicitly_wrong and not is_unrelated and not has_partial_flag and (
                    any(w in clean_ans for w in ["atomicity", "consistency", "isolation", "durability", "acid", "transaction", "commit", "rollback", "b-tree", "foreign key", "primary key"])
                )

                # Data Structures & Algorithms
                is_ds_strong = is_ds_context and not is_explicitly_wrong and not is_unrelated and not has_partial_flag and (
                    any(w in clean_ans for w in ["left", "right", "smaller", "greater", "larger", "root", "node", "log", "o(log", "traversal", "inorder", "balance", "leaf", "parent"])
                )

                # Computer Networks
                is_net_strong = is_net_context and not is_explicitly_wrong and not is_unrelated and not has_partial_flag and (
                    any(w in clean_ans for w in ["connection", "reliable", "unreliable", "handshake", "stream", "packet", "datagram", "acknowledg", "syn", "ack", "loss", "ordered"])
                )

                # General OOP
                is_oop_strong = is_oop_context and not is_explicitly_wrong and not is_unrelated and not has_partial_flag and (
                    is_poly_strong
                    or (any(w in clean_ans for w in ["bundle", "hide", "private", "getter", "restrict", "internal", "wrap"]) and "encapsulation" in clean_ans)
                    or (any(w in clean_ans for w in ["subclass", "superclass", "parent", "child", "extend", "derive", "reuse"]) and "inheritance" in clean_ans)
                    or ("encapsulation" in clean_ans and "inheritance" in clean_ans)
                )

                # Detect circular tautology (repeating question terms without explaining mechanisms)
                is_circular = False
                if not is_empty_or_idk and not is_unrelated and not is_explicitly_wrong:
                    if any(phrase in clean_ans for phrase in [
                        "is when a transaction is a transaction",
                        "is when it happens",
                        "polymorphism is polymorphism",
                        "is something that does things",
                        "a transaction is when a transaction",
                    ]):
                        is_circular = True
                    elif (
                        len(clean_ans) < 90
                        and not any(m in clean_ans for m in mechanics_markers)
                        and not any(w in clean_ans for w in ["dispatch", "overrid", "interfac", "virtual", "allocat", "atomic", "lock", "queue", "pointer", "travers", "handshake", "packet", "socket", "rollback", "commit", "acid"])
                        and (
                            any(f"{w} is {w}" in clean_ans or f"{w} is when {w}" in clean_ans or f"{w} is a {w}" in clean_ans for w in ["transaction", "polymorphism", "paging", "mutex", "deadlock", "process"])
                        )
                    ):
                        is_circular = True

                # Substantive valid technical answer across general CS topics
                is_general_strong = (
                    not is_polymorphism_context
                    and not is_vm_shallow
                    and not is_poly_partial
                    and not is_circular
                    and not has_partial_flag
                    and has_technical_overlap
                    and not is_explicitly_wrong
                    and not is_unrelated
                    and len(clean_ans) >= 25
                    and (not vm_requires_fault or has_fault_coverage or len(clean_ans) > 70)
                )

                is_strong = (
                    (is_poly_strong and not is_poly_partial)
                    or is_vm_strong
                    or is_concurrency_strong
                    or is_db_strong
                    or is_ds_strong
                    or is_net_strong
                    or (is_oop_strong and not is_poly_partial)
                    or is_general_strong
                ) and not is_explicitly_wrong and not is_unrelated and not is_poly_partial and not is_circular and not has_partial_flag and not is_vm_shallow

                is_partial = not is_strong and not is_empty_or_idk and not is_unrelated and not is_explicitly_wrong and not is_circular and (
                    is_poly_partial
                    or has_partial_flag
                    or is_vm_shallow
                    or (vm_requires_fault and not has_fault_coverage)
                )

                if is_empty_or_idk:
                    res_dict = {
                        "is_correct": False,
                        "numerical_score": 0.0,
                        "brief_explanation": "Candidate indicated unfamiliarity with the topic or did not provide an answer.",
                        "correct_concepts_identified": [],
                        "missing_concepts": ["Primary concept definitions and operational mechanics"],
                        "technical_inaccuracies": [],
                        "specific_improvement_suggestions": [
                            "Review foundational definitions and primary workflows before attempting this topic."
                        ],
                        "suggested_ideal_outline": [
                            "Define core terminology and architectural purpose",
                            "Explain step-by-step translation or execution flow",
                            "Discuss memory overhead and real-world system trade-offs"
                        ],
                        "next_difficulty_recommendation": "easier",
                        "rubric_breakdown": {
                            "technical_correctness": 0.0,
                            "conceptual_understanding": 0.0,
                            "completeness": 0.0,
                            "clarity": 0.0,
                            "relevance": 0.0
                        },
                        "should_follow_up": False,
                        "follow_up_reason": "shallow",
                        "follow_up_question": None,
                        "is_ai_estimate": True
                    }
                elif is_unrelated:
                    res_dict = {
                        "is_correct": False,
                        "numerical_score": 0.0,
                        "brief_explanation": "Irrelevant answer. The response is completely unrelated to the question asked.",
                        "correct_concepts_identified": [],
                        "missing_concepts": [
                            "Core concepts directly answering the posed question"
                        ],
                        "technical_inaccuracies": [
                            "The submitted answer is unrelated to the question asked."
                        ],
                        "specific_improvement_suggestions": [
                            "Read the question carefully and ensure your answer addresses the requested topic directly."
                        ],
                        "suggested_ideal_outline": [
                            "Address the specific topic and terminology requested",
                            "Explain key mechanisms concisely",
                            "Provide a relevant example"
                        ],
                        "next_difficulty_recommendation": "easier",
                        "rubric_breakdown": {
                            "technical_correctness": 0.0,
                            "conceptual_understanding": 0.0,
                            "completeness": 0.0,
                            "clarity": 0.0,
                            "relevance": 0.0
                        },
                        "should_follow_up": False,
                        "follow_up_reason": "misconception",
                        "follow_up_question": None,
                        "is_ai_estimate": True
                    }
                elif is_explicitly_wrong:
                    misconception_text = (
                        "Incorrect answer. The response confuses polymorphism with database persistence or permanent data storage. "
                        "Polymorphism is an object-oriented programming concept where the same method call exhibits different behaviors depending on the target object."
                        if is_db_misconception else
                        "Incorrect answer. The response contains fundamental factual inaccuracies and technical misconceptions."
                    )
                    inaccuracies_list = [
                        "Asserted that polymorphism is used to store data permanently in a database."
                    ] if is_db_misconception else [
                        "The proposed explanation is factually incorrect and does not reflect actual system implementation."
                    ]
                    missing_list = [
                        "Method overriding (runtime polymorphism) and method overloading (compile-time polymorphism)",
                        "Dynamic method dispatch via JVM vtables",
                        "Subclass hierarchy and interface implementations"
                    ] if is_db_misconception else [
                        "Core architectural mechanism and factual principles",
                        "Accurate technical definitions and workflow"
                    ]
                    res_dict = {
                        "is_correct": False,
                        "numerical_score": 0.0,
                        "brief_explanation": misconception_text,
                        "correct_concepts_identified": [],
                        "missing_concepts": missing_list,
                        "technical_inaccuracies": inaccuracies_list,
                        "specific_improvement_suggestions": [
                            "Review object-oriented programming fundamentals: polymorphism is about behavior and method dispatch, not database persistence."
                            if is_db_misconception else
                            "Review textbook definitions and fundamental operating principles for this topic."
                        ],
                        "suggested_ideal_outline": [
                            "Define the fundamental term and its primary purpose",
                            "Explain correct internal mechanisms and control flow",
                            "Contrast with common misconceptions"
                        ],
                        "next_difficulty_recommendation": "easier",
                        "rubric_breakdown": {
                            "technical_correctness": 0.0,
                            "conceptual_understanding": 0.0,
                            "completeness": 0.0,
                            "clarity": 0.0,
                            "relevance": 0.0
                        },
                        "should_follow_up": False,
                        "follow_up_reason": "misconception",
                        "follow_up_question": None,
                        "is_ai_estimate": True
                    }
                elif is_circular:
                    res_dict = {
                        "is_correct": False,
                        "numerical_score": 10.0,
                        "brief_explanation": "Circular or tautological answer. The response merely repeats terms from the question without explaining the underlying mechanism or concept.",
                        "correct_concepts_identified": [],
                        "missing_concepts": [
                            "Clear technical definition explaining the operational purpose and mechanism",
                            "Internal component interactions and practical engineering principles"
                        ],
                        "technical_inaccuracies": [
                            "Tautological rephrasing that restates the prompt without adding technical substance."
                        ],
                        "specific_improvement_suggestions": [
                            "Define the concept by describing how it functions internally, why it is used, and what system problem it solves."
                        ],
                        "suggested_ideal_outline": [
                            "Define the core concept and its architectural role",
                            "Explain step-by-step mechanisms and state transitions",
                            "Give a practical scenario or trade-off"
                        ],
                        "next_difficulty_recommendation": "easier",
                        "rubric_breakdown": {
                            "technical_correctness": 2.0,
                            "conceptual_understanding": 2.0,
                            "completeness": 2.0,
                            "clarity": 2.0,
                            "relevance": 2.0
                        },
                        "should_follow_up": False,
                        "follow_up_reason": "shallow",
                        "follow_up_question": None,
                        "is_ai_estimate": True
                    }
                elif is_partial:
                    partial_fb = (
                        "Partially correct answer. You noted that polymorphism means having many forms, but omitted Java implementation mechanisms such as method overriding, method overloading, dynamic method dispatch, and interface implementations."
                        if is_poly_partial else
                        "Partially correct answer. The candidate demonstrates basic awareness of key concepts, but omits core operational mechanics and edge case handling."
                    )
                    partial_concepts = [
                        "Understands the high-level concept of polymorphism ('many forms')"
                    ] if is_poly_partial else [
                        "Recognized key concept terminology and primary mapping relationship"
                    ]
                    partial_missing = [
                        "Method overriding (runtime polymorphism)",
                        "Method overloading (compile-time polymorphism)",
                        "Dynamic method dispatch via JVM vtables",
                        "Inheritance and interface implementations in Java"
                    ] if is_poly_partial else [
                        "Detailed mechanical workflow, exception handling traps, and caching layer effects"
                    ]
                    res_dict = {
                        "is_correct": True,
                        "numerical_score": 45.0 if is_poly_partial else 52.0,
                        "brief_explanation": partial_fb,
                        "correct_concepts_identified": partial_concepts,
                        "missing_concepts": partial_missing,
                        "technical_inaccuracies": [],
                        "specific_improvement_suggestions": [
                            "Explain how Java implements polymorphism via method overriding, method overloading, and JVM dynamic dispatch."
                            if is_poly_partial else
                            "Detail the hardware-software interaction and explain what happens when cache misses or faults occur."
                        ],
                        "suggested_ideal_outline": [
                            "Define core terminology and architectural purpose",
                            "Explain step-by-step translation or execution flow",
                            "Discuss memory overhead and real-world system trade-offs"
                        ],
                        "next_difficulty_recommendation": "same",
                        "rubric_breakdown": {
                            "technical_correctness": 14.0 if is_poly_partial else 16.0,
                            "conceptual_understanding": 12.0 if is_poly_partial else 13.0,
                            "completeness": 9.0 if is_poly_partial else 10.0,
                            "clarity": 6.0 if is_poly_partial else 7.0,
                            "relevance": 4.0 if is_poly_partial else 6.0
                        },
                        "should_follow_up": True,
                        "follow_up_reason": "shallow",
                        "follow_up_question": {
                            "question_text": "Could you elaborate on how Java resolves which method implementation to execute at runtime?",
                            "question_type": "follow_up",
                            "difficulty": "medium",
                            "topic": "Polymorphism"
                        } if is_poly_partial else {
                            "question_text": "Could you elaborate on what occurs when the required page is not present in physical memory?",
                            "question_type": "follow_up",
                            "difficulty": "medium",
                            "topic": "Virtual Memory"
                        },
                        "is_ai_estimate": True
                    }
                elif is_strong:
                    # Provide rich, authentic topic-specific evaluation
                    if is_poly_strong:
                        strong_fb = "High quality response. Accurately articulates runtime polymorphism in Java where a single method invocation executes different logic depending on the concrete runtime object."
                        strong_concepts = [
                            "Dynamic method dispatch in Java",
                            "Method call behavioral variation based on runtime object type",
                            "Runtime polymorphism core principle"
                        ]
                        strong_missing = [
                            "Compile-time polymorphism via method overloading",
                            "JVM vtable implementation details"
                        ]
                        follow_up_q = {
                            "question_text": "How does the JVM internally implement dynamic method dispatch using virtual method tables (vtables)?",
                            "question_type": "follow_up",
                            "difficulty": "hard",
                            "topic": "Java"
                        }
                        ideal_outline = [
                            "Define polymorphism and distinguish runtime from compile-time polymorphism",
                            "Explain method overriding and JVM dynamic method dispatch",
                            "Provide a clear class hierarchy example with overridden methods"
                        ]
                        improvement_sugg = ["Consider discussing how JVM vtables enable dynamic method dispatch."]
                    elif is_concurrency_strong:
                        strong_fb = "High quality response. Accurately articulates concurrency synchronization, critical section protection, and prevention of race conditions."
                        strong_concepts = [
                            "Mutual exclusion locking semantics",
                            "Prevention of race conditions on shared resources",
                            "Thread synchronization and critical section isolation"
                        ]
                        strong_missing = [
                            "Lock-free atomic Compare-And-Swap (CAS) alternatives",
                            "Deadlock prevention conditions (e.g. Coffman conditions)"
                        ]
                        follow_up_q = {
                            "question_text": "How does a spinlock compare to a mutex in terms of CPU utilization and context-switch overhead?",
                            "question_type": "follow_up",
                            "difficulty": "hard",
                            "topic": "Concurrency"
                        }
                        ideal_outline = [
                            "Define synchronization primitive and its mutual exclusion guarantee",
                            "Explain critical section isolation and thread scheduling transitions",
                            "Contrast lock-based synchronization with lock-free atomic alternatives"
                        ]
                        improvement_sugg = ["Consider discussing lock contention and spinlock vs mutex trade-offs."]
                    elif is_db_strong:
                        strong_fb = "High quality response. Accurately articulates database transaction guarantees, relational schema consistency, and persistent storage principles."
                        strong_concepts = [
                            "ACID transaction properties and atomic state transitions",
                            "Database consistency invariants and isolation guarantees",
                            "Relational data persistence and integrity constraints"
                        ]
                        strong_missing = [
                            "Distributed two-phase commit (2PC) or Write-Ahead Logging (WAL) internals",
                            "Isolation anomalies like dirty reads, non-repeatable reads, and phantom reads"
                        ]
                        follow_up_q = {
                            "question_text": "How does Write-Ahead Logging (WAL) guarantee durability and crash recovery in transactional database engines?",
                            "question_type": "follow_up",
                            "difficulty": "hard",
                            "topic": "Databases"
                        }
                        ideal_outline = [
                            "Define transaction semantics and the four ACID guarantees",
                            "Explain concurrency isolation levels and locking mechanisms",
                            "Discuss write-ahead logging (WAL) and crash-recovery protocols"
                        ]
                        improvement_sugg = ["Consider discussing transaction isolation levels and how MVCC avoids read-write locks."]
                    elif is_ds_strong:
                        strong_fb = "High quality response. Accurately articulates data structure ordering invariants, traversal algorithms, and computational complexity."
                        strong_concepts = [
                            "Binary search tree left-smaller right-greater invariant",
                            "Logarithmic average-case search and insertion time complexity",
                            "Hierarchical node linkage and recursive traversal patterns"
                        ]
                        strong_missing = [
                            "Worst-case O(n) degeneration on pre-sorted data sequences",
                            "Self-balancing tree balancing mechanisms (AVL or Red-Black rotations)"
                        ]
                        follow_up_q = {
                            "question_text": "What worst-case insertion sequence causes a standard BST to degrade into an O(n) linked list, and how do self-balancing trees prevent this?",
                            "question_type": "follow_up",
                            "difficulty": "hard",
                            "topic": "Data Structures"
                        }
                        ideal_outline = [
                            "Define the tree ordering property and pointer structure",
                            "Detail search, insert, and delete workflows with time complexities",
                            "Contrast unbalanced worst-case with self-balancing tree invariants"
                        ]
                        improvement_sugg = ["Consider discussing tree balancing rotations in AVL or Red-Black trees to preserve O(log n) guarantees."]
                    elif is_net_strong:
                        strong_fb = "High quality response. Accurately articulates transport-layer communication protocols, connection establishment, and transmission reliability."
                        strong_concepts = [
                            "Connection-oriented byte-stream vs connectionless datagram mechanics",
                            "Reliable ordered delivery via sequence numbers and acknowledgments",
                            "Transport-layer delivery guarantees and latency/throughput trade-offs"
                        ]
                        strong_missing = [
                            "TCP congestion control algorithms (Slow Start, Congestion Avoidance, Fast Retransmit)",
                            "Head-of-Line (HoL) blocking and TIME_WAIT state lifecycle"
                        ]
                        follow_up_q = {
                            "question_text": "Why does TCP implement the TIME_WAIT state after sending the final ACK during connection teardown?",
                            "question_type": "follow_up",
                            "difficulty": "hard",
                            "topic": "Computer Networks"
                        }
                        ideal_outline = [
                            "Compare connection establishment and state tracking",
                            "Explain packet ordering, error detection, and retransmission mechanisms",
                            "Analyze application use cases (e.g. HTTP/web vs streaming/gaming)"
                        ]
                        improvement_sugg = ["Consider discussing TCP congestion window adaptation and flow control window advertisements."]
                    elif is_oop_strong:
                        strong_fb = "High quality response. Accurately articulates core Object-Oriented Programming principles, component encapsulation, and code reusability."
                        strong_concepts = [
                            "Information hiding and state encapsulation via access modifiers",
                            "Code reusability through inheritance hierarchies",
                            "Polymorphic interface contracts decoupling callers from implementations"
                        ]
                        strong_missing = [
                            "Liskov Substitution Principle (LSP) adherence",
                            "Favoring object composition over class inheritance"
                        ]
                        follow_up_q = {
                            "question_text": "Why do modern software engineering design principles strongly favor object composition over class inheritance?",
                            "question_type": "follow_up",
                            "difficulty": "hard",
                            "topic": "OOP"
                        }
                        ideal_outline = [
                            "Define core OOP pillars (Encapsulation, Inheritance, Polymorphism, Abstraction)",
                            "Explain interface design and dynamic binding mechanics",
                            "Discuss real-world software design trade-offs and best practices"
                        ]
                        improvement_sugg = ["Consider discussing composition over inheritance and the Liskov Substitution Principle."]
                    else:
                        strong_fb = "Exceptional, comprehensive technical response demonstrating clear conceptual depth and rigorous system mechanics."
                        strong_concepts = [
                            "Clear separation of virtual address space and physical frames",
                            "Hardware page table walk and TLB cache hit/miss behavior",
                            "Page fault interrupt trap and OS disk fetch sequence"
                        ]
                        strong_missing = [
                            "TLB shootdown overhead in multi-core symmetric multiprocessing"
                        ]
                        follow_up_q = {
                            "question_text": "How does the hardware Translation Lookaside Buffer (TLB) specifically prevent memory access latency from doubling?",
                            "question_type": "follow_up",
                            "difficulty": "medium",
                            "topic": "Virtual Memory"
                        }
                        ideal_outline = [
                            "Architectural purpose: Memory protection and contiguous addressing abstraction",
                            "Hardware translation workflow: MMU, CR3 register, TLB lookup",
                            "Page fault lifecycle: Trap to kernel, disk fetch to page frame, PTE update, restart instruction"
                        ]
                        improvement_sugg = ["Consider discussing multi-level paging memory overhead versus inverted page tables."]

                    res_dict = {
                        "is_correct": True,
                        "numerical_score": 88.0,
                        "brief_explanation": strong_fb,
                        "correct_concepts_identified": strong_concepts,
                        "missing_concepts": strong_missing,
                        "technical_inaccuracies": [],
                        "specific_improvement_suggestions": improvement_sugg,
                        "suggested_ideal_outline": ideal_outline,
                        "next_difficulty_recommendation": "harder",
                        "rubric_breakdown": {
                            "technical_correctness": 27.0,
                            "conceptual_understanding": 23.0,
                            "completeness": 18.0,
                            "clarity": 12.0,
                            "relevance": 8.0
                        },
                        "should_follow_up": False,
                        "follow_up_reason": "none",
                        "follow_up_question": None,
                        "is_ai_estimate": True
                    }
                else:
                    res_dict = {
                        "is_correct": True,
                        "numerical_score": 82.0,
                        "brief_explanation": f"Solid technical answer with accurate conceptual understanding of {topic_text or 'the subject'}.",
                        "correct_concepts_identified": [
                            "Accurately addressed primary question requirements",
                            "Correct core technical terminology and definitions"
                        ],
                        "missing_concepts": [
                            "In-depth mechanical execution steps and practical system trade-offs"
                        ],
                        "technical_inaccuracies": [],
                        "specific_improvement_suggestions": [
                            "Elaborate on low-level operational workflows and real-world system constraints."
                        ],
                        "suggested_ideal_outline": [
                            "Define primary terminology and purpose",
                            "Explain operational workflow and component interaction",
                            "Discuss practical edge cases and performance trade-offs"
                        ],
                        "next_difficulty_recommendation": "same",
                        "rubric_breakdown": {
                            "technical_correctness": 25.0,
                            "conceptual_understanding": 21.0,
                            "completeness": 16.0,
                            "clarity": 12.0,
                            "relevance": 8.0
                        },
                        "should_follow_up": False,
                        "follow_up_reason": "none",
                        "follow_up_question": None,
                        "is_ai_estimate": True
                    }

                # Prepare context-aware model answer and explanations
                if any(k in combined_context for k in ["page", "paging", "virtual memory", "mmu", "tlb"]):
                    model_correct_answer = (
                        "Virtual memory is a memory management architecture that decouples a program's logical address space from physical RAM. "
                        "The CPU Memory Management Unit (MMU) uses page tables and a Translation Lookaside Buffer (TLB) to map virtual page numbers "
                        "to physical frame numbers. When a program accesses an address whose page table entry indicates the page is not in RAM (valid bit = 0), "
                        "the hardware triggers a page fault exception. The operating system kernel traps the fault, retrieves the missing page from disk swap "
                        "storage into an available RAM frame, updates the page table entry, and transparently restarts the faulting instruction."
                    )
                    model_simple_explanation = (
                        "Think of virtual memory like an index in a library. Instead of keeping every single book on your small study desk (RAM), "
                        "you keep a catalog (page table). You only fetch the exact books you need right now. When your desk gets too full, "
                        "the system swaps inactive books back to storage shelves (disk) so active work never crashes from running out of space."
                    )
                    model_detailed_explanation = (
                        "1. Hardware Address Translation: Logical addresses are divided into a Virtual Page Number (VPN) and an offset. The MMU consults the "
                        "TLB cache. On a TLB hit, the Physical Frame Number (PFN) is concatenated with the offset in a single clock cycle. On a miss, "
                        "the MMU performs a multi-level page table walk (using CR3 register in x86-64).\n"
                        "2. Page Fault Interrupt Traps: If the PTE valid bit is 0, interrupt vector 14 traps execution to the kernel page fault handler. "
                        "The OS selects a victim frame using replacement policies (e.g. LRU or CLOCK), executes disk swap I/O, updates PTE permissions, "
                        "and invalidates stale TLB entries across cores via TLB shootdowns.\n"
                        "3. Protection & Isolation: Each process operates in its own virtual address space, preventing unauthorized cross-process memory access."
                    )
                    model_example = (
                        "In Linux, processes allocate virtual memory via the `mmap()` system call or `brk()`. When `malloc(1024 * 1024)` is called, "
                        "the kernel reserves virtual address space immediately, but zero physical RAM frames are committed until the program writes to a byte, "
                        "at which point a demand-paging minor fault allocates a 4KB physical frame on the fly."
                    )
                    model_key_points = [
                        "Virtual memory isolates processes into independent address spaces and protects kernel memory from corruption.",
                        "The Memory Management Unit (MMU) translates logical addresses to physical frame addresses using page tables and TLB caches.",
                        "A page fault is an architectural trap raised by the MMU when a virtual page is not currently resident in physical RAM.",
                        "Standard page size is 4KB, balancing internal fragmentation against page table entry overhead."
                    ]
                elif is_db_context:
                    model_correct_answer = (
                        "ACID in database systems defines the four fundamental transaction properties guaranteeing data validity despite errors, power failures, or concurrent crashes: "
                        "1) Atomicity (all operations execute completely or rollback entirely); "
                        "2) Consistency (transactions transform the database from one valid state to another, upholding all constraints and foreign keys); "
                        "3) Isolation (concurrent transactions execute independently without reading uncommitted dirty writes); "
                        "and 4) Durability (once committed, updates persist permanently in non-volatile storage even after a system crash)."
                    )
                    model_simple_explanation = (
                        "Think of an ATM bank transfer: if money is deducted from your checking account, it MUST arrive in your savings account. "
                        "If the machine loses power halfway through, the whole transfer is reversed (Atomicity), neither money vanishes nor duplicates (Consistency), "
                        "simultaneous withdrawals don't interfere (Isolation), and once completed, the receipt is permanent (Durability)."
                    )
                    model_detailed_explanation = (
                        "1. Atomicity & Durability via Write-Ahead Logging (WAL): Changes are recorded in an append-only transaction log on disk before updating data pages (ARIES recovery protocol).\n"
                        "2. Isolation via Concurrency Control: Implemented via Strict Two-Phase Locking (S2PL) or Multi-Version Concurrency Control (MVCC) with snapshot isolation.\n"
                        "3. Consistency Enforcement: Maintained by schema constraints (NOT NULL, UNIQUE, FOREIGN KEY, CHECK) and transaction commit validation."
                    )
                    model_example = (
                        "```sql\n"
                        "BEGIN TRANSACTION;\n"
                        "  UPDATE accounts SET balance = balance - 500 WHERE id = 1;\n"
                        "  UPDATE accounts SET balance = balance + 500 WHERE id = 2;\n"
                        "COMMIT; -- Guarantees atomic persistence across both accounts\n"
                        "```"
                    )
                    model_key_points = [
                        "Atomicity guarantees 'all-or-nothing' transaction execution.",
                        "Consistency preserves all relational integrity constraints and business invariants.",
                        "Isolation prevents anomalies (dirty reads, non-repeatable reads, phantom reads).",
                        "Durability ensures committed state survives crashes via Write-Ahead Logging (WAL)."
                    ]
                elif is_ds_context:
                    model_correct_answer = (
                        "A Binary Search Tree (BST) is a hierarchical node-based data structure where each node contains a key and at most two children. "
                        "It maintains the binary search invariant: all keys in the left subtree are strictly less than the node's key, and all keys in the right subtree are strictly greater. "
                        "Average time complexity for search, insertion, and deletion is O(log n), but can degrade to O(n) if inserted in sorted order without self-balancing."
                    )
                    model_simple_explanation = (
                        "Think of a BST like searching a printed dictionary: opening at the middle tells you whether your target word lies in the left half or the right half, "
                        "eliminating half of the remaining words at each step."
                    )
                    model_detailed_explanation = (
                        "1. Search Invariant: Starting at the root, if target < node.val go left, if target > node.val go right, else found.\n"
                        "2. In-Order Traversal: Visiting left, root, right produces keys in monotonically ascending sorted order in O(n) time.\n"
                        "3. Self-Balancing Augmentation: AVL and Red-Black trees perform tree rotations during mutations to maintain balanced height O(log n)."
                    )
                    model_example = (
                        "```python\n"
                        "class TreeNode:\n"
                        "    def __init__(self, val=0, left=None, right=None):\n"
                        "        self.val = val\n"
                        "        self.left = left\n"
                        "        self.right = right\n\n"
                        "def search_bst(root, target):\n"
                        "    if not root or root.val == target:\n"
                        "        return root\n"
                        "    return search_bst(root.left, target) if target < root.val else search_bst(root.right, target)\n"
                        "```"
                    )
                    model_key_points = [
                        "BST ordering invariant: Left subtree < Parent < Right subtree.",
                        "In-order traversal always outputs keys in strictly sorted order.",
                        "Average time complexity is O(log n); worst-case is O(n) for degenerate skewed trees.",
                        "Self-balancing trees (AVL / Red-Black) guarantee worst-case O(log n) performance."
                    ]
                elif is_net_context:
                    model_correct_answer = (
                        "TCP (Transmission Control Protocol) is a connection-oriented, reliable transport protocol that provides ordered, error-checked delivery of a stream of octets via a 3-way handshake (SYN, SYN-ACK, ACK), flow control (sliding window), and congestion control. "
                        "In contrast, UDP (User Datagram Protocol) is a connectionless, lightweight, unreliable datagram protocol without connection state, retransmission, or flow control, prioritizing minimal latency for real-time applications."
                    )
                    model_simple_explanation = (
                        "Think of TCP like registered mail with delivery confirmation and signature receipts. UDP is like a public loudspeaker announcement: fast and broadcast immediately, but if someone is momentarily distracted, the missed word is not repeated."
                    )
                    model_detailed_explanation = (
                        "1. Connection Lifecycle: TCP establishes connections using a 3-way handshake and tears down via a 4-way FIN-ACK exchange; UDP has zero handshake overhead.\n"
                        "2. Reliability & Ordering: TCP attaches 32-bit sequence and acknowledgment numbers to reorder out-of-order packets and retransmit lost segments.\n"
                        "3. Flow & Congestion Control: TCP adapts throughput to network capacity via sliding windows and congestion avoidance algorithms."
                    )
                    model_example = (
                        "```python\n"
                        "import socket\n\n"
                        "# TCP: Reliable stream socket\n"
                        "tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n\n"
                        "# UDP: Low-latency datagram socket\n"
                        "udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
                        "```"
                    )
                    model_key_points = [
                        "TCP is connection-oriented with guaranteed in-order delivery; UDP is connectionless with best-effort delivery.",
                        "TCP uses a 3-way handshake (SYN -> SYN-ACK -> ACK) and sequence numbers.",
                        "TCP includes flow control (sliding window) and congestion control algorithms.",
                        "UDP header is minimal (8 bytes vs 20+ bytes TCP), reducing transmission latency for real-time gaming and VoIP."
                    ]
                elif any(k in combined_context for k in ["mutex", "lock", "concurrency", "thread", "semaphore", "deadlock"]):
                    model_correct_answer = (
                        "A mutex (mutual exclusion lock) is a synchronization primitive used in concurrent programming to guarantee that only one thread "
                        "can execute a critical section at any given time, preventing race conditions and shared memory corruption."
                    )
                    model_simple_explanation = (
                        "Think of a mutex like the single key to a locked meeting room. Only the person holding the key can enter and work. "
                        "Anyone else who wants to use the room must wait outside until the key is handed back."
                    )
                    model_detailed_explanation = (
                        "1. Atomic Primitives: Mutexes rely on hardware atomic instructions such as Compare-And-Swap (CAS) or Test-and-Set to ensure race-free state updates.\n"
                        "2. OS Scheduler Integration: When contention occurs, waiting threads are placed into a kernel wait queue, avoiding CPU-wasting busy-waits (spinlocks).\n"
                        "3. Strict Ownership: Only the specific thread that acquired the mutex has permission to release it, distinguishing it from counting semaphores."
                    )
                    model_example = (
                        "```python\n"
                        "import threading\n\n"
                        "lock = threading.Lock()\n"
                        "shared_balance = 100\n\n"
                        "def withdraw(amount):\n"
                        "    global shared_balance\n"
                        "    with lock:  # Critical section protected against concurrent race conditions\n"
                        "        if shared_balance >= amount:\n"
                        "            shared_balance -= amount\n"
                        "```"
                    )
                    model_key_points = [
                        "A mutex guarantees mutual exclusion by allowing only one thread inside a critical section at a time.",
                        "Possesses strict ownership semantics—only the thread that locked the mutex is allowed to unlock it.",
                        "Implemented using atomic CPU primitives (CAS) combined with OS scheduler wait queues.",
                        "To prevent deadlocks, mutexes across multiple resources must always be acquired in a globally consistent order."
                    ]
                elif is_oop_context:
                    model_correct_answer = (
                        "Object-Oriented Programming (OOP) is an architectural programming paradigm organized around objects that encapsulate both state (attributes) and behavior (methods). "
                        "The four fundamental pillars of OOP are: 1) Encapsulation (bundling data with operations and restricting direct access to internal representation); "
                        "2) Abstraction (hiding complex internal implementation details behind clean, minimal public interfaces); "
                        "3) Inheritance (enabling subclasses to inherit and extend attributes and behaviors from superclasses for code reuse); "
                        "and 4) Polymorphism (allowing diverse types to be treated uniformly through shared interfaces, dynamically dispatched at runtime)."
                    )
                    model_simple_explanation = (
                        "Think of driving an automobile: the dashboard and pedals provide a clean interface (Abstraction) while complex engine components are contained under the hood (Encapsulation). "
                        "Both sedans and trucks inherit common vehicle mechanics (Inheritance), and pressing the accelerator accelerates either vehicle regardless of engine type (Polymorphism)."
                    )
                    model_detailed_explanation = (
                        "1. Dynamic Method Dispatch: At runtime, overridden methods invoked through interface or superclass references execute the concrete subclass implementation.\n"
                        "2. Information Hiding & Invariant Protection: Access modifiers (private, protected, public) prevent outside actors from corrupting internal class invariants.\n"
                        "3. Architectural Design Patterns: Adheres to SOLID engineering principles, emphasizing loose coupling and high cohesion."
                    )
                    model_example = (
                        "```python\n"
                        "class Vehicle:\n"
                        "    def __init__(self, brand):\n"
                        "        self._brand = brand  # Encapsulation\n\n"
                        "    def drive(self):\n"
                        "        return f'{self._brand} is driving'\n\n"
                        "class ElectricCar(Vehicle):  # Inheritance\n"
                        "    def drive(self):         # Polymorphism\n"
                        "        return f'{self._brand} drives silently on electric power'\n"
                        "```"
                    )
                    model_key_points = [
                        "Encapsulation bundles data with operations and protects internal state invariants.",
                        "Abstraction presents clean public contracts while hiding internal implementation mechanics.",
                        "Inheritance enables code reuse and hierarchical domain modeling.",
                        "Polymorphism allows dynamic method dispatch across diverse concrete classes."
                    ]
                else:
                    topic_name = topic_text or "Computer Science"
                    ref_sol = reference_answers.get_authoritative_reference_solution(
                        question_text=q_text,
                        topic=topic_name,
                        subject="Computer Science"
                    )
                    model_correct_answer = ref_sol["correct_answer"]
                    model_simple_explanation = ref_sol["simple_explanation"]
                    model_detailed_explanation = ref_sol["detailed_explanation"]
                    model_example = ref_sol["example"]
                    model_key_points = ref_sol["key_points_to_remember"]

                # Inject model answer and standardized evaluation fields into res_dict
                ref_sol_general = reference_answers.get_authoritative_reference_solution(
                    question_text=q_text,
                    topic=topic_text,
                    subject="Computer Science"
                )
                res_dict["correct_answer"] = model_correct_answer
                res_dict["simple_explanation"] = model_simple_explanation
                res_dict["detailed_explanation"] = model_detailed_explanation
                res_dict["example"] = model_example
                res_dict["key_points_to_remember"] = model_key_points
                res_dict["improved_answer"] = ref_sol_general.get("improved_answer")
                res_dict["correctness_status"] = "empty" if is_empty_or_idk else ("incorrect" if (is_unrelated or is_explicitly_wrong or is_circular) else ("partially_correct" if is_partial else "correct"))
                res_dict["score"] = res_dict["numerical_score"]
                res_dict["evaluation_feedback"] = res_dict["brief_explanation"]
                res_dict["correct_concepts"] = res_dict.get("correct_concepts_identified", [])
                res_dict["incorrect_concepts"] = res_dict.get("technical_inaccuracies", [])

                return MockMessageResponse(json.dumps(res_dict))

            elif "Evaluate the candidate's answer" in prompt or "Candidate's Answer:" in prompt:
                # Legacy 10-point Evaluation request
                answer = ""
                if "<candidate_answer>" in prompt:
                    try:
                        answer = prompt.split("<candidate_answer>")[1].split("</candidate_answer>")[0]
                    except Exception:
                        answer = ""
                elif 'Candidate\'s Answer: """' in prompt:
                    try:
                        answer = prompt.split('Candidate\'s Answer: """')[1].split('"""')[0]
                    except Exception:
                        answer = ""
                
                clean_leg = answer.strip().lower()
                is_leg_wrong = any(w in clean_leg for w in ["wrong", "incorrect", "false", "potato", "garbage", "nonsense", "not true", "fake"])
                is_weak = len(clean_leg) < 25 or "don't know" in clean_leg
                is_strong = len(clean_leg) > 80 and any(w in clean_leg for w in ["table", "tlb", "frame", "virtual", "memory", "fault"])
                
                if is_leg_wrong or "don't know" in clean_leg:
                    res_dict = {
                        "score": 0.0,
                        "feedback": "Incorrect answer. The explanation contains fundamental inaccuracies.",
                        "strengths": None,
                        "improvement_areas": "Review core concepts and accurate terminology.",
                        "should_follow_up": False,
                        "follow_up_reason": "misconception",
                        "follow_up_question": None
                    }
                elif is_weak:
                    res_dict = {
                        "score": 4.5,
                        "feedback": "Your response is very brief and misses key technical mechanisms and architectural trade-offs.",
                        "strengths": "Addressed the general topic area.",
                        "improvement_areas": "Elaborate on data structures, hardware translation steps, and page fault traps.",
                        "should_follow_up": False,
                        "follow_up_reason": "shallow",
                        "follow_up_question": None
                    }
                elif is_strong:
                    res_dict = {
                        "score": 8.5,
                        "feedback": "Strong, technically precise explanation of address translation and hardware paging structures.",
                        "strengths": "Accurately distinguished virtual pages from physical frames and explained page fault handling.",
                        "improvement_areas": "Could further detail TLB shootdowns in multi-core symmetric multiprocessing (SMP).",
                        "should_follow_up": True,
                        "follow_up_reason": "advanced_probe",
                        "follow_up_question": {
                            "question_text": "How does the hardware Translation Lookaside Buffer (TLB) specifically prevent memory access latency from doubling?",
                            "question_type": "follow_up",
                            "difficulty": "medium",
                            "topic": "Virtual Memory"
                        }
                    }
                else:
                    res_dict = {
                        "score": 7.0,
                        "feedback": "Good fundamental explanation of the concept with acceptable technical reasoning.",
                        "strengths": "Correctly identified the primary memory mapping role.",
                        "improvement_areas": "Consider edge cases like page fault traps and disk swapping overhead.",
                        "should_follow_up": True,
                        "follow_up_reason": "shallow",
                        "follow_up_question": {
                            "question_text": "What exact sequence of operations does the OS perform when a page fault exception is raised by the MMU?",
                            "question_type": "follow_up",
                            "difficulty": "medium",
                            "topic": "Virtual Memory"
                        }
                    }
                return MockMessageResponse(json.dumps(res_dict))
            else:
                # Question generation request
                topic = "Virtual Memory & Paging"
                subject = "Computer Science"
                if "Subject / Domain:" in prompt:
                    try:
                        subject = prompt.split("Subject / Domain:")[1].split("\n")[0].strip()
                    except Exception:
                        subject = "Computer Science"
                if "Specific Topic:" in prompt:
                    try:
                        topic = prompt.split("Specific Topic:")[1].split("\n")[0].strip()
                    except Exception:
                        topic = "Virtual Memory"

                is_os = any(s in (subject + " " + topic).lower() for s in ["operating system", "os", "linux", "paging", "memory management", "deadlock", "process"])
                first_q_text = (
                    f"Explain the fundamental mechanism of {topic} and how the operating system manages it."
                    if is_os else
                    f"Explain the fundamental principles and operational mechanisms of {topic} in {subject}."
                )

                mock_questions = [
                    {
                        "question_text": first_q_text,
                        "question_type": "conceptual",
                        "difficulty": "medium",
                        "topic": topic,
                        "evaluation_criteria": [
                            "Defines primary concept accurately",
                            "Explains interaction between components or systems",
                            "Discusses performance trade-offs"
                        ],
                        "question_order": 1
                    },
                    {
                        "question_text": f"What critical failure modes or performance bottlenecks can occur when implementing {topic}?",
                        "question_type": "edge_case",
                        "difficulty": "hard",
                        "topic": topic,
                        "evaluation_criteria": [
                            "Identifies specific overhead or thrashing scenario",
                            "Explains mitigation strategies",
                            "Demonstrates systems-level reasoning"
                        ],
                        "question_order": 2
                    },
                    {
                        "question_text": f"Compare and contrast the primary algorithmic trade-offs used to optimize {topic}.",
                        "question_type": "trade_off",
                        "difficulty": "medium",
                        "topic": topic,
                        "evaluation_criteria": [
                            "Compares at least two contrasting approaches",
                            "Analyzes time vs space complexity",
                            "Mentions real-world operating system implementations"
                        ],
                        "question_order": 3
                    },
                    {
                        "question_text": f"In a high-concurrency production environment, how does the system maintain consistency in {topic}?",
                        "question_type": "mechanism",
                        "difficulty": "hard",
                        "topic": topic,
                        "evaluation_criteria": [
                            "Explains synchronization and locking primitives",
                            "Addresses race conditions and atomic operations"
                        ],
                        "question_order": 4
                    }
                ]

                num_q = 2
                if "Generate exactly " in prompt:
                    try:
                        num_q = int(prompt.split("Generate exactly ")[1].split(" ")[0])
                    except Exception:
                        num_q = 2

                selected = mock_questions[:max(1, min(num_q, len(mock_questions)))]
                for i, q in enumerate(selected, 1):
                    q["question_order"] = i

                return MockMessageResponse(json.dumps(selected))


def get_anthropic_client(custom_client: Optional[Any] = None) -> anthropic.Anthropic:
    """
    Initializes and returns an Anthropic client using configured settings.
    If custom_client is provided, returns that (used for testing/mocking).
    If ANTHROPIC_API_KEY is set to 'mock' or 'demo', returns offline developer client.
    """
    if custom_client is not None:
        return custom_client

    settings = get_settings()
    api_key = settings.ANTHROPIC_API_KEY

    if not api_key or not api_key.strip():
        logger.error("Attempted to call AI service without ANTHROPIC_API_KEY configured.")
        raise AIConfigurationError(
            "Anthropic API key is not configured. Please set ANTHROPIC_API_KEY in your environment or .env file (or set to 'mock' for local offline testing)."
        )

    clean_key = api_key.strip().lower()
    if clean_key in ("mock", "demo", "mock-developer-key", "test-mock"):
        logger.info("Using simulated offline AI examiner (ANTHROPIC_API_KEY=mock)")
        return MockAnthropicClient()

    return anthropic.Anthropic(api_key=api_key.strip(), timeout=45.0)


def extract_json_payload(raw_text: str) -> Any:
    """
    Extracts and parses JSON from raw LLM output, stripping markdown code blocks if present.
    Safely handles nested markdown fences inside JSON strings.
    """
    cleaned = raw_text.strip()

    # 1. Direct parse attempt (fastest and handles JSON with inner code fences)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Outer markdown code fence stripping: ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", cleaned)
        if match:
            inner = match.group(1).strip()
            try:
                return json.loads(inner)
            except Exception:
                cleaned = inner

    # 3. Extract outermost JSON object {...}
    start_brace = cleaned.find('{')
    end_brace = cleaned.rfind('}')
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        try:
            return json.loads(cleaned[start_brace:end_brace + 1])
        except Exception:
            pass

    # 4. Extract outermost JSON array [...]
    start_bracket = cleaned.find('[')
    end_bracket = cleaned.rfind(']')
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        try:
            return json.loads(cleaned[start_bracket:end_bracket + 1])
        except Exception:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"JSON decode failure: {exc}. Raw text: {raw_text[:200]}")
        raise AIResponseParseError(f"AI response did not contain valid JSON: {exc.msg}")


def generate_interview_questions(
    request: GenerateQuestionsRequest,
    custom_client: Optional[Any] = None
) -> List[GeneratedQuestion]:
    """
    Generates structured viva/interview questions using Anthropic Claude.

    Args:
        request: Interview setup parameters (subject, topic, level, count, difficulty, etc.)
        custom_client: Optional injected Anthropic client (for testing/mocking)

    Returns:
        List of validated GeneratedQuestion objects

    Raises:
        AIConfigurationError: Missing or empty API key (503)
        AIRateLimitError: API rate limit hit (429)
        AIResponseParseError: Malformed JSON or schema validation failure (502)
        AIServiceError: General network/API error (502)
    """
    client = get_anthropic_client(custom_client)
    settings = get_settings()
    model_name = settings.ANTHROPIC_MODEL

    user_prompt = build_question_generation_prompt(
        subject=request.subject,
        topic=request.topic,
        target_level=request.target_level,
        interview_type=request.interview_type,
        num_questions=request.num_questions,
        difficulty=request.difficulty,
        session_duration=request.session_duration,
        custom_instructions=request.custom_instructions
    )

    logger.info(
        f"Requesting {request.num_questions} questions from Claude ({model_name}) "
        f"for topic '{request.topic}' ({request.target_level})"
    )

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=2048,
            temperature=0.3,
            system=EXAMINER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
    except anthropic.AuthenticationError as exc:
        logger.error(f"Anthropic authentication error: {exc}")
        raise AIServiceError("Invalid Anthropic API key credentials.", status_code=502)
    except anthropic.RateLimitError as exc:
        logger.warning(f"Anthropic rate limit exceeded: {exc}")
        raise AIRateLimitError()
    except anthropic.APITimeoutError as exc:
        logger.error(f"Anthropic API request timed out: {exc}")
        raise AIServiceError("AI service request timed out. Please try again.", status_code=504)
    except anthropic.APIConnectionError as exc:
        logger.error(f"Could not connect to Anthropic API: {exc}")
        raise AIServiceError("Could not connect to Anthropic AI service.", status_code=502)
    except anthropic.APIStatusError as exc:
        logger.error(f"Anthropic API returned status {exc.status_code}: {exc.message}")
        raise AIServiceError(f"Anthropic API error: {exc.message}", status_code=502)
    except Exception as exc:
        logger.error(f"Unexpected error calling Claude API: {exc}")
        raise AIServiceError(f"Unexpected AI service error: {str(exc)}", status_code=502)

    # Extract text from response blocks
    content_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            content_text += block.text

    if not content_text.strip():
        raise AIResponseParseError("Claude returned an empty response.")

    # Parse and validate JSON structure
    raw_data = extract_json_payload(content_text)
    if not isinstance(raw_data, list):
        raise AIResponseParseError("AI response was expected to be a JSON array of questions.")

    validated_questions: List[GeneratedQuestion] = []
    for idx, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            raise AIResponseParseError(f"Question item {idx} is not a valid JSON object.")
        
        # Ensure question_order is present
        if "question_order" not in item:
            item["question_order"] = idx
        if "topic" not in item or not item["topic"]:
            item["topic"] = request.topic

        try:
            validated_q = GeneratedQuestion.model_validate(item)
            validated_questions.append(validated_q)
        except Exception as val_exc:
            logger.error(f"Validation failed for generated question item {idx}: {val_exc}")
            raise AIResponseParseError(
                f"Generated question {idx} failed schema validation: {str(val_exc)}"
            )

    logger.info(f"Successfully generated and validated {len(validated_questions)} questions.")
    return validated_questions


def evaluate_and_adapt_answer(
    question_text: str,
    topic: str,
    difficulty: str,
    target_level: str,
    interview_type: str,
    candidate_answer: str,
    current_follow_up_count: int = 0,
    max_follow_ups: int = 1,
    custom_client: Optional[Any] = None
) -> dict:
    """
    Evaluates candidate answer using Claude and determines if an adaptive follow-up is warranted.

    Returns dict with keys:
        - score: float (0.0 to 10.0)
        - feedback: str
        - strengths: Optional[str]
        - improvement_areas: Optional[str]
        - should_follow_up: bool
        - follow_up_reason: Optional[str]
        - follow_up_question: Optional[dict]
    """
    client = get_anthropic_client(custom_client)
    settings = get_settings()
    model_name = settings.ANTHROPIC_MODEL

    eval_prompt = build_evaluation_prompt(
        question_text=question_text,
        topic=topic,
        difficulty=difficulty,
        target_level=target_level,
        interview_type=interview_type,
        candidate_answer=candidate_answer,
        current_follow_up_count=current_follow_up_count,
        max_follow_ups=max_follow_ups
    )

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=1500,
            temperature=0.2,
            system=EXAMINER_EVALUATION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": eval_prompt}
            ]
        )
    except anthropic.AuthenticationError as exc:
        raise AIServiceError("Invalid Anthropic API key credentials.", status_code=502)
    except anthropic.RateLimitError:
        raise AIRateLimitError()
    except anthropic.APITimeoutError:
        raise AIServiceError("AI evaluation request timed out.", status_code=504)
    except anthropic.APIConnectionError:
        raise AIServiceError("Could not connect to Anthropic AI service.", status_code=502)
    except anthropic.APIStatusError as exc:
        raise AIServiceError(f"Anthropic API error: {exc.message}", status_code=502)
    except Exception as exc:
        raise AIServiceError(f"Unexpected AI error: {str(exc)}", status_code=502)

    content_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            content_text += block.text

    if not content_text.strip():
        raise AIResponseParseError("Claude returned empty response for evaluation.")

    eval_data = extract_json_payload(content_text)
    if not isinstance(eval_data, dict):
        raise AIResponseParseError("Expected evaluation response to be a JSON object.")

    # Validate and clamp score to 0.0 - 10.0
    try:
        score = float(eval_data.get("score", 5.0))
    except (ValueError, TypeError):
        score = 5.0
    score = max(0.0, min(10.0, round(score, 1)))

    feedback = str(eval_data.get("feedback", "Answer evaluated.")).strip()
    strengths = eval_data.get("strengths")
    improvement_areas = eval_data.get("improvement_areas")

    should_follow_up = bool(eval_data.get("should_follow_up", False))
    # Enforce strict maximum follow-up limit
    if current_follow_up_count >= max_follow_ups:
        should_follow_up = False

    follow_up_q = eval_data.get("follow_up_question")
    if not should_follow_up or not isinstance(follow_up_q, dict) or not follow_up_q.get("question_text"):
        should_follow_up = False
        follow_up_q = None
    else:
        # Normalize fields
        if "question_type" not in follow_up_q:
            follow_up_q["question_type"] = "follow_up"
        if "difficulty" not in follow_up_q:
            follow_up_q["difficulty"] = difficulty
        if "topic" not in follow_up_q:
            follow_up_q["topic"] = topic

    return {
        "score": score,
        "feedback": feedback,
        "strengths": str(strengths).strip() if strengths else None,
        "improvement_areas": str(improvement_areas).strip() if improvement_areas else None,
        "should_follow_up": should_follow_up,
        "follow_up_reason": eval_data.get("follow_up_reason"),
        "follow_up_question": follow_up_q
    }
