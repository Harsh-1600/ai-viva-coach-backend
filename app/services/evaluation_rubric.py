"""Configurable scoring rubric and weight definitions for AI answer evaluation."""

from typing import Dict, Any
from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    """Definition of an individual evaluation criterion."""
    name: str
    max_score: float
    description: str
    guidelines: str


class ScoringRubric(BaseModel):
    """Configurable scoring rubric summing to 100 points."""
    technical_correctness: RubricCriterion = Field(
        default=RubricCriterion(
            name="Technical Correctness",
            max_score=30.0,
            description="Factual accuracy of technical definitions, terms, algorithms, and system mechanics.",
            guidelines="Penalize factual inaccuracies. Accept alternative valid explanations. Do not penalize harmless wording differences."
        )
    )
    conceptual_understanding: RubricCriterion = Field(
        default=RubricCriterion(
            name="Conceptual Understanding",
            max_score=25.0,
            description="Depth of understanding of 'why' and 'how' mechanisms function rather than rote memorization.",
            guidelines="Do not reward keyword matching alone. Look for genuine reasoning and awareness of trade-offs."
        )
    )
    completeness: RubricCriterion = Field(
        default=RubricCriterion(
            name="Completeness & Depth",
            max_score=20.0,
            description="Coverage of essential components, edge cases, and architectural nuances required by the prompt.",
            guidelines="Assess whether core components were addressed or critical steps were omitted."
        )
    )
    clarity: RubricCriterion = Field(
        default=RubricCriterion(
            name="Clarity & Communication",
            max_score=15.0,
            description="Logical structuring, coherence, and professional technical articulation.",
            guidelines="Do not automatically reward long rambling answers. Reward crisp, precise explanations."
        )
    )
    relevance: RubricCriterion = Field(
        default=RubricCriterion(
            name="Relevance to Prompt",
            max_score=10.0,
            description="Direct alignment with the examiner's specific question.",
            guidelines="Penalize evasiveness or pivoting to unrelated concepts."
        )
    )

    def total_max_score(self) -> float:
        return (
            self.technical_correctness.max_score
            + self.conceptual_understanding.max_score
            + self.completeness.max_score
            + self.clarity.max_score
            + self.relevance.max_score
        )


DEFAULT_RUBRIC = ScoringRubric()
