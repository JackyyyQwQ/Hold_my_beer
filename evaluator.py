from __future__ import annotations

from pathlib import Path

from llm_client import LLMClient
from schema import EvaluatedFeedback, validate_evaluated_feedback


class FeedbackEvaluator:
    """Convert free-form user feedback into structured records plus scores."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path) -> None:
        self.llm_client = llm_client
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def evaluate_feedback(self, feedback_sentence: str) -> EvaluatedFeedback:
        """Run one evaluator prompt and validate the returned JSON."""
        user_prompt = (
            "Evaluate the following user feedback sentence and return JSON only.\n\n"
            f"Feedback: {feedback_sentence}"
        )

        result = self.llm_client.generate_json(
            system_prompt=self.prompt,
            user_prompt=user_prompt,
            validator=validate_evaluated_feedback,
        )

        # Keep the original input sentence as the final source of truth.
        result["raw_feedback"] = feedback_sentence
        return validate_evaluated_feedback(result)

    def evaluate_all(self, feedback_sentences: list[str]) -> list[EvaluatedFeedback]:
        """Evaluate each feedback sentence in sequence."""
        evaluated_records: list[EvaluatedFeedback] = []

        for index, sentence in enumerate(feedback_sentences, start=1):
            print(f"Evaluating feedback {index}/{len(feedback_sentences)}...")
            evaluated_records.append(self.evaluate_feedback(sentence))

        return evaluated_records
