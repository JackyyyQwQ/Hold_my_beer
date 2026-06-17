from __future__ import annotations

import json
from pathlib import Path

from llm_client import LLMClient
from schema import (
    EvaluatedFeedback,
    NextStrategy,
    validate_evaluated_feedback_list,
    validate_next_strategy,
)


class StrategyOptimizer:
    """Use prior evaluated feedback to propose the next handoff strategy."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path) -> None:
        self.llm_client = llm_client
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def load_history(self, history_path: Path) -> list[EvaluatedFeedback]:
        """Load and validate prior evaluated feedback from disk."""
        raw_history = json.loads(history_path.read_text(encoding="utf-8"))
        return validate_evaluated_feedback_list(raw_history)

    def propose_next_strategy(self, history: list[EvaluatedFeedback]) -> NextStrategy:
        """Send the evaluation history to the optimizer prompt."""
        history_json = json.dumps(history, indent=2)
        user_prompt = (
            "Review this evaluated feedback history and propose the next improved handoff strategy.\n\n"
            f"{history_json}"
        )

        return self.llm_client.generate_json(
            system_prompt=self.prompt,
            user_prompt=user_prompt,
            validator=validate_next_strategy,
        )

    def propose_next_strategy_from_file(self, history_path: Path) -> NextStrategy:
        """Read the saved history file, then ask the LLM for the next strategy."""
        history = self.load_history(history_path)
        return self.propose_next_strategy(history)
