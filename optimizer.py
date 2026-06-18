from __future__ import annotations

import json
from pathlib import Path

from llm_client import LLMClient
from schema import (
    EvaluatedFeedback,
    NextStrategy,
    Strategy,
    StrategyScorePair,
    validate_strategy,
    validate_next_strategy,
    validate_strategy_score_pair_list,
)


class StrategyOptimizer:
    """Use prior strategy-score pairs to propose the next handoff strategy."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path) -> None:
        self.llm_client = llm_client
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def load_history(self, history_path: Path) -> list[StrategyScorePair]:
        """Load and validate prior strategy-score pairs from disk."""
        raw_history = json.loads(history_path.read_text(encoding="utf-8"))
        return validate_strategy_score_pair_list(raw_history)

    def build_strategy_score_pair(
        self,
        strategy: Strategy,
        evaluated_feedback: EvaluatedFeedback,
    ) -> StrategyScorePair:
        """Pair the previous strategy with the evaluated feedback it produced."""
        return {
            "strategy": validate_strategy(strategy),
            "evaluated_feedback": evaluated_feedback,
        }

    def next_strategy_to_strategy(self, next_strategy: NextStrategy) -> Strategy:
        """Drop optimizer-only fields and keep the strategy fields for the next round."""
        return validate_strategy(
            {
                "height": next_strategy["height"],
                "distance": next_strategy["distance"],
                "speed": next_strategy["speed"],
                "orientation": next_strategy["orientation"],
                "handoff_timing": next_strategy["handoff_timing"],
            }
        )

    def propose_next_strategy(
        self,
        task_description: str,
        history: list[StrategyScorePair],
    ) -> NextStrategy:
        """Send the task description and strategy-score history to the optimizer prompt."""
        meta_prompt_payload = {
            "task_description": task_description,
            "strategy_score_pairs": history,
        }
        user_prompt = (
            "Review the task description and strategy-score pairs below. "
            "Propose the next improved robot handoff strategy as JSON only.\n\n"
            f"{json.dumps(meta_prompt_payload, indent=2)}"
        )

        return self.llm_client.generate_json(
            system_prompt=self.prompt,
            user_prompt=user_prompt,
            validator=validate_next_strategy,
        )

    def propose_next_strategy_from_file(
        self,
        task_description: str,
        history_path: Path,
    ) -> NextStrategy:
        """Read the saved history file, then ask the LLM for the next strategy."""
        history = self.load_history(history_path)
        return self.propose_next_strategy(task_description, history)
