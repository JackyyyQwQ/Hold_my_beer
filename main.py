from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from evaluator import FeedbackEvaluator
from llm_client import LLMClient
from optimizer import StrategyOptimizer
from schema import EvaluatedFeedback, NextStrategy, Strategy, StrategyScorePair, validate_strategy


def clean_feedback_line(line: str) -> str:
    """Remove leading numbering like '1. ' or '2)' from a feedback line."""
    cleaned_line = line.strip()
    return re.sub(r"^\d+[\.\)]\s*", "", cleaned_line)


def load_feedback_lines(feedback_path: Path) -> list[str]:
    """Load all non-empty feedback entries in file order."""
    lines = feedback_path.read_text(encoding="utf-8").splitlines()
    return [clean_feedback_line(line) for line in lines if line.strip()]


def save_json(output_path: Path, data: object) -> None:
    """Write JSON output with readable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_initial_strategy(strategy_path: Path) -> Strategy:
    """Load the baseline strategy used before the first feedback arrives."""
    raw_strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    if not isinstance(raw_strategy, dict):
        raise ValueError("initial_strategy.json must contain one JSON object.")
    return validate_strategy(raw_strategy)


def load_task_description(task_description_path: Path) -> str:
    """Load the optimizer task description from disk."""
    task_description = task_description_path.read_text(encoding="utf-8").strip()
    if not task_description:
        raise ValueError("task_description.txt is empty.")
    return task_description


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    feedback_path = base_dir / "feedback.txt"
    initial_strategy_path = base_dir / "initial_strategy.json"
    task_description_path = base_dir / "prompts" / "task_description.txt"
    evaluated_output_path = base_dir / "outputs" / "evaluated_feedback.json"
    pair_output_path = base_dir / "outputs" / "strategy_score_pairs.json"
    next_strategy_output_path = base_dir / "outputs" / "next_strategy.json"

    evaluator_prompt_path = base_dir / "prompts" / "evaluator_prompt.txt"
    optimizer_prompt_path = base_dir / "prompts" / "optimizer_prompt.txt"

    # Load local inputs and initialize the shared LLM client.
    feedback_lines = load_feedback_lines(feedback_path)
    if not feedback_lines:
        raise ValueError("feedback.txt is empty. Add one feedback sentence per line before running the demo.")
    if not initial_strategy_path.exists():
        raise ValueError("initial_strategy.json is missing. Add the starting strategy before running the demo.")
    if not task_description_path.exists():
        raise ValueError("task_description.txt is missing. Add the optimizer task description before running the demo.")

    llm_client = LLMClient()
    evaluator = FeedbackEvaluator(llm_client, evaluator_prompt_path)
    optimizer = StrategyOptimizer(llm_client, optimizer_prompt_path)
    task_description = load_task_description(task_description_path)
    current_strategy = load_initial_strategy(initial_strategy_path)

    # Step 1: evaluate one feedback at a time and convert it into OPRO history.
    evaluated_feedback: list[EvaluatedFeedback] = []
    strategy_score_pairs: list[StrategyScorePair] = []
    next_strategy: NextStrategy | None = None

    for index, sentence in enumerate(feedback_lines, start=1):
        print(f"Evaluating feedback {index}/{len(feedback_lines)}...")
        evaluated_record = evaluator.evaluate_feedback(current_strategy, sentence)
        evaluated_feedback.append(evaluated_record)

        strategy_score_pair = optimizer.build_strategy_score_pair(current_strategy, evaluated_record)
        strategy_score_pairs.append(strategy_score_pair)

        print(f"Optimizing strategy with {len(strategy_score_pairs)} strategy-score pair(s)...")
        next_strategy = optimizer.propose_next_strategy(task_description, strategy_score_pairs)
        current_strategy = optimizer.next_strategy_to_strategy(next_strategy)

    save_json(evaluated_output_path, evaluated_feedback)
    save_json(pair_output_path, strategy_score_pairs)

    if next_strategy is None:
        raise RuntimeError("No strategy was generated from the provided feedback.")

    save_json(next_strategy_output_path, next_strategy)

    scores = [record["score"] for record in evaluated_feedback]
    average_score = statistics.mean(scores) if scores else 0.0
    best_score = max(scores) if scores else 0

    # Print a short demo summary for quick inspection.
    print("\nDemo summary")
    print(f"- Feedback sentences processed: {len(feedback_lines)}")
    print(f"- Strategy-score pairs generated: {len(strategy_score_pairs)}")
    print(f"- Average score: {average_score:.2f}")
    print(f"- Best score: {best_score}")
    print("- Next strategy:")
    print(json.dumps(next_strategy, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Demo failed: {error}")
        raise
