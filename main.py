from __future__ import annotations

import json
import statistics
from pathlib import Path

from evaluator import FeedbackEvaluator
from llm_client import LLMClient
from optimizer import StrategyOptimizer


def load_feedback_lines(feedback_path: Path) -> list[str]:
    """Read one feedback sentence per non-empty line."""
    lines = feedback_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def save_json(output_path: Path, data: object) -> None:
    """Write JSON output with readable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    feedback_path = base_dir / "feedback.txt"
    evaluated_output_path = base_dir / "outputs" / "evaluated_feedback.json"
    next_strategy_output_path = base_dir / "outputs" / "next_strategy.json"

    evaluator_prompt_path = base_dir / "prompts" / "evaluator_prompt.txt"
    optimizer_prompt_path = base_dir / "prompts" / "optimizer_prompt.txt"

    # Load local inputs and initialize the shared LLM client.
    feedback_lines = load_feedback_lines(feedback_path)
    if not feedback_lines:
        raise ValueError("feedback.txt is empty. Add one feedback sentence per line before running the demo.")

    llm_client = LLMClient()
    evaluator = FeedbackEvaluator(llm_client, evaluator_prompt_path)
    optimizer = StrategyOptimizer(llm_client, optimizer_prompt_path)

    # Step 1: score and structure each feedback sentence.
    evaluated_feedback = evaluator.evaluate_all(feedback_lines)
    save_json(evaluated_output_path, evaluated_feedback)

    # Step 2: read the saved history file and optimize the next policy.
    next_strategy = optimizer.propose_next_strategy_from_file(evaluated_output_path)
    save_json(next_strategy_output_path, next_strategy)

    scores = [record["score"] for record in evaluated_feedback]
    average_score = statistics.mean(scores) if scores else 0.0
    best_score = max(scores) if scores else 0

    # Print a short demo summary for quick inspection.
    print("\nDemo summary")
    print(f"- Feedback sentences processed: {len(feedback_lines)}")
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
