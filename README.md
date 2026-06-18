# Hold My Beer: OPRO-style Human-Robot Handoff Optimization

This is a small offline Python demo that mirrors the core idea of OPRO:

1. An LLM evaluates natural-language human feedback from robot handoff attempts.
2. Each evaluated feedback item is converted into a strategy-score pair.
3. The task description plus accumulated strategy-score pairs are fed back into the LLM as an optimizer.
4. The optimizer proposes the next improved robot handoff strategy.

The scenario is a UR5 robot arm with a gripper handing a beer can to a user. This project does not connect to a real robot.

## Project structure

```text
hold_my_beer_demo/
├── initial_strategy.json
├── feedback.txt
├── main.py
├── evaluator.py
├── optimizer.py
├── llm_client.py
├── schema.py
├── prompts/
│   ├── evaluator_prompt.txt
│   ├── optimizer_prompt.txt
│   └── task_description.txt
├── outputs/
│   ├── evaluated_feedback.json
│   ├── strategy_score_pairs.json
│   └── next_strategy.json
└── README.md
```

## Requirements

- Python 3.10 or newer
- An OpenAI API key

## Setup

Install the OpenAI SDK:

```bash
pip install openai
```

Set your API key:

```bash
export OPENAI_API_KEY="my_key"
```

PowerShell version:

```powershell
$env:OPENAI_API_KEY="my_key"
```

Optional: choose a different model.

```bash
export OPENAI_MODEL="gpt-4o"
```

## Run

Change into the project directory and start the demo:

```bash
cd hold_my_beer_demo
python main.py
```

## Input and outputs

- `feedback.txt`: one user feedback sentence per line
- `initial_strategy.json`: the starting handoff strategy used before the first feedback
- `prompts/task_description.txt`: the optimization objective used by the optimizer
- `outputs/evaluated_feedback.json`: structured evaluator output for every feedback line
- `outputs/strategy_score_pairs.json`: OPRO history built from previous strategies plus evaluated feedback
- `outputs/next_strategy.json`: the next proposed handoff strategy

## Notes

- The demo processes the feedback lines in file order, one at a time.
- The code uses JSON mode for LLM calls and validates the returned schema.
- The evaluator receives the executed strategy plus one user feedback sentence.
- The optimizer receives a separate task description plus strategy-score pairs, where each pair links one previous strategy to the evaluated feedback it produced.
- If the model returns invalid JSON or the API call fails, the client retries automatically.
- This is an offline demo only. No real robot control is included.
