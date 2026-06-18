# Hold My Beer: OPRO-style Human-Robot Handoff Optimization

This is a small offline Python demo that mirrors the core idea of OPRO:

1. An LLM evaluates natural-language human feedback from robot handoff attempts.
2. The evaluated history is fed back into the LLM as an optimizer.
3. The optimizer proposes the next improved robot handoff strategy.

The scenario is a UR5 robot arm with a gripper handing a beer can to a user. This project does not connect to a real robot.

## Project structure

```text
hold_my_beer_demo/
├── feedback.txt
├── main.py
├── evaluator.py
├── optimizer.py
├── llm_client.py
├── schema.py
├── prompts/
│   ├── evaluator_prompt.txt
│   └── optimizer_prompt.txt
├── outputs/
│   ├── evaluated_feedback.json
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
- `outputs/evaluated_feedback.json`: structured evaluator output for every feedback line
- `outputs/next_strategy.json`: the next proposed handoff strategy

## Notes

- The sample `feedback.txt` contains 100 example lines.
- The code uses JSON mode for LLM calls and validates the returned schema.
- If the model returns invalid JSON or the API call fails, the client retries automatically.
- This is an offline demo only. No real robot control is included.
