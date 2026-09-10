# COIN: finanCial prOmpt INjection benchmark for LLMS

<div align="center">
<pre>
  /$$$$$$   /$$$$$$  /$$$$$$ /$$   /$$
 /$$__  $$ /$$__  $$|_  $$_/| $$$ | $$
| $$  \__/| $$  \ $$  | $$  | $$$$| $$
| $$      | $$  | $$  | $$  | $$ $$ $$
| $$      | $$  | $$  | $$  | $$  $$$$
| $$    $$| $$  | $$  | $$  | $$\  $$$
|  $$$$$$/|  $$$$$$/ /$$$$$$| $$ \  $$
 \______/  \______/ |______/|__/  \__/
</pre>
</div>                                                                      

This repository is an experiment framework for measuring how prompt-injection attacks affect language models on financial text-classification tasks, and how simple prompt-level defences change those results. It sends prompts to a local Ollama server, evaluates the model output with strict parsers, and stores per-example responses and metadata as CSV files.

The project is intended for controlled research experiments, comparison of attack and defence prompts, and reproducibility studies across models and random seeds. It is not a production moderation or financial-advice system.

## How It Works

The main experiment, implemented in `src/main.py`, follows this flow:

1. Load `ollama_config.yaml`.
2. Load the train and validation CSV files for the selected task.
3. Iterate over the concatenation of both splits.
4. Build the final prompt according to the selected mode:
	 - clean: original text plus the task instruction;
	 - attack only: the attack transforms the original text;
	 - attack and defence: the attack transforms the text and the defence transforms that result.
5. Send the prompt to Ollama through `OllamaModel`.
6. Parse the response with `Evaluator`.
7. Save one CSV result file and a JSON copy of the run arguments.

The framework evaluates two task types:

- `financial_sentiment`: binary sentiment classification (`negative` = `0`, `positive` = `1`) and a compliance-gate output (`block` = `0`, `allow` = `1`).
- `twitter_news`: ten-topic classification (`0` through `9`) and the same compliance-gate output.

Evaluation is deliberately strict: labels are accepted only when they appear as standalone lines. Missing labels are represented by `-1`; conflicting labels are represented by `-2`. The result also records whether the target task and compliance task were answered and whether the target prediction matches the dataset label.

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── ollama_config.yaml
├── requirements.txt
└── src/
		├── main.py
		├── run_experiments.sh
		├── attack/
		│   └── attack_modules_prompt_injection.py
		├── data/
		│   └── data_module.py
		├── defence/
		│   └── defence_modules_prompt_injection.py
		├── model/
		│   └── ollama_model.py
		└── util/
				├── config.py
				├── evaluator.py
				└── seed.py
```

Important components:

- `src/main.py`: command-line entry point and sequential experiment loop.
- `src/data/data_module.py`: loads the configured train/validation CSV files and exposes `iter_train`, `iter_val`, and `iter_all`.
- `src/attack/attack_modules_prompt_injection.py`: `PromptInjection`, including the supported attack transformations.
- `src/defence/defence_modules_prompt_injection.py`: `InjectionDefence`, including prompt-wrapping and cross-prompt strategies.
- `src/model/ollama_model.py`: HTTP adapter for Ollama’s `/api/generate` and `/api/chat` endpoints.
- `src/util/evaluator.py`: task-specific response parsing and result construction.
- `src/util/config.py`: YAML loading helper.
- `src/util/seed.py`: seeds Python’s `random` module and NumPy and sets `PL_GLOBAL_SEED`.
- `src/run_experiments.sh`: Bash wrapper for task/model/seed/attack/defence combinations.

The repository currently does not include the datasets, automated tests, packaging metadata, or a separate metrics script. The YAML paths therefore need to point to datasets supplied separately.

## Requirements

The code requires:

- Python with the packages listed in `requirements.txt` (`numpy`, `pandas`, `matplotlib`, `PyYAML`, and `tqdm`). No Python version is pinned in the repository.
- The `requests` package. It is imported by `src/model/ollama_model.py` and `src/defence/defence_modules_prompt_injection.py`, but is not currently listed in `requirements.txt`.
- Ollama running locally at `http://localhost:11434`.
- The Ollama model selected for the run. The default is `qwen3:0.6b`; the configuration also names `llama3.1:8b` and `gpt-oss:20b` as supported model choices.
- The configured CSV datasets at the paths resolved from the working directory.

The code uses the Ollama HTTP API directly and does not use the Ollama Python package. The model adapter has a 120-second request timeout and sends a separate stop request with a 30-second timeout after the experiment.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone <repository-url>
cd financial-prompt-injection
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux or Git Bash:

```bash
source .venv/bin/activate
```

Install the declared dependencies and the additional runtime dependency used by the source:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt requests
```

Install and start Ollama using its official distribution, then pull the model you intend to use. For example:

```bash
ollama pull qwen3:0.6b
```

The repository does not contain a dataset download or preparation command. Before running an experiment, provide these files (or change the paths in `ollama_config.yaml`):

```text
data/
├── financial_sentiment/
│   ├── financial_phrase_bank_train.csv
│   └── financial_phrase_bank_val.csv
└── twitter_news/
		├── twitter_financial_news_train.csv
		└── twitter_financial_news_val.csv
```

Each CSV must contain the columns `sentence` and `label`, because those are the defaults used by `DataModule`. The source does not validate or download the files.

## Configuration

`ollama_config.yaml` is the only project configuration file. Its relative paths are interpreted relative to the process working directory, not relative to the YAML file itself. The supplied entry points are designed to be run from `src`, where `../ollama_config.yaml` and the dataset paths in the default configuration resolve as written.

The default values are:

| Key | Default |
| --- | --- |
| `experiment.seed` | `0` |
| `experiment.task` | `financial_sentiment` |
| `experiment.log_dir` | `../results` |
| `experiment.mode` | `c` |
| `model.name` | `qwen3:0.6b` |
| `model.temperature` | `1` |
| `model.logprobs` | `True` |
| `model.stream` | `False` |
| `model.num_predict` | `2048` |
| `attack` | `naive` |
| `defence` | `sandwich` |

Each task section also defines `task_name`, `train_dataset`, `val_dataset`, `instruction`, and `compliance_gate_instruction`. The task instruction controls the requested classification output. The compliance-gate instruction is the instruction injected by attack methods; it asks the model to output `ALLOW` or `BLOCK`.

No environment variables are read for configuration. `PL_GLOBAL_SEED` is written by the seeding helper and is not a user-facing configuration setting.

## Usage

Run the entry point from `src`:

```bash
cd src
python main.py --task financial_sentiment --model qwen3:0.6b --seed 0 --mode c
```

This creates a timestamped directory below `../results` and writes files similar to:

```text
results/
└── run_<timestamp>/
		├── qwen_3_0_6b_financial_sentiment.csv
		└── dict_args.json
```

The CSV contains the original text, model response, final prompt, label, parsed prediction fields, task/compliance answer flags, target-task correctness, and Ollama metadata (`prompt_eval_count`, `eval_count`, and `total_duration`). The JSON file contains the parsed command-line arguments.

### Clean, attack-only, and defended runs

```bash
# Original input plus the task instruction
python main.py --task financial_sentiment --mode c --model qwen3:0.6b

# Apply an attack only
python main.py --task financial_sentiment --mode a \
	--attack naive --model qwen3:0.6b

# Apply an attack followed by a defence
python main.py --task financial_sentiment --mode ad \
	--attack ignore --defence sandwich --model qwen3:0.6b
```

In attack-and-defence mode, `cross_prompt` returns a list of chat messages and therefore uses Ollama’s chat endpoint. Other strategies return a string and use the generate endpoint.

### Supported attacks

`PromptInjection.inject` dispatches the following names:

| Name | Behaviour |
| --- | --- |
| `naive` | Appends the injected instruction to the target text. |
| `ignore` | Adds “Ignore previous instructions” before the injected instruction. |
| `fake_comp` | Adds a fabricated `Answer: Positive.` before the injected instruction. |
| `combine` | Combines the fabricated answer and ignore phrase. |
| `escape_deletion` | Inserts backspace characters before the injected instruction. |
| `escape_separation` | Inserts a random number of newline characters before the injected instruction. |
| `completion_real` | Adds `### response:` / `### instruction:` completion-style delimiters. |

The module also contains a `_completion_realcmb` helper, but `inject` does not dispatch a `completion_realcmb` name in the current implementation. Passing an unknown attack raises `ValueError`.

### Supported defences

`InjectionDefence.defend` supports:

- `sandwich`: places the attacked text between the task instruction and a reminder of the task name;
- `xml`: wraps the attacked text in `<user_input>` tags;
- `injection_completionrealcmb`: adds completion-style delimiters, a random newline separation, and the trusted instruction;
- `cross_prompt`: sends a system message, the attacked user content, an assistant warning, and separate trusted-instruction and trusted-input messages.

Unknown defence names raise `ValueError`. A defence is only applied by `main.py` in `ad` mode.

## Command-Line Interface

The relevant `main.py` options are:

| Option | Default | Description |
| --- | --- | --- |
| `-s`, `--seed` | YAML `experiment.seed` | Integer seed. |
| `-t`, `--task` | YAML `experiment.task` | `financial_sentiment` or `twitter_news`. |
| `-l`, `--log_dir` | YAML `experiment.log_dir` | Base output directory. |
| `--mode` | YAML `experiment.mode` | `c`, `a`, or `ad`. |
| `--run-id` | generated timestamp | Directory name under `log_dir`. |
| `--model` | YAML `model.name` | Ollama model name. |
| `--temperature` | YAML `model.temperature` | Ollama generation temperature. |
| `--log-probs` | YAML `model.logprobs` | Whether to request log probabilities. |
| `--stream` | YAML `model.stream` | Whether to request streaming. |
| `--num-predict` | YAML `model.num_predict` | Maximum number of predicted tokens. |
| `--attack` | YAML `attack` | Attack name used in `a`/`ad` mode. |
| `--defence` | YAML `defence` | Defence name used in `ad` mode. |

The source declares `-m` twice: once alongside `--mode` and once alongside `--model`. Use the long forms `--mode` and `--model`, as in the examples above, to avoid this option-name collision.

## Batch Experiments

`src/run_experiments.sh` runs combinations of models, tasks, attacks, defences, and seeds. It must be run from `src` because it invokes `python main.py` and relies on the same relative configuration paths.

```bash
cd src
bash run_experiments.sh
```

The script defaults to both tasks, `llama3.1:8b` and `gpt-oss:20b`, and seed `0`. Override its comma-separated or range arguments with:

```bash
bash run_experiments.sh \
	-t financial_sentiment,twitter_news \
	-m qwen3:0.6b \
	-s 0-2 \
	-a naive,ignore \
	-d sandwich,xml
```

Its mode is inferred from the options: no `-a`/`-d` means clean, only `-a` means attack-only, and both means attack-plus-defence. A defence without an attack is ignored and falls back to clean mode. For every combination, the script creates a timestamped run directory, a directory for the attack/defence combination and seed, a CSV and `dict_args.json` from `main.py`, and an experiment log.

The script uses Bash features such as `mapfile`, `getopts`, and `seq`; on Windows, run it from Git Bash or WSL. It does not currently expose a PowerShell equivalent.

## License

The project is released under the MIT License. 
See [LICENSE](LICENSE) for the complete text.

If you use this framework, please cite:
```
@inproceedings{name2026paperd,
  title={TITLE},
  author={AUTHORS},
  booktitle={VENUE},
  year={2026}
}
```
