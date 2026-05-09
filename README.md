# AgentPSO DeepMath Run Guide

This folder contains the submission code needed to train AgentPSO on DeepMath and evaluate it on a held-out DeepMath test pool.

## 1. Setup

Activate your Python environment first. For example:

```bash
conda activate AgentPSO
cd path/to/this/directory
```

If you use the OpenAI backend, set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 2. Quick Start

Run:

```bash
bash run_agentpso_deepmath.sh
```

By default, this script will:

- install packages from `requirements.txt`
- download DeepMath-103K shard 0 if it is missing
- train AgentPSO on DeepMath
- build a non-overlapping DeepMath test pool from the same shard
- run final test evaluation
- save outputs under `./deepmath_agentpso_train100_val100_test200/`

## 3. Common Options

Skip dependency installation:

```bash
INSTALL_REQUIREMENTS=0 bash run_agentpso_deepmath.sh
```

Activate a conda environment inside the script:

```bash
CONDA_ENV=AgentPSO bash run_agentpso_deepmath.sh
```

Run a fast smoke test without API calls:

```bash
BACKEND=mock NUM_ITERATIONS=1 TEST_LIMIT=20 bash run_agentpso_deepmath.sh
```

Use a custom run name:

```bash
RUN_NAME=my_deepmath_run bash run_agentpso_deepmath.sh
```

Use an existing DeepMath parquet file:

```bash
DEEPMATH_DATASET=../../data/DeepMath-103K/data/train-00000-of-00010.parquet bash run_agentpso_deepmath.sh
```

## 4. Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `CONDA_ENV` | empty | If set, the script activates this conda environment. |
| `PYTHON_BIN` | `python3` | Python executable to use. |
| `BACKEND` | `openai` | LLM backend: `openai`, `local-openai`, `anthropic`, or `mock`. |
| `MODEL` | `gpt-5.4-mini` | Model name passed to AgentPSO. |
| `RUN_NAME` | `deepmath_agentpso_train100_val100_test200` | Output run directory name. |
| `OUTPUT_ROOT` | current directory | Root directory for outputs. |
| `DEEPMATH_DATASET` | `../../data/DeepMath-103K/data/train-00000-of-00010.parquet` | DeepMath parquet file path. |
| `INSTALL_REQUIREMENTS` | `1` | If `1`, install `requirements.txt` before running. |
| `DOWNLOAD_DEEPMATH` | `1` | If `1`, download DeepMath if the dataset file is missing. |
| `DEEPMATH_SHARDS` | `0` | Shards to download. Examples: `0`, `0,1`, `all`. |
| `TRAIN_POOL_SIZE` | `100` | Number of training examples. |
| `VALIDATION_POOL_SIZE` | `100` | Number of validation examples. |
| `TEST_LIMIT` | `200` | Number of final test examples. |
| `NUM_ITERATIONS` | `10` | Number of AgentPSO training iterations. |
| `TEST_MODE` | `personal_best` | Final test mode: `personal_best` or `adaptive_agentpso`. |
| `OVERWRITE` | `1` | If `1`, overwrite an existing run directory. |

## 5. Output Files

The default output directory is:

```text
./deepmath_agentpso_train100_val100_test200/
```

Important files and folders:

- `config.json`: run configuration
- `global_best.md`: final global-best skill
- `skills/`: per-agent skill history
- `results/train/`: training results
- `results/validation/`: validation results
- `results/test/`: final test results
- `scores/final_test_summary.json`: final test summary
- `launch.log`: full launch log

## 6. Compatibility Script

The old script name is still available:

```bash
bash run_deepmath_train_math_test_gpt54mini.sh
```

It simply forwards to `run_agentpso_deepmath.sh`.
