# Running the Original HiRAS Implementation

These notes are for users who want to run the actual HiRAS codebase from https://github.com/KOU-199024/HiRAS, not just apply the methodology.

## Prerequisites

- Python 3.11 (conda recommended)
- 8x NVIDIA H200 GPUs (as per the paper's experimental setup)
- Conda for environment management

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create base experiment environment
conda create -n ExpBase python=3.11
conda activate ExpBase
pip install -r exp_env_example.txt
conda deactivate

# 3. Configure model API in run.py
#    Edit the API_BASE_URL and api_key values for your model provider
```

## Data preparation

Download benchmarks:
- PaperBench: https://github.com/openai/frontier-evals/tree/main/project/paperbench
- Paper2Code: https://github.com/going-doer/Paper2Code

Place under `data/`:
```
data/
├── paperbench/
│   ├── adaptive-pruning/
│   │   ├── paper.md
│   │   └── addendum.md
│   └── ...
└── paper2code/
    ├── dataset_info.json
    ├── iclr2024/
    ├── icml2024/
    └── nips2024/
```

## Running

```bash
# PaperBench (debug split)
python run.py --model deepseek-v3 --benchmark paperbench --split debug \
    --base_env_name ExpBase --exp_env_name Temp4Exp

# PaperBench (all papers)
python run.py --model deepseek-v3 --benchmark paperbench --split all \
    --base_env_name ExpBase --exp_env_name Temp4Exp

# Paper2Code
python run.py --model deepseek-v3 --benchmark paper2code --split debug \
    --base_env_name ExpBase --exp_env_name Temp4Exp
```

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | deepseek-v3 | Backbone model (deepseek-v3, deepseek-r1, qwen3-480b, or local model name) |
| `--benchmark` | paperbench | Benchmark to run (paperbench, paper2code) |
| `--split` | debug | Paper split (debug, lite, human, all, dev) |
| `--base_env_name` | ExpBase | Name of the base conda environment |
| `--exp_env_name` | Temp4Exp-dpsk-v3 | Name of the temporary execution environment (cloned from base) |
| `--step_multiplier` | 1 | Multiply agent step limits for more thorough reproduction |
| `--data_dir` | data/ | Path to benchmark data |
| `--output_dir` | output/ | Path to save results |
| `--gpu_count` | 8 | Number of GPUs for local model serving |

## Architecture notes

- Uses `smolagents` as the agent scaffold
- Agent classes in `agents/`: `ManagerAgent`, `PlanningAgent`, `AnalysingAgent`, `CodingAgent`, `ExecutionAgent`
- Prompts in `prompts/paperbench/` and `prompts/paper2code/`
- Each agent runs as a `CodeAgent` with file management tools
- Execution agent additionally has a `BashTool` that runs commands via `conda run -n <env>`
- Environments are cloned per-run and destroyed after to ensure isolation

## Cost tracking

Token usage and cost are tracked per-agent in `cost/` directories. Summary in `cost.json` at the run level.

## Limitations

- Requires substantial GPU resources for local model serving
- Execution phase is the primary bottleneck (most failures happen there)
- Time and token costs are higher than sequential pipelines due to hierarchical supervision
- Some experiments may fail due to dataset/model download issues rather than code errors
