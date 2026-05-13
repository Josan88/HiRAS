# HiRAS Methodology Reference

Based on: "HiRAS: A Hierarchical Multi-Agent Framework for Paper-to-Code Generation and Execution" (Hong et al., 2025, arXiv:2604.17745v2)

## What HiRAS is

HiRAS (Hierarchical Research Agent System) is a multi-agent framework for end-to-end experiment reproduction from research papers. Unlike prior sequential pipelines (PaperCoder, AutoReproduce), HiRAS introduces supervisory manager agents that actively inspect intermediate artefacts, diagnose failures, and re-invoke specialised agents to correct errors.

## Key insight

Existing paper-to-code systems use fixed sequential pipelines with limited global supervision. Errors introduced early propagate unchecked, resulting in stalled workflows and incomplete repositories. HiRAS addresses this with hierarchical management: a global manager agent has authority to inspect any intermediate output and re-invoke the responsible agent.

## Agent architecture

### Specialised agents (leaf-level workers)

Each handles one phase of reproduction:

| Agent | Phase | Output |
|---|---|---|
| Overall planning agent | High-level experiment plan | `plan.md` |
| Architecture planning agent | Software structure design | `architecture.md` |
| Dependency planning agent | Inter-module dependencies | `dependency.md` |
| Config planning agent | Hyperparameter extraction | `config.yaml` |
| Analysis agent | Per-component implementation analysis | `analysis/*.md` |
| Coding agent | Code generation | `code/*.py` |
| Execution agent | Run code, record results | `results/*` |

### Manager agents (supervisors)

Two levels of hierarchy:

**Planning manager** — oversees the four planning sub-agents:
- Subordinates: overall, architecture, dependency, config planning agents
- Validates each planning artefact before passing to the next sub-agent
- Can re-invoke any planning sub-agent if output is insufficient

**Global manager** — oversees the entire reproduction pipeline:
- Subordinates: planning manager, analysis agent, coding agent, execution agent
- Inspects workspace artefacts after each phase
- Detects incomplete outputs, hallucinations, or errors
- Re-invokes the responsible agent with corrective instructions
- Coordinates cross-phase error resolution

### Invocation model

All agents follow the ReAct paradigm (reason + act + observe). Each agent:
- Has a maximum of K reasoning-action iterations per invocation
- Can call `end_task` to submit outputs early
- Operates in a shared workspace via file system tools (`list_dir`, `read_file`, `write_file`)
- Execution agent additionally has `bash` for running commands

When a manager invokes a subordinate:
1. Manager provides an instruction prompt specifying the task
2. Subordinate executes up to K steps, producing artefacts in the shared workspace
3. Subordinate returns a report via `final_answer`
4. Manager inspects the workspace to validate the output
5. If insufficient, manager re-invokes with corrective instructions

## What makes HiRAS different from prior work

| Aspect | PaperCoder | AutoReproduce | HiRAS |
|---|---|---|---|
| Pipeline | Sequential, fixed | Sequential + search | Hierarchical with supervision |
| Error handling | None (error propagates) | None | Manager re-invokes agents |
| Planning granularity | Single planning phase | Single planning phase | 4 sub-phases with dedicated agents |
| Execution validation | None | None | Dedicated execution agent + manager review |
| Artefact inspection | None | None | Manager inspects after every phase |

## Evaluation benchmarks

### PaperBench

- 20 ML papers from ICML 2024
- Tree-structured rubrics assessing: code development, execution, result matching
- Percentage scores (0-100%)
- PaperBench-CodeDev subset: code development only
- Full PaperBench: includes execution and result matching

### Paper2Code

- 90 papers from ICML 2024, NeurIPS 2024, ICLR 2024
- 1-5 scoring scale
- Reference-free: evaluate generated repo against paper only
- Reference-based: also provide gold repository

### Paper2Code-Extra (P2C-Ex)

HiRAS authors identified that the original Paper2Code reference-free metric systematically overestimates quality. Empty repositories can score 3.89/5 due to evaluator hallucination. P2C-Ex fixes this by adding:
- File count information
- Hierarchical repository structure illustration
- Improved instructions to mitigate hallucination

Result: Pearson correlation with reference-based metric improves from 0.423 to 0.862.

## Key findings from the paper

1. Hierarchical supervision alone contributes ~10% improvement over specialised agents without management.
2. Fine-grained phase decomposition outperforms monolithic planning agents.
3. Iterative self-refinement (agents correcting their own outputs) adds substantial gains.
4. Execution is the primary bottleneck — most failures happen at runtime, not code generation.
5. Only 3 out of 110 runs failed at the code generation stage; the rest failed during execution (import errors, environment setup, dataset/model downloads).
