# P2C-Ex Evaluation Protocol Reference

Based on: Paper2Code-Extra (P2C-Ex) from Hong et al., 2025.

## Problem with original Paper2Code reference-free evaluation

The original reference-free metric systematically overestimates repository quality:

- Empty repositories score 3.89/5 (exceeds all model-generated results)
- Config-only repos (no executable code) score 4.67/5
- Root cause: evaluator hallucination — evaluators infer non-existent code from documentation

## P2C-Ex solution

Augment the evaluation prompt with explicit repository-level information:

1. **File count** — tell the evaluator how many files exist
2. **Repository structure** — provide the hierarchical file tree
3. **Improved instructions** — explicit anti-hallucination guidance

Result: Pearson correlation with reference-based metric improves from 0.423 to 0.862.

## Evaluation prompt template

Use this template when evaluating a generated paper-to-code repository:

```
You will be given a research paper along with its corresponding code repository.

Your task is to rate the code repository on one metric and provide a critique highlighting key differences.

---

Evaluation Criteria:

Correctness (1-5): The quality of the repository in accurately implementing the paper's
methodology and algorithms without logical errors.

1: Very Poor. Does not correctly implement core concepts. Major logical errors or missing components.
2: Poor. Attempts implementation but contains significant mistakes or missing components.
3: Fair. Some core components correctly implemented, but notable errors or inaccuracies.
4: Good. Correctly implements key components and methodology with only minor inaccuracies.
5: Excellent. Fully and accurately implements all key components without logical errors.

---

Evaluation Steps:

1. Identify key aspects of the paper (core concepts, methodology, algorithms)
2. Examine the code repository for alignment with the paper
3. Identify logical errors and deviations
4. Provide critique: for each issue, specify affected files and functions
5. Assess completeness: are all critical components implemented?
6. Code verification: verify all key components have actual code, not just documentation
7. Assign score (1-5)

---

Severity Level:

- High: Missing/incorrect implementation of core concepts, fundamental to methodology
- Medium: Issues in training logic, data preprocessing, or core functionality
- Low: Errors in features causing deviations but workable; evaluation process issues

---

Sample:

Research Paper:
{{Paper}}

Code Repository:
File Count: {{File_Count}}
File Structure: {{File_Structure}}
{{Code}}

---

Provide critique list and single numerical rating (1-5) in this JSON format, no commentary:

{
    "critique_list": [
        {
            "file_name": "example.py",
            "func_name": "train_loop",
            "severity_level": "medium",
            "critique": "Description of the issue."
        }
    ],
    "score": 3
}
```

## Anti-hallucination checklist

When evaluating, always:

- [ ] Count actual files — do not assume files exist based on README descriptions
- [ ] Read actual code — do not infer implementation from config files or documentation
- [ ] Check for empty files or stub implementations
- [ ] Verify imports resolve to actual code in the repository
- [ ] Distinguish between "described in README" and "implemented in code"
