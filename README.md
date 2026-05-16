# Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?

This repository contains the code for the paper "Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?" (AAAI-2026).

## Overview

The code supports various debate strategies and evaluation metrics for analyzing agent interactions.

## Requirements

- Python 3.x
- openai
- numpy
- tqdm
- asyncio

## Configuration

Before running the code, please configure your API settings in the respective Python files:

```python
API_URL = "YOUR_API_URL_HERE"
API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "your-model-name"
MODEL_TAG = "your-model-tag"
```

The code supports OpenAI-compatible API endpoints.

## Directory Structure

```
├── common/                    # Common utilities
│   ├── math_equivalence.py   # Math expression equivalence checking
│   └── utils.py              # Utility functions
├── prompt/                    # Prompt templates and configurations
├── data/                      # Dataset files (in model-specific directories)
├── results/                   # Output results (in model-specific directories)
├── case_study.py             # Case study analysis script
├── debate_bbh_qwen3b.py      # Main debate implementation
└── eval_all_round.py         # Evaluation script
```

## Usage

### Running Debates

To run multi-agent debates:

```bash
python debate_bbh_qwen3b.py
```

### Evaluation

To evaluate debate results:

```bash
python eval_all_round.py
```

### Case Study

To run case study analysis:

```bash
python case_study.py
```

```bibtex
@article{your-paper-2026,
  title={Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?},
  author={Qian Zhang, Yan Zheng, Jinyi Liu, Hebin Liang, Lanjun Wang},
  journal={AAAI},
  year={2026}
}
```
