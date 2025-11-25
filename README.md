# DeepEval CI/CD - LLM Evaluation Pipeline

A comprehensive LLM evaluation framework using [DeepEval](https://github.com/confident-ai/deepeval) with AWS Bedrock Qwen models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepEval Evaluation                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │   Application    │          │   Judge Model    │         │
│  │     Model        │          │                  │         │
│  │                  │          │                  │         │
│  │  Qwen3-32B       │  ───▶    │  Qwen3-235B      │         │
│  │  (Under Test)    │  output  │  (Evaluator)     │         │
│  └──────────────────┘          └──────────────────┘         │
│                                        │                     │
│                                        ▼                     │
│                               ┌──────────────────┐          │
│                               │  Metric Scores   │          │
│                               │  Pass/Fail       │          │
│                               └──────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Models

| Role | Model | Model ID |
|------|-------|----------|
| Application (Under Test) | Qwen3-32B | `qwen.qwen3-32b-v1:0` |
| Judge (Evaluator) | Qwen3-235B | `qwen.qwen3-235b-a22b-2507-v1:0` |

## Metrics Covered

### Basic Metrics (`test_qwen_eval.py`)
- **Answer Relevancy**: Measures if the output is relevant to the input query
- **Faithfulness**: Measures if the output is faithful to the provided context
- **Hallucination**: Detects if the model makes up facts not in context

### RAG Metrics (`test_rag_metrics.py`)
- **Contextual Precision**: Measures if relevant context nodes are ranked higher
- **Contextual Recall**: Measures if all relevant info from context is captured
- **Contextual Relevancy**: Measures if retrieved context is relevant to query

### Custom G-Eval Metrics (`test_geval_metrics.py`)
- **Code Quality**: Custom metric for evaluating generated code
- **Explanation Clarity**: Custom metric for evaluating explanations
- **Technical Accuracy**: Custom metric for technical correctness
- **Conciseness**: Custom metric for response brevity

### Safety Metrics (`test_safety_metrics.py`)
- **Bias**: Detects biased content in model outputs
- **Toxicity**: Detects toxic or harmful content

### Dataset Evaluation (`test_dataset_eval.py`)
- Batch evaluation using `EvaluationDataset`
- Benchmark-style testing with category breakdown
- Pass rate aggregation

## Project Structure

```
deep-eval-cicd/
├── bedrock_qwen.py       # Application model client (Qwen3-32B)
├── qwen_judge.py         # Judge model wrapper for DeepEval (Qwen3-235B)
├── test_qwen_eval.py     # Basic metrics tests
├── test_rag_metrics.py   # RAG-specific metrics tests
├── test_geval_metrics.py # Custom G-Eval metrics tests
├── test_safety_metrics.py# Bias and Toxicity tests
├── test_dataset_eval.py  # Dataset-based evaluation tests
├── requirements.txt      # Python dependencies
└── .github/
    └── workflows/
        └── llm-eval.yml  # CI/CD pipeline
```

## Setup

### Prerequisites
- Python 3.11+
- AWS account with Bedrock access
- Access to Qwen models in `ap-south-1` region

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deep-eval-cicd.git
cd deep-eval-cicd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### AWS Configuration

Set your AWS credentials:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_SESSION_TOKEN="your-session-token"  # If using temporary credentials
export AWS_REGION="ap-south-1"
```

## Running Tests

### Run All Tests
```bash
deepeval test run
```

### Run Specific Test Files
```bash
# Basic metrics
deepeval test run test_qwen_eval.py -v

# RAG metrics
deepeval test run test_rag_metrics.py -v

# Custom G-Eval metrics
deepeval test run test_geval_metrics.py -v

# Safety metrics
deepeval test run test_safety_metrics.py -v

# Dataset evaluation
deepeval test run test_dataset_eval.py -v
```

### Run with pytest directly
```bash
pytest test_qwen_eval.py -v
```

## CI/CD Pipeline

The GitHub Actions workflow runs parallel evaluation jobs:

1. **eval-basic**: Answer Relevancy, Faithfulness, Hallucination
2. **eval-rag**: Contextual Precision, Recall, Relevancy
3. **eval-geval**: Custom G-Eval metrics
4. **eval-safety**: Bias and Toxicity detection
5. **eval-dataset**: Dataset-based batch evaluation

### Required Secrets

Configure these in your GitHub repository settings:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN` (if using temporary credentials)

## Extending the Framework

### Adding New Metrics

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

my_metric = GEval(
    name="My Custom Metric",
    model=qwen_judge,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    criteria="Your evaluation criteria here...",
    threshold=0.7,
)
```

### Creating Test Cases

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="Your question here",
    actual_output=model_response,
    expected_output="Expected answer",  # Optional
    retrieval_context=["Context 1", "Context 2"],  # For RAG
    context=["Context for hallucination check"],  # For hallucination
)
```

### Dataset Evaluation

```python
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(test_cases=[...])
results = evaluate(
    test_cases=dataset.test_cases,
    metrics=[metric1, metric2],
)
```

## DeepEval Features Demonstrated

- [x] Custom LLM Judge (Bedrock Qwen)
- [x] Answer Relevancy Metric
- [x] Faithfulness Metric
- [x] Hallucination Metric
- [x] Contextual Precision Metric
- [x] Contextual Recall Metric
- [x] Contextual Relevancy Metric
- [x] Bias Metric
- [x] Toxicity Metric
- [x] G-Eval Custom Metrics
- [x] EvaluationDataset
- [x] Batch Evaluation
- [x] Parametrized Tests
- [x] CI/CD Integration

## Troubleshooting

### AWS Credentials Error
```
The security token included in the request is invalid
```
Ensure your AWS credentials are valid and not expired. For temporary credentials, refresh your session token.

### Model Not Found
Ensure you have access to the Qwen models in the `ap-south-1` region through AWS Bedrock.

### Metric Threshold Failures
Adjust the `threshold` parameter in metric definitions. Lower thresholds are stricter for metrics like Bias and Toxicity.

## License

MIT
