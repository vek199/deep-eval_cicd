"""
DeepEval Dataset-Based Evaluation Test Suite

This test file demonstrates dataset-based evaluation features:
- EvaluationDataset: Managing collections of test cases
- evaluate(): Batch evaluation of datasets
- Metrics aggregation and reporting

Dataset-based evaluation is useful for:
- Running comprehensive test suites
- Benchmarking model performance
- Regression testing
"""
import pytest
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from bedrock_qwen import call_qwen, call_qwen_with_context
from qwen_judge import QwenJudge

# Initialize the judge model
qwen_judge = QwenJudge()

# Define metrics for dataset evaluation
answer_relevancy = AnswerRelevancyMetric(model=qwen_judge, threshold=0.7)
faithfulness = FaithfulnessMetric(model=qwen_judge, threshold=0.7)


# ==============================================================================
# DATASET DEFINITIONS
# ==============================================================================

def create_factual_qa_dataset() -> EvaluationDataset:
    """
    Create a dataset of factual Q&A test cases.
    
    Returns:
        EvaluationDataset with factual Q&A test cases
    """
    test_cases = []
    
    # Define Q&A pairs
    qa_pairs = [
        ("What is the chemical symbol for water?", "H2O"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What planet is known as the Red Planet?", "Mars"),
    ]
    
    for question, expected in qa_pairs:
        actual = call_qwen(question)
        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=actual,
                expected_output=expected,
            )
        )
    
    return EvaluationDataset(test_cases=test_cases)


def create_rag_dataset() -> EvaluationDataset:
    """
    Create a dataset of RAG-style test cases with retrieval context.
    
    Returns:
        EvaluationDataset with RAG test cases
    """
    test_cases = []
    
    # Define RAG test data
    rag_data = [
        {
            "question": "When was SpaceX founded?",
            "context": [
                "SpaceX was founded in 2002 by Elon Musk.",
                "The company is headquartered in Hawthorne, California.",
            ],
            "expected": "SpaceX was founded in 2002.",
        },
        {
            "question": "What does NASA stand for?",
            "context": [
                "NASA stands for National Aeronautics and Space Administration.",
                "NASA was established in 1958.",
            ],
            "expected": "NASA stands for National Aeronautics and Space Administration.",
        },
        {
            "question": "What is the main ingredient in bread?",
            "context": [
                "Bread is made primarily from flour.",
                "Other common ingredients include water, yeast, and salt.",
            ],
            "expected": "Flour is the main ingredient in bread.",
        },
    ]
    
    for item in rag_data:
        actual = call_qwen_with_context(item["question"], item["context"])
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=actual,
                expected_output=item["expected"],
                retrieval_context=item["context"],
            )
        )
    
    return EvaluationDataset(test_cases=test_cases)


# ==============================================================================
# DATASET EVALUATION TESTS
# ==============================================================================

class TestDatasetEvaluation:
    """Tests using dataset-based evaluation."""
    
    def test_factual_qa_dataset(self):
        """Evaluate factual Q&A dataset with answer relevancy."""
        dataset = create_factual_qa_dataset()
        
        # Run evaluation
        results = evaluate(
            test_cases=dataset.test_cases,
            metrics=[answer_relevancy],
        )
        
        # Check that evaluation completed
        assert results is not None
        
        # Check pass rate
        passed = sum(1 for tc in results.test_results if tc.success)
        total = len(results.test_results)
        pass_rate = passed / total if total > 0 else 0
        
        print(f"Factual QA Pass Rate: {pass_rate:.2%} ({passed}/{total})")
        
        # Assert minimum pass rate
        assert pass_rate >= 0.6, f"Pass rate {pass_rate:.2%} below threshold 60%"
    
    def test_rag_dataset(self):
        """Evaluate RAG dataset with faithfulness metric."""
        dataset = create_rag_dataset()
        
        # Run evaluation
        results = evaluate(
            test_cases=dataset.test_cases,
            metrics=[faithfulness],
        )
        
        # Check that evaluation completed
        assert results is not None
        
        # Check pass rate
        passed = sum(1 for tc in results.test_results if tc.success)
        total = len(results.test_results)
        pass_rate = passed / total if total > 0 else 0
        
        print(f"RAG Faithfulness Pass Rate: {pass_rate:.2%} ({passed}/{total})")
        
        # Assert minimum pass rate
        assert pass_rate >= 0.6, f"Pass rate {pass_rate:.2%} below threshold 60%"


# ==============================================================================
# INLINE DATASET TESTS
# ==============================================================================

class TestInlineDataset:
    """Tests with inline dataset definition."""
    
    def test_coding_questions_dataset(self):
        """Test dataset of coding questions."""
        questions = [
            "What is a variable in programming?",
            "Explain what a function is.",
            "What is a loop in programming?",
        ]
        
        test_cases = []
        for q in questions:
            test_cases.append(
                LLMTestCase(
                    input=q,
                    actual_output=call_qwen(q),
                )
            )
        
        dataset = EvaluationDataset(test_cases=test_cases)
        
        results = evaluate(
            test_cases=dataset.test_cases,
            metrics=[answer_relevancy],
        )
        
        assert results is not None
        print(f"Completed evaluation of {len(results.test_results)} test cases")


# ==============================================================================
# BENCHMARK-STYLE TESTS
# ==============================================================================

class TestBenchmark:
    """Benchmark-style evaluation tests."""
    
    def test_general_knowledge_benchmark(self):
        """Run a general knowledge benchmark."""
        benchmark_questions = [
            {
                "input": "What is the speed of light?",
                "category": "physics",
            },
            {
                "input": "Who painted the Mona Lisa?",
                "category": "art",
            },
            {
                "input": "What is the largest mammal?",
                "category": "biology",
            },
            {
                "input": "What year did World War II end?",
                "category": "history",
            },
        ]
        
        test_cases = []
        for item in benchmark_questions:
            test_cases.append(
                LLMTestCase(
                    input=item["input"],
                    actual_output=call_qwen(item["input"]),
                    additional_metadata={"category": item["category"]},
                )
            )
        
        results = evaluate(
            test_cases=test_cases,
            metrics=[answer_relevancy],
        )
        
        # Aggregate results
        categories = {}
        for i, tc in enumerate(results.test_results):
            cat = benchmark_questions[i]["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if tc.success:
                categories[cat]["passed"] += 1
        
        # Print category breakdown
        print("\nBenchmark Results by Category:")
        for cat, scores in categories.items():
            rate = scores["passed"] / scores["total"]
            print(f"  {cat}: {rate:.2%}")
        
        # Overall pass rate check
        total_passed = sum(c["passed"] for c in categories.values())
        total = sum(c["total"] for c in categories.values())
        overall_rate = total_passed / total if total > 0 else 0
        
        assert overall_rate >= 0.5, f"Benchmark pass rate {overall_rate:.2%} below 50%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

