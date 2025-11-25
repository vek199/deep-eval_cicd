"""
DeepEval G-Eval Custom Metrics Test Suite

G-Eval allows creating custom evaluation metrics using LLM-as-judge.
This enables evaluation of domain-specific criteria not covered by built-in metrics.

This file demonstrates:
- Creating custom G-Eval metrics
- Defining evaluation criteria
- Using custom metrics in tests
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from bedrock_qwen import call_qwen
from qwen_judge import QwenJudge

# Initialize the judge model
qwen_judge = QwenJudge()


# ==============================================================================
# CUSTOM G-EVAL METRICS
# ==============================================================================

# Custom metric: Code Quality
code_quality_metric = GEval(
    name="Code Quality",
    model=qwen_judge,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    criteria=(
        "Evaluate the code quality of the response. Consider: "
        "1. Correctness - Does the code solve the problem? "
        "2. Readability - Is the code easy to understand? "
        "3. Best practices - Does it follow coding conventions? "
        "4. Efficiency - Is the solution reasonably efficient?"
    ),
    threshold=0.7,
)

# Custom metric: Explanation Clarity
explanation_clarity_metric = GEval(
    name="Explanation Clarity",
    model=qwen_judge,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    criteria=(
        "Evaluate how clear and understandable the explanation is. Consider: "
        "1. Structure - Is the explanation well-organized? "
        "2. Simplicity - Are complex concepts broken down? "
        "3. Completeness - Are all key points covered? "
        "4. Examples - Are helpful examples provided when appropriate?"
    ),
    threshold=0.7,
)

# Custom metric: Technical Accuracy
technical_accuracy_metric = GEval(
    name="Technical Accuracy",
    model=qwen_judge,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "Evaluate the technical accuracy of the response compared to expected output. "
        "1. Factual correctness - Are stated facts accurate? "
        "2. Technical terminology - Is terminology used correctly? "
        "3. Alignment - Does output align with expected answer? "
        "4. No misinformation - Are there any incorrect statements?"
    ),
    threshold=0.7,
)

# Custom metric: Response Conciseness
conciseness_metric = GEval(
    name="Conciseness",
    model=qwen_judge,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    criteria=(
        "Evaluate how concise the response is while remaining complete. "
        "1. No unnecessary repetition "
        "2. Direct and to the point "
        "3. Appropriate length for the question "
        "4. No filler content or excessive verbosity"
    ),
    threshold=0.6,
)


# ==============================================================================
# TEST CASES - CODE QUALITY
# ==============================================================================

class TestCodeQuality:
    """Tests using custom Code Quality G-Eval metric."""
    
    def test_python_function_quality(self):
        """Test quality of generated Python function."""
        test_input = "Write a Python function to check if a number is prime."
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[code_quality_metric])
    
    def test_sorting_algorithm(self):
        """Test quality of sorting implementation."""
        test_input = "Write a Python function to sort a list using bubble sort."
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[code_quality_metric])


# ==============================================================================
# TEST CASES - EXPLANATION CLARITY
# ==============================================================================

class TestExplanationClarity:
    """Tests using custom Explanation Clarity G-Eval metric."""
    
    def test_technical_concept_explanation(self):
        """Test clarity of technical concept explanation."""
        test_input = "Explain what a REST API is to a beginner."
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[explanation_clarity_metric])
    
    def test_algorithm_explanation(self):
        """Test clarity of algorithm explanation."""
        test_input = "Explain how binary search works step by step."
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[explanation_clarity_metric])


# ==============================================================================
# TEST CASES - TECHNICAL ACCURACY
# ==============================================================================

class TestTechnicalAccuracy:
    """Tests using custom Technical Accuracy G-Eval metric."""
    
    def test_programming_fact(self):
        """Test technical accuracy of programming facts."""
        test_input = "What are the main features of Python?"
        expected_output = (
            "Python features include: dynamic typing, automatic memory management, "
            "interpreted execution, extensive standard library, support for multiple "
            "paradigms (OOP, functional, procedural), and cross-platform compatibility."
        )
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        assert_test(test_case, metrics=[technical_accuracy_metric])
    
    def test_database_concept(self):
        """Test technical accuracy of database concepts."""
        test_input = "What is the difference between SQL and NoSQL databases?"
        expected_output = (
            "SQL databases are relational, use structured schemas, and support ACID. "
            "NoSQL databases are non-relational, flexible schema, and designed for "
            "horizontal scaling. SQL uses tables, NoSQL uses documents/key-value/graphs."
        )
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        assert_test(test_case, metrics=[technical_accuracy_metric])


# ==============================================================================
# TEST CASES - CONCISENESS
# ==============================================================================

class TestConciseness:
    """Tests using custom Conciseness G-Eval metric."""
    
    def test_short_answer(self):
        """Test conciseness for short answer question."""
        test_input = "What is the time complexity of binary search?"
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[conciseness_metric])


# ==============================================================================
# COMBINED CUSTOM METRICS
# ==============================================================================

class TestCombinedCustomMetrics:
    """Tests combining multiple custom G-Eval metrics."""
    
    def test_code_with_explanation(self):
        """Test code generation with both quality and clarity metrics."""
        test_input = (
            "Write a Python function to find the longest common subsequence "
            "of two strings, and explain how it works."
        )
        actual_output = call_qwen(test_input, max_tokens=1024)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        
        assert_test(
            test_case,
            metrics=[code_quality_metric, explanation_clarity_metric]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

