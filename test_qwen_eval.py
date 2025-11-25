"""
DeepEval Test Suite for Qwen Models on AWS Bedrock

This test file demonstrates various DeepEval metrics and testing patterns:
- Answer Relevancy: Tests if output is relevant to input
- Faithfulness: Tests if output is faithful to provided context
- Hallucination: Tests if model hallucinates facts

Models:
- Application Model: Qwen3-32B (qwen.qwen3-32b-v1:0)
- Judge Model: Qwen3-235B (qwen.qwen3-235b-a22b-2507-v1:0)
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from bedrock_qwen import call_qwen, call_qwen_with_context
from qwen_judge import QwenJudge

# Initialize the judge model (Qwen3-235B)
qwen_judge = QwenJudge()


# ==============================================================================
# METRIC DEFINITIONS
# ==============================================================================

answer_relevancy_metric = AnswerRelevancyMetric(
    model=qwen_judge,
    threshold=0.7,
)

faithfulness_metric = FaithfulnessMetric(
    model=qwen_judge,
    threshold=0.7,
)

hallucination_metric = HallucinationMetric(
    model=qwen_judge,
    threshold=0.5,  # Lower threshold = stricter (less hallucination allowed)
)


# ==============================================================================
# TEST CASES - ANSWER RELEVANCY
# ==============================================================================

class TestAnswerRelevancy:
    """Tests for Answer Relevancy metric - checks if output answers the input."""
    
    def test_simple_factual_question(self):
        """Test relevancy for simple factual question."""
        test_input = "What is the capital of France?"
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[answer_relevancy_metric])
    
    def test_explanation_question(self):
        """Test relevancy for explanation-type question."""
        test_input = "Explain how photosynthesis works in simple terms."
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[answer_relevancy_metric])
    
    def test_coding_question(self):
        """Test relevancy for coding question."""
        test_input = "Write a Python function to calculate factorial of a number."
        actual_output = call_qwen(test_input)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
        )
        assert_test(test_case, metrics=[answer_relevancy_metric])


# ==============================================================================
# TEST CASES - FAITHFULNESS (RAG)
# ==============================================================================

class TestFaithfulness:
    """Tests for Faithfulness metric - checks if output is faithful to context."""
    
    def test_faithful_to_context(self):
        """Test that model response is faithful to provided context."""
        context = [
            "The Eiffel Tower is located in Paris, France.",
            "It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.",
            "The tower is 330 meters tall and was designed by Gustave Eiffel."
        ]
        test_input = "When was the Eiffel Tower built and how tall is it?"
        actual_output = call_qwen_with_context(test_input, context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            retrieval_context=context,
        )
        assert_test(test_case, metrics=[faithfulness_metric])
    
    def test_company_context(self):
        """Test faithfulness with company-specific context."""
        context = [
            "Acme Corp was founded in 2010 by John Smith.",
            "The company is headquartered in San Francisco.",
            "Acme Corp has 500 employees and revenue of $50 million."
        ]
        test_input = "Who founded Acme Corp and where is it located?"
        actual_output = call_qwen_with_context(test_input, context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            retrieval_context=context,
        )
        assert_test(test_case, metrics=[faithfulness_metric])


# ==============================================================================
# TEST CASES - HALLUCINATION
# ==============================================================================

class TestHallucination:
    """Tests for Hallucination metric - checks if model makes up facts."""
    
    def test_no_hallucination_with_context(self):
        """Test that model doesn't hallucinate when given context."""
        context = [
            "Python was created by Guido van Rossum.",
            "Python was first released in 1991.",
            "Python is known for its simple syntax and readability."
        ]
        test_input = "Who created Python and when?"
        actual_output = call_qwen_with_context(test_input, context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            context=context,
        )
        assert_test(test_case, metrics=[hallucination_metric])


# ==============================================================================
# COMBINED METRIC TESTS
# ==============================================================================

class TestCombinedMetrics:
    """Tests that evaluate multiple metrics simultaneously."""
    
    def test_rag_response_quality(self):
        """Test a RAG response with multiple quality metrics."""
        context = [
            "Machine learning is a subset of artificial intelligence.",
            "ML algorithms learn patterns from data without explicit programming.",
            "Common ML techniques include supervised, unsupervised, and reinforcement learning."
        ]
        test_input = "What is machine learning and what are its main techniques?"
        actual_output = call_qwen_with_context(test_input, context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            retrieval_context=context,
            context=context,
        )
        
        # Test with multiple metrics
        assert_test(
            test_case, 
            metrics=[answer_relevancy_metric, faithfulness_metric]
        )


# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================

@pytest.mark.parametrize("question,expected_topic", [
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is H2O commonly known as?", "water"),
    ("What is the speed of light approximately?", "speed"),
])
def test_factual_questions_parametrized(question: str, expected_topic: str):
    """Parametrized test for multiple factual questions."""
    actual_output = call_qwen(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
    )
    assert_test(test_case, metrics=[answer_relevancy_metric])
    
    # Additional assertion - check topic is mentioned
    assert expected_topic.lower() in actual_output.lower(), \
        f"Expected '{expected_topic}' in output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
