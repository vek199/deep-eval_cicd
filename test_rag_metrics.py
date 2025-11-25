"""
DeepEval RAG Metrics Test Suite

This test file focuses on Retrieval-Augmented Generation (RAG) specific metrics:
- Contextual Precision: Measures if relevant context is ranked higher
- Contextual Recall: Measures if all relevant info from context is captured
- Contextual Relevancy: Measures if retrieved context is relevant to query

These metrics are essential for evaluating RAG pipelines.
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from bedrock_qwen import call_qwen_with_context
from qwen_judge import QwenJudge

# Initialize the judge model
qwen_judge = QwenJudge()

# ==============================================================================
# RAG METRIC DEFINITIONS
# ==============================================================================

contextual_precision_metric = ContextualPrecisionMetric(
    model=qwen_judge,
    threshold=0.7,
)

contextual_recall_metric = ContextualRecallMetric(
    model=qwen_judge,
    threshold=0.7,
)

contextual_relevancy_metric = ContextualRelevancyMetric(
    model=qwen_judge,
    threshold=0.7,
)


# ==============================================================================
# TEST CASES - CONTEXTUAL PRECISION
# ==============================================================================

class TestContextualPrecision:
    """
    Tests for Contextual Precision metric.
    Measures whether the most relevant context nodes are ranked higher.
    """
    
    def test_well_ordered_context(self):
        """Test with context where relevant info comes first."""
        # Relevant context first, less relevant later
        retrieval_context = [
            "Albert Einstein developed the theory of relativity in 1905.",
            "Einstein was born in Germany in 1879.",
            "The weather in Germany is temperate.",  # Less relevant
        ]
        test_input = "When did Einstein develop the theory of relativity?"
        expected_output = "Einstein developed the theory of relativity in 1905."
        actual_output = call_qwen_with_context(test_input, retrieval_context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        assert_test(test_case, metrics=[contextual_precision_metric])


# ==============================================================================
# TEST CASES - CONTEXTUAL RECALL
# ==============================================================================

class TestContextualRecall:
    """
    Tests for Contextual Recall metric.
    Measures whether the output captures all relevant information from context.
    """
    
    def test_complete_answer_from_context(self):
        """Test that answer captures key information from context."""
        retrieval_context = [
            "The Great Wall of China is over 13,000 miles long.",
            "Construction began in the 7th century BC.",
            "It was built to protect against invasions from the north.",
        ]
        test_input = "Tell me about the Great Wall of China."
        expected_output = (
            "The Great Wall of China is over 13,000 miles long, "
            "construction began in the 7th century BC, and it was "
            "built to protect against northern invasions."
        )
        actual_output = call_qwen_with_context(test_input, retrieval_context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        assert_test(test_case, metrics=[contextual_recall_metric])


# ==============================================================================
# TEST CASES - CONTEXTUAL RELEVANCY  
# ==============================================================================

class TestContextualRelevancy:
    """
    Tests for Contextual Relevancy metric.
    Measures whether retrieved context is relevant to the input query.
    """
    
    def test_relevant_context(self):
        """Test with highly relevant context."""
        retrieval_context = [
            "Python is a high-level programming language.",
            "Python was created by Guido van Rossum in 1991.",
            "Python emphasizes code readability and simplicity.",
        ]
        test_input = "What is Python programming language?"
        actual_output = call_qwen_with_context(test_input, retrieval_context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        assert_test(test_case, metrics=[contextual_relevancy_metric])


# ==============================================================================
# COMBINED RAG TESTS
# ==============================================================================

class TestRAGPipeline:
    """End-to-end RAG pipeline tests with multiple metrics."""
    
    def test_complete_rag_evaluation(self):
        """Test a RAG response with all contextual metrics."""
        retrieval_context = [
            "Tesla, Inc. was founded in 2003 by Martin Eberhard and Marc Tarpenning.",
            "Elon Musk joined as chairman in 2004 and became CEO in 2008.",
            "Tesla is known for electric vehicles like Model S, Model 3, Model X, and Model Y.",
            "The company is headquartered in Austin, Texas.",
        ]
        test_input = "Who founded Tesla and what products do they make?"
        expected_output = (
            "Tesla was founded by Martin Eberhard and Marc Tarpenning in 2003. "
            "Elon Musk joined later as chairman. Tesla makes electric vehicles "
            "including Model S, Model 3, Model X, and Model Y."
        )
        actual_output = call_qwen_with_context(test_input, retrieval_context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        
        # Run all RAG metrics
        assert_test(
            test_case,
            metrics=[
                contextual_precision_metric,
                contextual_recall_metric,
                contextual_relevancy_metric,
            ]
        )
    
    def test_technical_documentation_rag(self):
        """Test RAG with technical documentation context."""
        retrieval_context = [
            "AWS Lambda supports Python, Node.js, Java, Go, and .NET runtimes.",
            "Lambda functions can be triggered by API Gateway, S3, DynamoDB, and other AWS services.",
            "Maximum execution time for Lambda is 15 minutes.",
            "Lambda pricing is based on number of requests and compute time.",
        ]
        test_input = "What programming languages does AWS Lambda support and what triggers it?"
        expected_output = (
            "AWS Lambda supports Python, Node.js, Java, Go, and .NET. "
            "It can be triggered by API Gateway, S3, DynamoDB, and other AWS services."
        )
        actual_output = call_qwen_with_context(test_input, retrieval_context)
        
        test_case = LLMTestCase(
            input=test_input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        
        assert_test(
            test_case,
            metrics=[contextual_relevancy_metric, contextual_recall_metric]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

