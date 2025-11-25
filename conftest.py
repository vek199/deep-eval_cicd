"""
Pytest configuration for DeepEval tests.
"""
import pytest


def pytest_configure(config):
    """Add custom markers for test categorization."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "rag: marks tests as RAG-specific")
    config.addinivalue_line("markers", "safety: marks tests as safety-related")
    config.addinivalue_line("markers", "geval: marks tests as custom G-Eval")


@pytest.fixture(scope="session")
def qwen_judge():
    """Provide a shared QwenJudge instance for the session."""
    from qwen_judge import QwenJudge
    return QwenJudge()

