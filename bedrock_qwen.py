"""
Bedrock Qwen Client - Main Model Under Test
Uses Qwen3-32B as the application model being evaluated.
"""
import json
import boto3

BEDROCK_REGION = "ap-south-1"
MODEL_ID = "qwen.qwen3-32b-v1:0"  # Main model under test

bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def call_qwen(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Call Qwen3-32B model on AWS Bedrock.
    
    Args:
        prompt: The input prompt to send to the model
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
    
    Returns:
        Model response text
    """
    body = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    payload = json.loads(resp["body"].read())
    return payload["choices"][0]["message"]["content"]


def call_qwen_with_context(prompt: str, context: list[str], max_tokens: int = 512) -> str:
    """
    Call Qwen with retrieval context for RAG-style evaluation.
    
    Args:
        prompt: User question/query
        context: List of retrieved context documents
        max_tokens: Maximum tokens in response
    
    Returns:
        Model response text
    """
    context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context)])
    full_prompt = f"""Use the following context to answer the question.

{context_text}

Question: {prompt}

Answer:"""
    return call_qwen(full_prompt, max_tokens=max_tokens, temperature=0.1)
