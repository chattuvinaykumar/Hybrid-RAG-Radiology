def build_prompt(query, context):
    prompt = f"""
You are an expert radiologist.

STRICT RULES:
- Use ONLY the provided context
- Do NOT hallucinate or add unsupported findings
- Keep output clinically accurate and structured

Context:
{context}

Patient Findings:
{query}

Output format:

FINDINGS:
- ...

IMPRESSION:
- ...

CONFIDENCE:
High / Medium / Low

UNSUPPORTED:
- List anything not supported by context (or write 'None')
"""
    return prompt
