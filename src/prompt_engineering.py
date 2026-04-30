def build_prompt(query, context):
    return f"""
You are an expert radiologist.

Use ONLY the context below.
Do NOT hallucinate.

Context:
{context}

Patient Findings:
{query}

Output format:

FINDINGS:
...

IMPRESSION:
...

CONFIDENCE:
High / Medium / Low

UNSUPPORTED:
None
"""
