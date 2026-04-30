import os
import re
from pathlib import Path

DATA_DIR = "data/mimic-cxr/files"

def extract_sections(text):
    findings = ""
    impression = ""

    f_match = re.search(r"FINDINGS:(.*?)(IMPRESSION:|$)", text, re.S | re.I)
    i_match = re.search(r"IMPRESSION:(.*)", text, re.S | re.I)

    if f_match:
        findings = f_match.group(1).strip()
    if i_match:
        impression = i_match.group(1).strip()

    return findings, impression

def load_reports():
    records = []

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".txt"):
                path = Path(root) / file
                text = path.read_text(errors="ignore")

                findings, impression = extract_sections(text)

                if findings or impression:
                    records.append({
                        "id": path.stem,
                        "findings": findings,
                        "impression": impression
                    })

    return records
