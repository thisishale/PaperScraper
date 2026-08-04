from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import json
import sys
import os
import numpy as np

load_dotenv("openai.env")
client = OpenAI()

QUERY = "datasets used, number of training samples, number of test samples, evaluation metrics, experimental results"
def embed(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([item.embedding for item in response.data])

def extract(pdf_path):
    reader = PdfReader(pdf_path)
    paper_text = " ".join(page.extract_text() or "" for page in reader.pages)

    chunks = [s.strip() for s in paper_text.split(".") if s.strip()]

    chunk_embeddings = embed(chunks)
    query_embedding = embed([QUERY])[0]

    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    scores = (chunk_embeddings / norms) @ (query_embedding / np.linalg.norm(query_embedding))
    top_indices = sorted(np.argsort(scores)[::-1][:20])
    retrieved_text = ". ".join(chunks[i] for i in top_indices)

    response = client.responses.create(
        model="gpt-5.4-nano",
        instructions="""
You extract experimental details from research papers.
""",
        input=f"""
Extract the datasets, train/test sample counts, and evaluation metrics from the text below.

If train/test sample counts are not explicitly reported, write "not reported".

Text:
{retrieved_text}
""",
        text={
            "format": {
                "type": "json_schema",
                "name": "experiment_details",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "datasets": {"type": "string"},
                        "train_sample_count": {"type": "string"},
                        "test_sample_count": {"type": "string"},
                        "metrics": {"type": "string"},
                    },
                    "required": [
                        "datasets",
                        "train_sample_count",
                        "test_sample_count",
                        "metrics",
                    ],
                    "additionalProperties": False,
                },
            }
        },
    )
    return json.loads(response.output_text)

def process(paper_name):
    pdf_path = os.path.join("papers", paper_name + ".pdf")
    print(f"Processing {pdf_path}...")
    result = extract(pdf_path)

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", paper_name + "_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {output_path}")

    return result

if len(sys.argv) > 1:
    paper_names = [sys.argv[1]]
else:
    paper_names = [
        os.path.splitext(f)[0]
        for f in os.listdir("papers")
        if f.endswith(".pdf")
    ]

def flatten(value):
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return value

all_results = []
for name in paper_names:
    result = process(name)
    all_results.append({"paper": name, **{k: flatten(v) for k, v in result.items()}})

summary_path = os.path.join("results", "summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSummary saved to {summary_path}")
