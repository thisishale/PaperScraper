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
    retrieved_text = "\n".join(f"[{i}] {chunks[i]}" for i in top_indices)

    # Each field gets its own value plus the id(s) of the chunk(s) it was
    # extracted from, so the answer can be checked against the source text.
    field_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "source_chunk_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["value", "source_chunk_ids"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model="gpt-5.4-nano",
        instructions="""
You extract experimental details from research papers.
""",
        input=f"""
Extract the datasets, train/test sample counts, and evaluation metrics from the text below.

Each chunk of text below is tagged with an id in brackets, e.g. "[3] some sentence.".

For each field, report:
- "value": the answer as a string. If not explicitly reported, write "not reported".
- "source_chunk_ids": the ids of the chunk(s) that support your answer. Use an empty list if the value is "not reported".

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
                        "datasets": field_schema,
                        "train_sample_count": field_schema,
                        "test_sample_count": field_schema,
                        "metrics": field_schema,
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
    result = json.loads(response.output_text)

    # Look up the actual chunk text for each citation ourselves, rather than
    # trusting the model to reproduce quotes verbatim.
    for field in result.values():
        field["source_chunks"] = [chunks[i] for i in field["source_chunk_ids"]]

    return result

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

all_results = []
for name in paper_names:
    result = process(name)
    # summary.json stays a flat, quick-glance table across papers;
    # the full citations live in each paper's own *_results.json.
    all_results.append({"paper": name, **{k: v["value"] for k, v in result.items()}})

summary_path = os.path.join("results", "summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSummary saved to {summary_path}")
