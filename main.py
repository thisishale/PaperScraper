from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import json
import argparse
import os
import numpy as np

load_dotenv("openai.env")
client = OpenAI()

# --num-result picks which results_{num_result}/ directory this run reads
# from and writes to, so separate runs (e.g. different prompts/models) don't
# clobber each other's output.
# --overwrite-summary starts summary.json over from scratch, using only
# what this run processes. Default (unset) merges into the existing file --
# papers not reprocessed this run keep their old row.
_flag_parser = argparse.ArgumentParser(add_help=False)
_flag_parser.add_argument("--num-result", type=int, default=1)
_flag_parser.add_argument("--overwrite-summary", action="store_true")
_flags = _flag_parser.parse_known_args()[0]
NUM_RESULT = _flags.num_result
OVERWRITE_SUMMARY = _flags.overwrite_summary
RESULTS_DIR = f"results_{NUM_RESULT}"

QUERY = "datasets used, training, number of training samples, number of test samples, evaluation metrics"
def embed(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([item.embedding for item in response.data])

MAX_CHUNK_WORDS = 200  # keeps every chunk safely under the 8192-token embedding limit
MAX_CHUNK_CHARS = 2000  # safety net for text with little/no whitespace to split on

def split_long_chunk(text, max_words=MAX_CHUNK_WORDS, max_chars=MAX_CHUNK_CHARS):
    words = text.split()
    if len(words) <= max_words and len(text) <= max_chars:
        return [text]

    # Text with no periods for a long stretch (tables, reference lists,
    # garbled extraction) would otherwise become one unbounded chunk.
    pieces = (
        [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        if words else [text]
    )

    # Some PDFs extract with broken/missing whitespace (bad font or glyph
    # mapping), so a "word" can itself be huge and word-splitting alone
    # won't bound it. Force a hard character-based split as a fallback.
    bounded = []
    for piece in pieces:
        if len(piece) <= max_chars:
            bounded.append(piece)
        else:
            bounded.extend(piece[i:i + max_chars] for i in range(0, len(piece), max_chars))
    return bounded

def extract(pdf_path):
    reader = PdfReader(pdf_path)
    paper_text = " ".join(page.extract_text() or "" for page in reader.pages)

    sentences = [s.strip() for s in paper_text.split(".") if s.strip()]
    chunks = [sub for s in sentences for sub in split_long_chunk(s)]

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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, paper_name + "_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {output_path}")

    return result


paper_names = [
    os.path.splitext(f)[0]
    for f in os.listdir("papers")
    if f.endswith(".pdf")
]


new_results = []
for name in paper_names:
    result = process(name)
    new_results.append({"paper": name, **{k: v["value"] for k, v in result.items()}})

summary_path = os.path.join(RESULTS_DIR, "summary.json")

if OVERWRITE_SUMMARY or not os.path.exists(summary_path):
    existing_results = []
else:
    with open(summary_path) as f:
        existing_results = json.load(f)

# Merge by paper name: this run's rows overwrite any old row for the same
# paper; papers not reprocessed this run keep their previous row.
merged = {row["paper"]: row for row in existing_results}
for row in new_results:
    merged[row["paper"]] = row
all_results = [merged[name] for name in sorted(merged)]

with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSummary saved to {summary_path}")
