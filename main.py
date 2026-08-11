from dotenv import load_dotenv
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table
import json
import argparse
import os
import re
import random
import numpy as np
import torch

# The hi_res layout-detection model runs on CPU here, and multi-threaded CPU
# matrix ops can sum floats in a different order run-to-run, occasionally
# shifting a detection's confidence score across a threshold -- we saw this
# concretely change whether a table's rows got picked up between identical
# runs. Single-threaded + deterministic algorithms eliminates that variance,
# at the cost of slower inference (this was the noticeable tradeoff we
# measured while testing).

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

load_dotenv("openai.env")
client = OpenAI()

# --num-result picks which results_{num_result}/ directory this run reads
# from and writes to, so separate runs (e.g. different prompts/models) don't
# clobber each other's output.
# --overwrite-summary starts summary.json over from scratch, using only
# what this run processes. Default (unset) merges into the existing file --
# papers not reprocessed this run keep their old row.
_flag_parser = argparse.ArgumentParser(add_help=False)
_flag_parser.add_argument("--num-result", type=int, default=5)
_flag_parser.add_argument("--overwrite-summary", action="store_true")
_flags = _flag_parser.parse_known_args()[0]
NUM_RESULT = _flags.num_result
OVERWRITE_SUMMARY = _flags.overwrite_summary
RESULTS_DIR = f"results_{NUM_RESULT}"

# HyDE: instead of one blended query, use a hand-written hypothetical answer
# passage per field. Answer-shaped text embeds closer to real answer-shaped
# text than a question or keyword list does, and each field gets its own
# focused search instead of competing in one merged ranking.
HYDE_PASSAGES = {
    "datasets": (
        "We evaluate our method on the CIFAR-10, ImageNet, and SQuAD datasets, "
        "as well as several other publicly available benchmarks commonly used "
        "in this research area. The dataset was collected from a public "
        "repository and split into training and test sets."
    ),
    "train_sample_count": (
        "The training set consists of 50,000 examples. We trained the model "
        "on a total of 100,000 labeled training samples."
    ),
    "test_sample_count": (
        "The test set contains 10,000 examples, held out for final "
        "evaluation. We evaluated our model on 5,000 test samples."
    ),
    "metrics": (
        "We report accuracy, precision, recall, F1 score, mean squared error "
        "(MSE), and area under the curve (AUC) as our evaluation metrics. "
        "The model's performance was measured using these standard metrics."
    ),
}
TOP_K_PER_FIELD = 10  # 4 fields x 10 = up to 40 retrieved chunks (fewer after dedupe overlap)

def embed(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([item.embedding for item in response.data])

MAX_CHUNK_WORDS = 200  # keeps every chunk safely under the 8192-token embedding limit
MAX_CHUNK_CHARS = 1000  # safety net for text with little/no whitespace to split on

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

def extract_body_text(pdf_path):
    # hi_res runs an actual layout-detection model (not a text heuristic) to
    # classify each region of the page, so table content can be tagged
    # distinctly from narrative text/captions instead of being guessed at
    # from raw text alone. Table rows are kept (not dropped) -- some papers
    # only state a number in a table with no restatement in prose -- but
    # tagged "[TABLE DATA]" so the extraction prompt can treat names/numbers
    # found there with more caution than the same content found in prose
    # (this is what caused CMU-Pose+++, a competing method, to get
    # misattributed as a dataset when it was only visible in raw table text).
    # Requires Tesseract OCR installed as a system binary (not pip-installable).
    elements = partition_pdf(filename=pdf_path, strategy="hi_res")
    body_elements = [
        f"[TABLE DATA] {e}" if isinstance(e, Table) else str(e) for e in elements
    ]
    return " ".join(body_elements)

def extract(pdf_path):
    paper_text = extract_body_text(pdf_path)

    # Split on periods, but not a period sandwiched between two digits
    # (e.g. "4.5" or "5.1 Training Data") -- that's a decimal/section
    # number, not a sentence boundary.
    sentences = [
        s.strip()
        for s in re.split(r"(?<!\d)\.|\.(?!\d)", paper_text)
        if s.strip()
    ]
    chunks = [sub for s in sentences for sub in split_long_chunk(s)]

    chunk_embeddings = embed(chunks)
    hyde_embeddings = embed(list(HYDE_PASSAGES.values()))

    # Run retrieval once per field's hypothetical passage, then union the
    # results -- a chunk only needs to win one field's search, not score
    # well against a single blended query covering all four at once.
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    normalized_chunks = chunk_embeddings / norms

    retrieved_ids = set()
    for hyde_embedding in hyde_embeddings:
        scores = normalized_chunks @ (hyde_embedding / np.linalg.norm(hyde_embedding))
        top_for_field = np.argsort(scores)[::-1][:TOP_K_PER_FIELD]
        retrieved_ids.update(top_for_field.tolist())

    top_indices = sorted(retrieved_ids)
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

Some chunks are additionally marked "[TABLE DATA]" -- this means the text comes from a raw
table row, extracted without its column headers or full surrounding context. Names next to
numbers in a "[TABLE DATA]" chunk are NOT necessarily datasets -- they are often competing
methods/models being compared in a results table. Only report a name found in "[TABLE DATA]"
as a dataset if non-table text elsewhere confirms it's a dataset; otherwise prefer non-table
text as the source for a field whenever both are available.

Field content -- include ONLY what's described, nothing else:
- "datasets": the name(s) of the dataset(s) used, and nothing else. Do not
  include sample counts, sizes, splits, or other statistics here -- those
  belong in the fields below.
- "train_sample_count": the number of training samples/examples or ratios in case of splits.
- "test_sample_count": the number of test samples/examples only or ratios in case of splits. Validation samples do not count as test.
- "metrics": the name(s) of the evaluation metric(s) used (e.g. accuracy,
  F1, MSE), not the numeric results/scores those metrics produced.

If the paper uses MORE THAN ONE dataset, and train/test counts or metrics differ per dataset,
report a per-dataset breakdown within the same field instead of only reporting one dataset's
numbers -- e.g. "COCO: 80K train, 35K val; Cityscapes: 2975 train". Do not silently drop a
dataset's numbers just because another dataset's are also present in the text.

For each field, report:
- "value": the answer as a string, containing only what's described above. If not explicitly reported, write "not reported".
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
    # trusting the model to reproduce quotes verbatim. The model can
    # hallucinate an id it wasn't actually shown (e.g. guessing a sequential
    # position instead of using the real tag), so drop anything outside the
    # set of chunks that were actually retrieved rather than crashing.
    retrieved_ids = set(top_indices)
    for field in result.values():
        field["source_chunk_ids"] = [i for i in field["source_chunk_ids"] if i in retrieved_ids]
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
