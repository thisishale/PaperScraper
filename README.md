# PaperScraper

Extracts experimental details (datasets, sample counts, evaluation metrics) from research
paper PDFs using retrieval + an LLM, with citations back to source text and an LLM-judged
evaluation.

## How it works

1. **`main.py`** — for each PDF in `papers/`:
   - Extracts page text via `unstructured`'s `hi_res` strategy (layout detection + OCR)
   - Chunks the text and retrieves relevant chunks using **HyDE**.
   - Calls the OpenAI API with a JSON-schema-enforced response.
2. **`evaluate.py`** — grades `main.py`'s output against hand-labeled files in `groundtruth/`,
   using both a statistical similarity score and an LLM-as-judge verdict per field.

## Setup

### 1. Python dependencies

```bash
pip install openai python-dotenv numpy "unstructured[pdf]"
```

### 2. Tesseract OCR (system binary, not pip-installable)

`hi_res` also requires Tesseract OCR installed separately:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

`main.py` points `pytesseract` directly at
`C:\Program Files\Tesseract-OCR\tesseract.exe` (the default winget install location) rather
than relying on it being resolvable via `PATH`. If your
Tesseract install lives somewhere else, update `_default_tesseract_path` near the top of
`main.py`.

### 3. OpenAI API key

Create `openai.env` in the project directory:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Extract

```bash
python main.py                       # process every PDF in papers/, merge into summary.json
python main.py --overwrite-summary   # same, but rebuild summary.json from scratch
python main.py --num-result 2        # read/write results_2/ 
```

There's no single-paper argument — every run processes everything in `papers/`. `--num-result`
is meant for comparing separate runs (e.g. different prompts/models).

Output per run:
- `results_N/{paper}_results.json` — one file per paper, each field shaped as
  `{"value": ..., "source_chunk_ids": [...], "source_chunks": [...]}`.
- `results_N/summary.json` — flat table (`{"paper": ..., "datasets": ..., ...}`) across all
  papers, merged across runs by paper name rather than overwritten (unless `--overwrite-summary`).

### Evaluate

Requires hand-labeled ground truth in `groundtruth/{paper}.json`, one file per paper, shaped
as flat strings per field: `{"datasets": "...", "train_sample_count": "...", ...}`.

```bash
python evaluate.py --num-result 5    # must match the results_N/ dir you're evaluating
```

Output:
- `eval_reports_N/{paper}.json` — per-paper field-by-field comparison (ground truth vs.
  predicted, statistical similarity, LLM judge verdict + reason).
- `results_N/eval_summary_N.json` — aggregate numbers (judge accuracy, verdict counts, average
  similarity).

## Notes

- `main.py` sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` and forces single-threaded,
  deterministic inference (`torch.set_num_threads(1)`... currently set to `4`, see the comment
  in `main.py`) 
- If a field isn't explicitly reported in a paper, its value is `"not reported"`.
