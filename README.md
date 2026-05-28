# PaperScraper

Extracts experimental details (datasets, sample counts, evaluation metrics) from research paper PDFs using keyword-filtered retrieval and an LLM.

## How to Use

### 1. Install dependencies

```bash
pip install openai pypdf python-dotenv
```

### 2. Set up your OpenAI API key

Create a file named `openai.env` in the project directory:

```
OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
python main.py path/to/paper.pdf
```

If no path is given, it looks for `paper.pdf` in the current directory by default.

### Output

Prints a JSON object with the following keys:

```json
{
  "datasets": "...",
  "train_sample_count": "...",
  "test_sample_count": "...",
  "metrics": "..."
}
```

If a field is not explicitly reported in the paper, its value will be `"not reported"`.
