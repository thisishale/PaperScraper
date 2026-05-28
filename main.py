from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import json
import sys
import os

load_dotenv("openai.env")      # reads the .env file
client = OpenAI()  # automatically uses OPENAI_API_KEY

paper_name = sys.argv[1] if len(sys.argv) > 1 else "paper"
pdf_path = os.path.join("papers", paper_name + ".pdf")
reader = PdfReader(pdf_path)
paper_text = " ".join(page.extract_text() or "" for page in reader.pages)

keywords = ["dataset", "datasets", "train", "test", "split", "samples", "metric", "evaluation", "report"]

# Split into sentence-level chunks and keep only those containing a keyword
chunks = [s.strip() for s in paper_text.split(".") if s.strip()]
relevant_chunks = [
    chunk for chunk in chunks
    if any(word.lower() in chunk.lower() for word in keywords)
]

retrieved_text = ". ".join(relevant_chunks)

response = client.responses.create(
    model="gpt-5.4-nano",
    instructions="""
You extract experimental details from research papers.
Return only valid JSON.
Do not add explanations.
""",
    input=f"""
Extract the datasets, train/test sample counts, and evaluation metrics from the text below.

If train/test sample counts are not explicitly reported, write "not reported".

Do not report additional explanations.

Json dict keys should only be: datasets, train_sample_count, test_sample_count, metrics

Text:
{retrieved_text}
"""
)

result = json.loads(response.output_text)
print(result)

os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", paper_name + "_results.json")
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {output_path}")