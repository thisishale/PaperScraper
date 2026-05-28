from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import json
import sys
import os
import numpy as np

load_dotenv("openai.env")      # reads the .env file
client = OpenAI()  # automatically uses OPENAI_API_KEY

paper_name = sys.argv[1] if len(sys.argv) > 1 else "paper"
pdf_path = os.path.join("papers", paper_name + ".pdf")
reader = PdfReader(pdf_path)
paper_text = " ".join(page.extract_text() or "" for page in reader.pages)

chunks = [s.strip() for s in paper_text.split(".") if s.strip()]

query = "datasets used, number of training samples, number of test samples, evaluation metrics, experimental results"

def embed(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([item.embedding for item in response.data])

chunk_embeddings = embed(chunks)
query_embedding = embed([query])[0]

# Cosine similarity: dot product of unit vectors
norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
scores = (chunk_embeddings / norms) @ (query_embedding / np.linalg.norm(query_embedding))

top_indices = np.argsort(scores)[::-1][:20]
top_indices_ordered = sorted(top_indices)  # preserve document order
retrieved_text = ". ".join(chunks[i] for i in top_indices_ordered)

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