from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import difflib

load_dotenv("openai.env")
client = OpenAI()

FIELDS = ["datasets", "train_sample_count", "test_sample_count", "metrics"]


def normalize(text):
    return " ".join(text.lower().split())


def statistical_similarity(predicted, ground_truth):
    # difflib ratio: 1.0 = identical (after normalizing case/whitespace),
    # 0.0 = nothing in common. Free, deterministic, catches near-identical
    # phrasing but not paraphrases with different wording.
    return difflib.SequenceMatcher(
        None, normalize(predicted), normalize(ground_truth)
    ).ratio()


def llm_judge(field, predicted, ground_truth):
    # Complements the statistical score: an LLM can recognize that
    # "MSE, CMSE" and "Mean Squared Error (MSE), Center Mean Squared
    # Error (CMSE)" mean the same thing, which difflib would score low.
    response = client.responses.create(
        model="gpt-5.4-nano",
        instructions="""
You are grading whether an extracted value matches a ground-truth value for
a research-paper field extraction task.
""",
        input=f"""
Field: {field}
Ground truth: {ground_truth}
Predicted: {predicted}

Grade the predicted value against the ground truth:
- "correct": matches in meaning, even if worded differently
- "partial": partially correct or incomplete (e.g. missing one of two datasets)
- "incorrect": wrong, contradicts ground truth, or hallucinated when ground
  truth says "not reported"

Give a one-sentence reason.
""",
        text={
            "format": {
                "type": "json_schema",
                "name": "judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["correct", "partial", "incorrect"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["verdict", "reason"],
                    "additionalProperties": False,
                },
            }
        },
    )
    return json.loads(response.output_text)


def evaluate_paper(paper_name):
    gt_path = os.path.join("groundtruth", paper_name + "_groundtruth.json")
    pred_path = os.path.join("results", paper_name + "_results.json")

    with open(gt_path) as f:
        ground_truth = json.load(f)
    with open(pred_path) as f:
        predicted = json.load(f)

    field_scores = {}
    for field in FIELDS:
        gt_value = ground_truth[field]["value"]
        pred_value = predicted[field]["value"]

        judgment = llm_judge(field, pred_value, gt_value)

        field_scores[field] = {
            "ground_truth": gt_value,
            "predicted": pred_value,
            "similarity": round(statistical_similarity(pred_value, gt_value), 3),
            "judge_verdict": judgment["verdict"],
            "judge_reason": judgment["reason"],
        }

    return field_scores


def paper_names_with_ground_truth():
    suffix = "_groundtruth.json"
    return [
        f[: -len(suffix)]
        for f in os.listdir("groundtruth")
        if f.endswith(suffix)
    ]


def summarize(report):
    verdict_counts = {"correct": 0, "partial": 0, "incorrect": 0}
    similarities = []
    for field_scores in report.values():
        for score in field_scores.values():
            verdict_counts[score["judge_verdict"]] += 1
            similarities.append(score["similarity"])

    total = sum(verdict_counts.values())
    return {
        "num_papers": len(report),
        "num_fields_graded": total,
        "judge_accuracy": round(verdict_counts["correct"] / total, 3) if total else None,
        "verdict_counts": verdict_counts,
        "avg_statistical_similarity": (
            round(sum(similarities) / len(similarities), 3) if similarities else None
        ),
    }


def main():
    paper_names = paper_names_with_ground_truth()
    if not paper_names:
        print("No ground-truth files found in groundtruth/. Nothing to evaluate.")
        return

    report = {}
    for paper_name in paper_names:
        print(f"Evaluating {paper_name}...")
        report[paper_name] = evaluate_paper(paper_name)

    summary = summarize(report)

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "eval_report.json")
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "per_paper": report}, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
