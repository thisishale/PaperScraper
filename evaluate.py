from dotenv import load_dotenv
from openai import OpenAI
import argparse
import json
import os
import difflib

load_dotenv("openai.env")
client = OpenAI()

FIELDS = ["datasets", "train_sample_count", "test_sample_count", "metrics"]

# Mirrors the field content boundaries in main.py's extraction prompt, so the
# judge grades against the same definition of what belongs in each field --
# not just "does it look similar," but "does it contain the right kind of thing."
FIELD_DESCRIPTIONS = {
    "datasets": (
        "the name(s) of the dataset(s) used, and nothing else -- no sample "
        "counts, sizes, splits, or other statistics"
    ),
    "train_sample_count": "the number of training samples/examples or ratios",
    "test_sample_count": "the number of test samples/examples only or ratios",
    "metrics": (
        "the name(s) of the evaluation metric(s) used, not the numeric "
        "results/scores those metrics produced"
    ),
}


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
This field should contain: {FIELD_DESCRIPTIONS[field]}

Ground truth: {ground_truth}
Predicted: {predicted}

Grade the predicted value against the ground truth:
- "correct": matches in meaning, even if worded differently, and contains only the kind of
  content described above
- "partial": prediction covers part of what ground truth reports, but is missing or gets wrong
  some of it (e.g. ground truth lists two datasets/metrics and prediction only has one), or
  prediction includes extra content that doesn't belong in this field per the description above
- "incorrect": ground truth says "not reported" but prediction has a real value, or ground truth
  has a meaningful value but prediction says "not reported". Nothing matches between ground truth and prediction.

Do NOT grade down for differences in phrasing, naming style, capitalization, or level of detail
if they clearly refer to the same underlying dataset/count/metric. For example, "ImageNet-5k
pretraining dataset" and "ImageNet (5k-class subset)" describe the same dataset and should be
graded "correct", not "partial" -- only use "partial" when something is actually missing but there are some matches, not because the wording differs.

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


def to_text(value):
    # main.py has no schema enforcement, so a field can come back as a
    # string, a list (join it), or missing/null (treat as "not reported").
    if value is None:
        return "not reported"
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


def evaluate_paper(paper_name, results_dir):
    gt_path = os.path.join("groundtruth", paper_name + ".json")
    pred_path = os.path.join(results_dir, paper_name + "_results.json")

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(pred_path, encoding="utf-8") as f:
        predicted = json.load(f)

    field_scores = {}
    for field in FIELDS:
        gt_value = to_text(ground_truth[field])
        pred_value = to_text(predicted[field]["value"])

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
    suffix = ".json"
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
    # --num-result must match the value main.py was run with, so this reads
    # predictions from (and writes eval_summary.json into) the same
    # results_{num_result}/ directory that run produced.
    flag_parser = argparse.ArgumentParser(add_help=False)
    flag_parser.add_argument("--num-result", type=int, default=5)
    num_result = flag_parser.parse_known_args()[0].num_result
    results_dir = f"results_{num_result}"
    eval_reports_dir = f"eval_reports_{num_result}"

    paper_names = paper_names_with_ground_truth()
    if not paper_names:
        print("No ground-truth files found in groundtruth/. Nothing to evaluate.")
        return

    os.makedirs(eval_reports_dir, exist_ok=True)

    report = {}
    for paper_name in paper_names:
        print(f"Evaluating {paper_name}...")
        field_scores = evaluate_paper(paper_name, results_dir)
        report[paper_name] = field_scores

        paper_report_path = os.path.join(eval_reports_dir, paper_name + ".json")
        with open(paper_report_path, "w", encoding="utf-8") as f:
            json.dump(field_scores, f, indent=2)
        print(f"  Saved {paper_report_path}")

    summary = summarize(report)

    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"eval_summary_{num_result}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-paper reports saved to {eval_reports_dir}/, summary saved to {summary_path}")


if __name__ == "__main__":
    main()
