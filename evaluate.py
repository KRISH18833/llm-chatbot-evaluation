import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from collections import Counter
from dotenv import load_dotenv

CHATBOT_DIR = Path(r"d:\Valentius Kryptix\valentius-llm-chatbot")
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))

load_dotenv(CHATBOT_DIR / ".env")
load_dotenv(Path(__file__).parent / ".env")

from main import app
from fastapi.testclient import TestClient
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")


def query_chatbot(client: TestClient, question: str) -> tuple[str, float, bool]:
    start = time.perf_counter()
    try:
        resp = client.post(
            "/chat",
            json={"message": question, "conversation_id": None},
        )
        latency = round(time.perf_counter() - start, 3)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("reply", ""), latency, True
        return f"HTTP {resp.status_code}: {resp.text}", latency, False
    except Exception as exc:
        latency = round(time.perf_counter() - start, 3)
        return f"{type(exc).__name__}: {exc}", latency, False


def judge_answer(client: genai.Client, model: str, item: dict, answer: str) -> tuple[dict, str]:
    prompt = f"""You are an impartial judge evaluating an LLM chatbot response.

Category: {item.get('category')}
Question: {item.get('question')}
Reference / Success Criterion: {item.get('reference')}
Chatbot Answer: {answer}

Evaluate on a 1-5 integer scale:
- correctness: Matches reference or meets success criterion (strictly evaluate criteria for edge cases).
- relevance: Directly answers the user prompt.
- clarity: Concise, clear, no filler.

Return JSON only:
{{"correctness": int, "relevance": int, "clarity": int, "reason": "one line"}}"""

    candidate_models = [model]
    if model != "gemini-3.5-flash":
        candidate_models.append("gemini-3.5-flash")

    last_error = ""
    for candidate in candidate_models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                text = response.text or ""
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    cleaned = cleaned[start_idx : end_idx + 1]

                parsed = json.loads(cleaned)
                scores = {
                    "correctness": max(1, min(5, int(parsed["correctness"]))),
                    "relevance": max(1, min(5, int(parsed["relevance"]))),
                    "clarity": max(1, min(5, int(parsed["clarity"]))),
                }
                reason = str(parsed.get("reason", "")).strip().replace("\n", " ")
                return scores, reason
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    if len(candidate_models) > 1 and candidate != candidate_models[-1]:
                        break
                if attempt < 2:
                    time.sleep(2 ** attempt)

    return {"correctness": 1, "relevance": 1, "clarity": 1}, f"Judge failed: {last_error}"


def print_summary(results: list[dict]):
    if not results:
        print("No evaluation results to summarize.")
        return

    metrics = ["correctness", "relevance", "clarity"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\n--- Metric Aggregates ---")
    print(f"{'Metric':<15} {'Average':<10} {'Min':<6} {'Max':<6}")
    print("-" * 37)
    for m in metrics:
        values = [r["scores"][m] for r in results]
        avg = sum(values) / len(values)
        print(f"{m.capitalize():<15} {avg:<10.2f} {min(values):<6} {max(values):<6}")

    print("\n--- Score Distribution (Counts 1-5) ---")
    print(f"{'Metric':<15} {'1':<6} {'2':<6} {'3':<6} {'4':<6} {'5':<6}")
    print("-" * 45)
    for m in metrics:
        counts = Counter(r["scores"][m] for r in results)
        print(
            f"{m.capitalize():<15} "
            f"{counts[1]:<6} {counts[2]:<6} {counts[3]:<6} {counts[4]:<6} {counts[5]:<6}"
        )

    print("\n--- Category Averages ---")
    categories = sorted(set(r["category"] for r in results))
    print(f"{'Category':<10} {'Count':<8} {'Correct':<10} {'Relevance':<10} {'Clarity':<10} {'Overall Mean':<12}")
    print("-" * 60)
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        c_avg = sum(r["scores"]["correctness"] for r in cat_items) / len(cat_items)
        r_avg = sum(r["scores"]["relevance"] for r in cat_items) / len(cat_items)
        cl_avg = sum(r["scores"]["clarity"] for r in cat_items) / len(cat_items)
        overall = (c_avg + r_avg + cl_avg) / 3.0
        print(f"{cat:<10} {len(cat_items):<8} {c_avg:<10.2f} {r_avg:<10.2f} {cl_avg:<10.2f} {overall:<12.2f}")

    print("\n--- 3 Lowest-Scoring Items ---")
    sorted_items = sorted(
        results,
        key=lambda r: (sum(r["scores"].values()) / 3.0, r["id"]),
    )
    for item in sorted_items[:3]:
        mean_score = sum(item["scores"].values()) / 3.0
        print(f"\n[ID {item['id']}] Category: {item['category']} | Mean Score: {mean_score:.2f}")
        print(f"  Scores: Correctness={item['scores']['correctness']}, Relevance={item['scores']['relevance']}, Clarity={item['scores']['clarity']}")
        print(f"  Question: {item['question']}")
        print(f"  Reason: {item['reason']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM chatbot with test_set.json.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N items")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)

    test_set_path = Path(__file__).parent / "test_set.json"
    if not test_set_path.exists():
        print(f"Error: {test_set_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(test_set_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]

    test_client = TestClient(app)
    judge_client = genai.Client(api_key=GEMINI_API_KEY)
    results = []

    print(f"Starting evaluation of {len(items)} items using in-process TestClient and judge '{JUDGE_MODEL}'...")

    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] Item ID {item['id']} ({item['category']})... ", end="", flush=True)

        answer, latency, success = query_chatbot(test_client, item["question"])

        if not success:
            scores = {"correctness": 1, "relevance": 1, "clarity": 1}
            reason = f"Chatbot call failed: {answer}"
        else:
            scores, reason = judge_answer(judge_client, JUDGE_MODEL, item, answer)
            time.sleep(1)

        result_entry = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "reference": item["reference"],
            "answer": answer,
            "scores": scores,
            "reason": reason,
            "latency": latency,
        }
        results.append(result_entry)
        mean_score = sum(scores.values()) / 3.0
        print(f"Done (latency: {latency}s, mean score: {mean_score:.2f})")

    results_path = Path(__file__).parent / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed evaluation to {results_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
