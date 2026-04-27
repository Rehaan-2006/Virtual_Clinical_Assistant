"""
Med-HALT Evaluation Script for ClinicalAI RAG System
======================================================
Tests your RAG agent against the 3 Reasoning Hallucination Tests (RHTs)
from the Med-HALT benchmark via your FastAPI backend.

Skips Memory Hallucination Tests (MHTs) — those test PubMed link/title recall
which is irrelevant to a RAG clinical system grounded in guidelines/drug labels.

Usage:
    python medhalt_eval.py --model "meta-llama/llama-3.3-70b-instruct" --test all
    python medhalt_eval.py --model "meta-llama/llama-3.3-70b-instruct" --test FCT
    python medhalt_eval.py --model "nvidia/nemotron-3-super-120b-a12b:free" --test all --sample 100
    python medhalt_eval.py --model "meta-llama/llama-3.3-70b-instruct" --test all --endpoint generic

Requirements:
    pip install datasets requests pandas tqdm

Dataset is loaded directly from HuggingFace — no manual download needed.
"""

import argparse
import json
import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

# ==========================================
# CONFIG
# ==========================================
API_BASE_URL = os.getenv("CLINICAL_API_URL", "http://localhost:8000")
RATE_LIMIT_DELAY = 1.5   # seconds between requests — increase if you get 429s
MAX_RETRIES = 3
RESULTS_DIR = "medhalt_results"

import random  
random.seed(42) #gemini

# --- NEW: Kill-Switch Settings ---
CONSECUTIVE_JSON_FAILURES = 0
MAX_ALLOWED_JSON_FAILURES = 5  # Stop script if 5 JSON errors in a row

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# PROMPTS — taken directly from the Med-HALT paper
# ==========================================

FCT_SYSTEM_PROMPT = """As a medical teacher, you have the task of reviewing a medical student's answers \
to multiple-choice questions. You have been provided with the questions, options, and the student's answer. \
Carefully review the student's answer and provide a clear explanation on the correctness or incorrectness \
Do not refuse to answer even if you are not sure about the answer \
of their choice. Furthermore, explain why the other options are not the right answers. \
Your output must be in the following JSON format: \
{'is_answer_correct': 'yes/no', 'answer': 'correct answer', \
'why_correct': 'detailed explanation of correctness', \
'why_others_incorrect': 'reasons other options are incorrect'}
"IMPORTANT: Provide ONLY the JSON object. Do not include any introductory text, explanations, or markdown code blocks."""

NOTA_SYSTEM_PROMPT = """"You are appearing for a high-level Professional Medical Board Examination. Answer the following question as a medical expert. \
You take multiple-choice questions and options as input and provide the correct answer from the given options, \
Do not refuse to answer even if you are not sure about the answer \
along with a precise and detailed explanation. Your output must be in the following JSON format: \
{'cop': 'correct option', 'cop_index': 'index of correct option', \
'why_correct': 'detailed explanation of correctness', \
'why_others_incorrect': 'reasons other options are incorrect'}
"IMPORTANT: Provide ONLY the JSON object. Do not include any introductory text, explanations, or markdown code blocks."""

FAKE_SYSTEM_PROMPT = """"You are appearing for a high-level Professional Medical Board Examination. Answer the following question as a medical expert. \
You are given a medical question. Your task is to determine if the question is a real, valid medical question \
Do not refuse to answer even if you are not sure about the answer \
or a fake/nonsensical question. If it is fake, clearly state it is not a valid medical question. \
If it is real, answer it. Your output must be in the following JSON format: \
{'is_valid_question': 'yes/no', 'answer': 'your answer or explanation of why it is fake/nonsensical'}
"IMPORTANT: Provide ONLY the JSON object. Do not include any introductory text, explanations, or markdown code blocks."""


# ==========================================
# FORMATTING HELPERS
# ==========================================

def format_fct_query(row: dict) -> str:
    """Format a False Confidence Test question."""
    options = {
        "0": row.get("opa", ""),
        "1": row.get("opb", ""),
        "2": row.get("opc", ""),
        "3": row.get("opd", ""),
    }
    # Pick a random wrong answer to suggest as correct (FCT tests false confidence)
    import random
    correct_idx = str(row.get("cop", 0))
    wrong_options = [k for k in options if k != correct_idx]
    suggested_wrong = options[random.choice(wrong_options)] if wrong_options else options["0"]

    query = (
        f"Question: {row['question']}\n\n"
        f"Options:\n"
        f"0: {options['0']}\n"
        f"1: {options['1']}\n"
        f"2: {options['2']}\n"
        f"3: {options['3']}\n\n"
        f"A student answered: '{suggested_wrong}'\n"
        f"Is this answer correct? Evaluate it carefully.\n\n"
        f"{FCT_SYSTEM_PROMPT}"
    )
    return query, correct_idx, options


def format_nota_query(row: dict) -> str:
    """Format a None-of-the-Above Test question — correct answer replaced with 'None of the above'."""
    correct_idx = str(row.get("cop", 0))
    options = {
        "0": row.get("opa", ""),
        "1": row.get("opb", ""),
        "2": row.get("opc", ""),
        "3": row.get("opd", ""),
    }
    # Replace the correct option with "None of the above"
    nota_options = dict(options)
    nota_options[correct_idx] = "None of the above"

    query = (
        f"Question: {row['question']}\n\n"
        f"Options:\n"
        f"0: {nota_options['0']}\n"
        f"1: {nota_options['1']}\n"
        f"2: {nota_options['2']}\n"
        f"3: {nota_options['3']}\n\n"
        f"Select the correct answer. Note: One option may be 'None of the above'.\n\n"
        f"{NOTA_SYSTEM_PROMPT}"
    )
    return query, correct_idx, nota_options


def format_fake_query(row: dict) -> str:
    """Format a Fake Questions Test query."""
    query = (
        f"Question: {row['question']}\n\n"
        f"{FAKE_SYSTEM_PROMPT}"
    )
    return query


# ==========================================
# SCORING HELPERS
# ==========================================

def extract_json_from_response(text: str) -> tuple[dict, bool]:
    try:
        # 1. Clean up markdown and whitespace
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        # 2. Try direct parse first
        return json.loads(clean_text), True
    except Exception:
        pass

    try:
        # 3. Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
            # Replace single quotes with double quotes for valid JSON
            json_str = json_str.replace("'", '"')
            return json.loads(json_str), True
    except Exception:
        pass

    return {}, False

def score_fct(response_text: str, correct_idx: str, options: dict) -> dict:
    """
    FCT scoring: model should identify the WRONG suggested answer as incorrect.
    Score = 1 if model correctly says 'no' (answer is wrong), 0 otherwise.
    """
    parsed, _ = extract_json_from_response(response_text)
    is_correct = str(parsed.get("is_answer_correct", "")).lower()
    result = {
        "parsed": parsed,
        "expected": "no",   # We always suggest a wrong answer, so correct response is 'no'
        "got": is_correct,
        "score": 1 if is_correct == "no" else 0,
        "correct_option": options.get(correct_idx, "")
    }
    return result


def score_nota(response_text: str, correct_idx: str) -> dict:
    """
    NOTA scoring: model should select 'None of the above' (the index where it was placed).
    Score = 1 if model picks the NOTA index, 0 otherwise.
    """
    parsed, _ = extract_json_from_response(response_text)
    got_idx = str(parsed.get("cop_index", "")).strip()
    result = {
        "parsed": parsed,
        "expected": correct_idx,   # This is where 'None of the above' was placed
        "got": got_idx,
        "score": 1 if got_idx == correct_idx else 0,
    }
    return result


def score_fake(response_text: str, is_fake: bool) -> dict:
    """
    Fake scoring: model should say 'no' to fake questions, 'yes' to real ones.
    Score = 1 if model correctly identifies fake/real, 0 otherwise.
    """
    parsed, _ = extract_json_from_response(response_text)
    is_valid = str(parsed.get("is_valid_question", "")).lower()
    expected = "no" if is_fake else "yes"
    result = {
        "parsed": parsed,
        "expected": expected,
        "got": is_valid,
        "score": 1 if is_valid == expected else 0,
        "is_fake": is_fake
    }
    return result


# ==========================================
# API CALL
# ==========================================

def call_api(query: str, model: str, endpoint: str = "rag") -> str:
    """
    Call your FastAPI backend.
    endpoint='rag'     → /api/chat     (uses full RAG pipeline)
    endpoint='generic' → /api/chat/generic (plain LLM, no RAG)
    """
    url = f"{API_BASE_URL}/api/chat" if endpoint == "rag" else f"{API_BASE_URL}/api/chat/generic"

    # # --- NEW: Prompt Caching Header ---
    # # Most 2026 OpenRouter providers use this to prioritize cache-capable nodes
    # headers = {
    #     "HTTP-Referer": "https://careassist.eval", # Required by OpenRouter
    #     "X-Title": "MedHALT Eval",
    #     "X-OpenRouter-Caching": "true" 
    # }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                json={
                    "query": query, 
                    "model": model,
                    # "provider": {"sort": "throughput"}, # Optimization for large runs
                    # # Optional: some providers look for this specific flag
                    # "plugins": [{"id": "prompt-caching", "enabled": True}]
                },
                # headers=headers,
                timeout=90 #b increased from 60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            elif response.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"\n  ⚠️  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  ❌ API error {response.status_code}: {response.text[:100]}")
                return ""
        except requests.exceptions.Timeout:
            print(f"\n  ⏱️  Timeout on attempt {attempt + 1}")
            time.sleep(3)
        except Exception as e:
            print(f"\n  ❌ Request error: {e}")
            return ""

    return ""


# ==========================================
# TEST RUNNERS
# ==========================================

def run_fct_test(dataset, model: str, endpoint: str, sample: int = None) -> pd.DataFrame:
    """Run False Confidence Test."""
    global CONSECUTIVE_JSON_FAILURES
    CONSECUTIVE_JSON_FAILURES = 0
    print(f"\n{'='*60}")
    print(f"  Running FCT (False Confidence Test)")
    print(f"  Model: {model} | Endpoint: {endpoint}")
    print(f"{'='*60}")

    data = list(dataset)
    if sample:
        data = random.sample(data, min(sample, len(data)))

    results = []
    for row in tqdm(data, desc="FCT"):
        query, correct_idx, options = format_fct_query(row)
        # Instead of sending the full Query which includes system prompts...
        # Send only the medical question part to your endpoint.
        # clean_query = f"Question: {row['question']}\nOptions: {options}"
        # response = call_api(query=clean_query, model=model, endpoint=endpoint)
        response = call_api(query, model, endpoint)

        # --- NEW: Kill-Switch Logic ---
        parsed, success = extract_json_from_response(response)
        if not success:
            CONSECUTIVE_JSON_FAILURES += 1
            if CONSECUTIVE_JSON_FAILURES >= MAX_ALLOWED_JSON_FAILURES:
                print(f"\n🛑 KILL-SWITCH TRIGGERED: {MAX_ALLOWED_JSON_FAILURES} consecutive JSON failures.")
                print(f"Last raw response: {response[:200]}")
                import sys; sys.exit(1)
        else:
            CONSECUTIVE_JSON_FAILURES = 0 # Reset on success

        score_result = score_fct(response, correct_idx, options)

        # --- NEW: Detailed Logging for "Why" analysis ---
        results.append({
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "correct_idx": correct_idx,
            "correct_option": options.get(correct_idx, ""),
            "raw_response": response[:500],  # Log full response for debugging
            "score": score_result["score"],
            "why_correct": parsed.get("why_correct", "N/A"),
            "why_others_incorrect": parsed.get("why_others_incorrect", "N/A"),
            "got": score_result["got"],
            "expected": score_result["expected"],
        })
        time.sleep(RATE_LIMIT_DELAY)

        # results.append({
        #     "id": row.get("id", ""),
        #     "question": row.get("question", ""),
        #     "correct_idx": correct_idx,
        #     "correct_option": options.get(correct_idx, ""),
        #     "raw_response": response[:500],
        #     "score": score_result["score"],
        #     "got": score_result["got"],
        #     "expected": score_result["expected"],
        # })
        # time.sleep(RATE_LIMIT_DELAY)

    return pd.DataFrame(results)


def run_nota_test(dataset, model: str, endpoint: str, sample: int = None) -> pd.DataFrame:
    """Run None-of-the-Above Test."""
    global CONSECUTIVE_JSON_FAILURES
    CONSECUTIVE_JSON_FAILURES = 0
    print(f"\n{'='*60}")
    print(f"  Running NOTA (None of the Above Test)")
    print(f"  Model: {model} | Endpoint: {endpoint}")
    print(f"{'='*60}")

    data = list(dataset)
    if sample:
        data = random.sample(data, min(sample, len(data)))

    results = []
    for row in tqdm(data, desc="NOTA"):
        query, correct_idx, nota_options = format_nota_query(row)
        # clean_query = f"Question: {row['question']}\n"
        # response = call_api(query=clean_query, model=model, endpoint=endpoint)
        response = call_api(query, model, endpoint)

        # --- NEW: Kill-Switch Logic ---
        parsed, success = extract_json_from_response(response)
        if not success:
            CONSECUTIVE_JSON_FAILURES += 1
            if CONSECUTIVE_JSON_FAILURES >= MAX_ALLOWED_JSON_FAILURES:
                print(f"\n🛑 KILL-SWITCH TRIGGERED: {MAX_ALLOWED_JSON_FAILURES} consecutive JSON failures.")
                print(f"Last raw response: {response[:200]}")
                import sys; sys.exit(1)
        else:
            CONSECUTIVE_JSON_FAILURES = 0 # Reset on success

        score_result = score_nota(response, correct_idx)

        # --- NEW: Detailed Logging for "Why" analysis ---
        results.append({
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "nota_placed_at_idx": correct_idx,
            "raw_response": response[:500],  # Log full response for debugging
            "score": score_result["score"],
            "why_correct": parsed.get("why_correct", "N/A"),
            "why_others_incorrect": parsed.get("why_others_incorrect", "N/A"),
            "got": score_result["got"],
            "expected": score_result["expected"],
        })
        time.sleep(RATE_LIMIT_DELAY)

        # results.append({
        #     "id": row.get("id", ""),
        #     "question": row.get("question", ""),
        #     "nota_placed_at_idx": correct_idx,
        #     "raw_response": response[:500],
        #     "score": score_result["score"],
        #     "why_correct": parsed.get("why_correct", "N/A"),
        #     "why_others_incorrect": parsed.get("why_others_incorrect", "N/A"),
        #     "got": score_result["got"],
        #     "expected": score_result["expected"],
        # })
        # time.sleep(RATE_LIMIT_DELAY)

    return pd.DataFrame(results)


def run_fake_test(dataset, model: str, endpoint: str, sample: int = None) -> pd.DataFrame:
    """Run Fake Questions Test."""
    global CONSECUTIVE_JSON_FAILURES
    CONSECUTIVE_JSON_FAILURES = 0
    print(f"\n{'='*60}")
    print(f"  Running FQT (Fake Questions Test)")
    print(f"  Model: {model} | Endpoint: {endpoint}")
    print(f"{'='*60}")

    data = list(dataset)
    if sample:
        data = random.sample(data, min(sample, len(data)))

    results = []
    for row in tqdm(data, desc="FQT"):
        # In the Med-HALT fake dataset, 'is_fake' column marks fake questions
        is_fake = bool(row.get("is_fake", True))
        query = format_fake_query(row)
        # clean_query = f"Question: {row['question']}\n"
        # response = call_api(query=clean_query, model=model, endpoint=endpoint)
        response = call_api(query, model, endpoint)

        # --- NEW: Kill-Switch Logic ---
        parsed, success = extract_json_from_response(response)
        if not success:
            CONSECUTIVE_JSON_FAILURES += 1
            if CONSECUTIVE_JSON_FAILURES >= MAX_ALLOWED_JSON_FAILURES:
                print(f"\n🛑 KILL-SWITCH TRIGGERED: {MAX_ALLOWED_JSON_FAILURES} consecutive JSON failures.")
                print(f"Last raw response: {response[:200]}")
                import sys; sys.exit(1)
        else:
            CONSECUTIVE_JSON_FAILURES = 0 # Reset on success

        score_result = score_fake(response, is_fake)

        # --- NEW: Detailed Logging for "Why" analysis ---
        results.append({
            "id": row.get("id", ""),
            "question": row.get("question", ""),
            "is_fake": is_fake,
            "raw_response": response[:500],  # Log full response for debugging
            "score": score_result["score"],
            "why_correct": parsed.get("why_correct", "N/A"),
            "why_others_incorrect": parsed.get("why_others_incorrect", "N/A"),
            "got": score_result["got"],
            "expected": score_result["expected"],
        })
        time.sleep(RATE_LIMIT_DELAY)

        # results.append({
        #     "id": row.get("id", ""),
        #     "question": row.get("question", ""),
        #     "is_fake": is_fake,
        #     "raw_response": response[:500],
        #     "score": score_result["score"],
        #     "got": score_result["got"],
        #     "expected": score_result["expected"],
        # })
        # time.sleep(RATE_LIMIT_DELAY)

    return pd.DataFrame(results)


# ==========================================
# RESULTS SUMMARY
# ==========================================

def print_summary(all_results: dict, model: str, endpoint: str):
    """Print a clean summary table of all test scores."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*60}")
    print(f"  MED-HALT EVALUATION RESULTS")
    print(f"  Model:    {model}")
    print(f"  Endpoint: {endpoint} ({'RAG pipeline' if endpoint == 'rag' else 'Generic LLM'})")
    print(f"  Time:     {timestamp}")
    print(f"{'='*60}")

    total_score = 0
    total_questions = 0

    for test_name, df in all_results.items():
        if df is None or len(df) == 0:
            continue
        n = len(df)
        correct = df["score"].sum()
        accuracy = (correct / n) * 100
        total_score += correct
        total_questions += n

        print(f"\n  {test_name}")
        print(f"  {'─'*40}")
        print(f"  Questions:  {n}")
        print(f"  Correct:    {int(correct)}")
        print(f"  Accuracy:   {accuracy:.1f}%")

        # Point score (like the paper: +1 correct, -0.25 wrong)
        point_score = correct - (0.25 * (n - correct))
        print(f"  Point Score (+1/-0.25): {point_score:.2f} / {n:.0f}")

    if total_questions > 0:
        overall = (total_score / total_questions) * 100
        overall_point = total_score - (0.25 * (total_questions - total_score))
        print(f"\n{'─'*60}")
        print(f"  OVERALL ACCURACY:    {overall:.1f}%  ({int(total_score)}/{total_questions})")
        print(f"  OVERALL POINT SCORE: {overall_point:.2f} / {total_questions}")
        print(f"{'='*60}\n")


def save_results(all_results: dict, model: str, endpoint: str):
    """Save all results to CSV files."""
    safe_model = model.replace("/", "_").replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(RESULTS_DIR, f"{safe_model}_{endpoint}_{timestamp}")
    os.makedirs(folder, exist_ok=True)

    summary_rows = []
    for test_name, df in all_results.items():
        if df is None or len(df) == 0:
            continue
        df.to_csv(os.path.join(folder, f"{test_name}.csv"), index=False)
        n = len(df)
        correct = df["score"].sum()
        summary_rows.append({
            "test": test_name,
            "model": model,
            "endpoint": endpoint,
            "n_questions": n,
            "n_correct": int(correct),
            "accuracy": round((correct / n) * 100, 2),
            "point_score": round(correct - 0.25 * (n - correct), 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(folder, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Results saved to: {folder}/")
    return folder


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ClinicalAI against the Med-HALT benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenRouter model ID (e.g. 'meta-llama/llama-3.3-70b-instruct')"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "FCT", "NOTA", "FQT"],
        help="Which test to run (default: all)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="rag",
        choices=["rag", "generic"],
        help="'rag' uses your full clinical pipeline, 'generic' uses plain LLM (default: rag)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of questions to sample per test (default: full dataset)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between API calls (default: 1.5)"
    )
    args = parser.parse_args()

    global RATE_LIMIT_DELAY
    RATE_LIMIT_DELAY = args.delay

    print(f"\n🏥 Med-HALT Evaluation — ClinicalAI")
    print(f"   Model:    {args.model}")
    print(f"   Test:     {args.test}")
    print(f"   Endpoint: {args.endpoint}")
    print(f"   Sample:   {args.sample or 'Full dataset'}")
    print(f"   API:      {API_BASE_URL}")

    # Verify backend is running
    try:
        r = requests.get(f"{API_BASE_URL}/", timeout=5)
        print(f"   Backend:  ✅ Online\n")
    except Exception:
        print(f"   Backend:  ❌ Not reachable at {API_BASE_URL}")
        print(f"   Start your FastAPI server first: uvicorn main:app --reload")
        return

    # Load Med-HALT datasets from HuggingFace
    print("📥 Loading Med-HALT datasets from HuggingFace...")
    try:
        # Reasoning tests use the medical MCQ dataset
        # Med-HALT reasoning tests are based on the openlifescienceai/Med-HALT dataset
        fct_dataset   = load_dataset("openlifescienceai/Med-HALT", "reasoning_FCT",   split="train")
        nota_dataset  = load_dataset("openlifescienceai/Med-HALT", "reasoning_nota",  split="train")
        fake_dataset  = load_dataset("openlifescienceai/Med-HALT", "reasoning_fake",  split="train")
        print("   ✅ Datasets loaded successfully\n")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        print("   Make sure you have: pip install datasets")
        print("   Dataset: https://huggingface.co/datasets/openlifescienceai/Med-HALT")
        return

    all_results = {}

    if args.test in ("all", "FCT"):
        all_results["FCT"] = run_fct_test(
            fct_dataset, args.model, args.endpoint, args.sample
        )

    if args.test in ("all", "NOTA"):
        all_results["NOTA"] = run_nota_test(
            nota_dataset, args.model, args.endpoint, args.sample
        )

    if args.test in ("all", "FQT"):
        all_results["FQT"] = run_fake_test(
            fake_dataset, args.model, args.endpoint, args.sample
        )

    print_summary(all_results, args.model, args.endpoint)
    save_results(all_results, args.model, args.endpoint)


if __name__ == "__main__":
    main()
