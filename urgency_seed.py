# %%
import csv
import json
import os
import re
import time

import pandas as pd
import requests
from tqdm.auto import tqdm

LOCAL_API_URL = "http://localhost:1234/api/v1/chat"
LOCAL_MODEL_NAME = "liquid/lfm2.5-1.2b"
REQUEST_TIMEOUT_SECONDS = 120

# %%
def extract_text(payload):
    """Extract the assistant text from common local-model response shapes."""
    if isinstance(payload, str):
        return payload

    if isinstance(payload, list):
        pieces = []
        for item in payload:
            piece = extract_text(item)
            if piece:
                pieces.append(piece)
        return "\n".join(pieces).strip()

    if isinstance(payload, dict):
        for key in ("text", "content", "response", "output", "message", "reply"):
            if key in payload:
                extracted = extract_text(payload[key])
                if extracted:
                    return extracted

        if "choices" in payload and isinstance(payload["choices"], list):
            for choice in payload["choices"]:
                extracted = extract_text(choice)
                if extracted:
                    return extracted

        if "delta" in payload and isinstance(payload["delta"], dict):
            extracted = extract_text(payload["delta"])
            if extracted:
                return extracted

    return ""


def call_local_model(system_prompt, prompt, model_name=LOCAL_MODEL_NAME):
    """Send a prompt to the local chat endpoint and return the text response."""
    payload = {
        "model": model_name,
        "system_prompt": system_prompt,
        "input": prompt,
    }

    response = requests.post(LOCAL_API_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(f"Local model returned non-JSON output: {response.text[:200]}") from exc

    text = extract_text(data)
    if not text:
        raise ValueError(f"Local model returned an unexpected payload: {data}")

    return text.strip()


def extract_json_array(text_response):
    """Find and parse a JSON array from the model output."""
    match = re.search(r"(\[.*?\])", text_response, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    if text_response.startswith("```"):
        code_text = text_response.strip("`").strip()
        if code_text.lower().startswith("json"):
            code_text = code_text[4:].strip()
        return json.loads(code_text)

    raise ValueError("No valid JSON array found in the local model response.")


def generate_seed_topics(total_needed=1000, output_file="input_topics.csv"):
    """Generate non-urgent seed topics using the local model endpoint."""
    seed_system_instruction = """
You are a synthetic data specialist creating seed topics for a machine learning dataset.
Your task is to generate a JSON array containing exactly 50 distinct, realistic, and completely NON-URGENT user support scenarios or topics.
Output ONLY a valid JSON list of strings. No markdown, no explanations.
"""

    file_exists = os.path.isfile(output_file)
    existing_topics = set()

    if file_exists:
        with open(output_file, mode="r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    existing_topics.add(row[0].strip().lower())

    print(f"Loaded {len(existing_topics)} existing seed topics.")

    with open(output_file, mode="a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["topic"])

        pbar = tqdm(total=total_needed, initial=len(existing_topics), desc="Generating Seeds")

        while len(existing_topics) < total_needed:
            prompt = (
                f"{seed_system_instruction}\n\n"
                "Generate 50 fresh topics. Focus on everyday routine user tasks."
            )

            try:
                text_response = call_local_model(
                    system_prompt=seed_system_instruction,
                    prompt="Generate 50 fresh topics. Focus on everyday routine user tasks.",
                )
                new_topics = extract_json_array(text_response)

                for topic in new_topics:
                    topic_cleaned = str(topic).strip().lower()
                    if topic_cleaned and topic_cleaned not in existing_topics:
                        existing_topics.add(topic_cleaned)
                        writer.writerow([topic_cleaned])
                        pbar.update(1)

                handle.flush()
                time.sleep(0.5)

            except Exception as exc:
                print(f"\n[Local model error] {exc}")
                time.sleep(2)

    print(f"\nSuccess! Generated {len(existing_topics)} unique seed topics in '{output_file}'.")


def generate_urgency_variants(input_file="input_topics.csv", output_file="synthetic_urgency_dataset.csv", n_variations=20):
    """Generate urgency variants for each seed topic using the local model endpoint."""
    system_instruction = """
You are a synthetic data generator for emotion and urgency detection in text for user side inputs, to help train an urgency text classification model.

Your task: Given an input topic, create N variations that add explicit urgency/pressure language while preserving the core emotion and meaning.

Rules:
1. Add urgency tactics naturally (time pressure, deadlines, consequences, immediacy)
2. Make variations realistic - they should sound like real human text
3. Vary the urgency intensity: mild, moderate, and high pressure
4. Output ONLY the generated texts, one per line, no numbering or explanations

Urgency tactics to use (vary them):
- Time references: "immediately", "right now", "before [time]", "ASAP", "urgent"
- Deadline language: "by tomorrow", "within 24 hours", "deadline", "expires"
- Consequence framing: "or else", "will be lost", "will fail", "crisis", "emergency"
- Authority/pressure: "must", "have to", "need to", "critical", "essential"
- Scarcity: "limited time", "limited spots", "running out", "last chance"
"""

    df = pd.read_csv(input_file)
    topics = df["topic"].tolist()

    file_exists = os.path.isfile(output_file)
    with open(output_file, mode="a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["source_topic", "synthetic_text"])

        print(f"Starting generation for {len(topics)} topics...")

        for topic in tqdm(topics):
            prompt = f"Seed Input Topic: '{topic}'\nN={n_variations} variations:"
            try:
                text_response = call_local_model(system_prompt=system_instruction, prompt=prompt)
                for line in text_response.splitlines():
                    cleaned_line = line.strip()
                    if cleaned_line:
                        writer.writerow([topic, cleaned_line])
                handle.flush()
                time.sleep(0.5)
            except Exception as exc:
                print(f"\n[Local model error for topic '{topic}'] {exc}")
                time.sleep(2)

    print(f"\nGeneration complete! Saved to {output_file}.")

# %%
if __name__ == "__main__":
    #%%
    generate_seed_topics()
    # %%
    generate_urgency_variants()
