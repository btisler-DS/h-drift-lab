import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/collective_alignment_1/comparisons.parquet")
OUT_PATH = Path("data/processed/ca1_responses.parquet")


def extract_user_prompt(prompt_obj):
    """
    CA-1 `prompt` is a dict with an `id` and `messages` list.
    We pull the first user message content as the prompt text.
    """
    messages = prompt_obj.get("messages", [])
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def extract_response_text(response_obj):
    """
    Each entry in `responses` is of the form:
      { "messages": [ { "role": "assistant", "content": "..." }, ... ],
        "response_index": "A" }
    We join all assistant message contents (usually just one).
    """
    msgs = response_obj.get("messages", [])
    parts = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    return " ".join(parts).strip()


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing CA-1 parquet at {RAW_PATH}")

    df = pd.read_parquet(RAW_PATH)
    print(f"Loaded {len(df)} CA-1 comparison rows from {RAW_PATH}")

    records = []

    for _, row in df.iterrows():
        prompt = row["prompt"]
        responses = row["responses"]

        conv_id = prompt.get("id")
        prompt_text = extract_user_prompt(prompt)

        # one record per candidate response (A/B/C/D…)
        for resp in responses:
            resp_index = resp.get("response_index")
            resp_text = extract_response_text(resp)

            records.append(
                {
                    "conversation_id": conv_id,
                    "response_index": resp_index,
                    "prompt_text": prompt_text,
                    "response_text": resp_text,
                }
            )

    out_df = pd.DataFrame.from_records(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)

    print(f"Wrote {len(out_df)} rows to {OUT_PATH}")
    print(out_df.head(5))


if __name__ == "__main__":
    main()
