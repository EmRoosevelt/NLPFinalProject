import json
from tqdm import tqdm
from openai import OpenAI

# Configuration
BASE_URL = "https://api.openai.com/v1"

API_KEY ="sk"
MODEL_NAME = "gpt-5"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# Load dataset
with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

predictions = []


# Inference loop
for sample in tqdm(data, desc="Predicting DMRS..."):
    # Build conversation string
    conversation = "\n".join(
        f"{turn['speaker']}: {turn['text']}"
        for turn in sample["dialogue"]
    )

    # Build prompt
    prompt = f"""
You are a Defense Mechanism Rating Scale (DMRS) specialist.
Examine the dialogue carefully and select the single most appropriate defense tier.

When multiple defenses seem plausible, choose the tier with the strongest supporting evidence.
If evidence is weak or contradictory, default to 0 (No defense).

Return exactly one line:
label: <0-8>

Dialogue context:
{conversation}

Target utterance:
{sample["current_text"]}

Available labels:
0 = No defense
1 = Action Defense Level
2 = Major Image-distorting Defense Level
3 = Disavowal Defense Level
4 = Minor Image-distorting Defense Level
5 = Neurotic Defense Level
6 = Obsessional Defense Level
7 = Highly Adaptive Defense Level
8 = Need More Information
""".strip()

    # Query model
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    output = response.choices[0].message.content.strip()

    # Extract label
    try:
        label = int(output.split(":")[1].strip())
    except Exception:
        label = 0  # fallback if model output is malformed

    predictions.append({
        "id": sample["id"],
        "label": label
    })

# Save predictions
with open("prediction.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)

print("Saved predictions to prediction.json")