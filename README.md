# LLM Chatbot Evaluation

A lightweight evaluation framework designed to benchmark and evaluate LLM responses across varied difficulty tiers and edge cases.

---

## Project Structure

```
llm-chatbot-evaluation/
├── test_set.json       # 20 curated evaluation test cases (easy, hard, edge)
├── evaluate.py         # Evaluation harness script (to be implemented)
├── requirements.txt    # Python dependencies
├── .env.example        # Template for environment configuration
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation and setup guide
```

---

## Test Set Composition

The evaluation dataset (`test_set.json`) contains **20 items** spanning general knowledge, Python/ML fundamentals, and mathematics:

| Category | Count | Description | Topics Covered |
| :--- | :--- | :--- | :--- |
| **`easy`** | 8 | Straightforward factual recall questions with concise ground-truth answers. | Geography, chemistry, elementary geometry, Python basics, basic ML definitions. |
| **`hard`** | 8 | Multi-step reasoning, calculations, chronological ordering, and code comprehension. | Multi-step percentages, list comprehensions, rate problems, mutable default args, ML metrics (precision), probability, chronological sorting, softmax monotonicity. |
| **`edge`** | 4 | Robustness tests checking model handling of ambiguity, emptiness, out-of-scope predictions, and prompt injection. | Subjective/ambiguous questions, whitespace-only input, unanswerable future financial predictions, jailbreak/prompt injection attempts. |

Each item contains:
- `id`: Unique identifier (1–20)
- `category`: `"easy"`, `"hard"`, or `"edge"`
- `question`: Prompt presented to the model
- `reference`: Short ground-truth answer or explicit evaluation criterion

---

## Getting Started

### 1. Clone & Navigate
```bash
git clone <repository-url>
cd llm-chatbot-evaluation
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and supply your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

---

## Next Steps

- Implement test evaluation logic in `evaluate.py`.
- Connect the evaluation harness to Google GenAI API (`google-genai`).
- Compute performance metrics across categories (accuracy on easy/hard, adherence on edge cases).

