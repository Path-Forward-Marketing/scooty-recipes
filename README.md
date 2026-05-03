# Scooty Recipes

> *Your Recipes, Only Healthier.*

Take any recipe (text or image) and get a modified version that's lower in cholesterol, diabetic-friendly, or both — with side-by-side nutrition comparison and citations to USDA / AHA / ADA guidelines.

Styled in the **Path Forward Marketing** design system (forest & cream palette, italic Poppins display, Barlow Condensed all-caps section titles).

## What it does

1. **Input:** paste a recipe as text, or upload an image (JPG/PNG/WebP/screenshot).
2. **Parse:** Claude extracts structured ingredients + instructions.
3. **Baseline nutrition:** estimated per serving from USDA FoodData Central reference values.
4. **Glycemic Index / Load:** looked up in the Atkinson 2021 peer-reviewed GI table; foods not in the table fall back to a Claude estimate (clearly labeled).
5. **Toggle goals:** lower cholesterol and/or diabetic-friendly.
6. **Modified recipe + comparison:** ingredient swaps with rationale, plus a variance table showing change in calories, sugar, sat fat, cholesterol, GI, GL, etc.

## Setup

### 1. Get API keys

- **Claude API key:** https://console.anthropic.com — needed for parsing, nutrition, and modification.

### 2. Install dependencies

```bash
cd recipe-modifier
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure secrets

Copy the example file and fill in your key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and paste your Claude key
```

Alternatively, set the `ANTHROPIC_API_KEY` environment variable, or paste the key into the sidebar at runtime.

### 4. Run

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deployment

The simplest way to share this with others is **Streamlit Community Cloud** (free):

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io and connect the repo.
3. In the app's "Advanced settings" → "Secrets", paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy. You'll get a public URL like `https://your-app.streamlit.app`.

> Note: API costs are charged to whichever Claude account is configured in secrets. For public deployment, consider rate limiting or asking users to bring their own key.

## Project layout

```
recipe-modifier/
├── app.py              # Streamlit UI
├── claude_calls.py     # Claude API wrappers (parse, nutrition, modify)
├── models.py           # Pydantic data models
├── gi_lookup.py        # GI/GL lookup: Atkinson table + Claude fallback
├── data/
│   └── gi_table.csv    # Starter Atkinson 2021 GI table (~60 common foods)
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example
├── .env.example
└── .gitignore
```

## Important notes

- **Not medical advice.** Nutrition values are estimates from Claude's training data based on USDA FDC reference values — not live database lookups. For clinical use, consult a registered dietitian.
- **GI table is a starter set.** ~60 common foods. To extend, add rows to `data/gi_table.csv` (food_key, display_name, gi, carbs_per_100g, reference). For a comprehensive set, ingest the supplementary tables from Atkinson et al. 2021, *American Journal of Clinical Nutrition*.
- **Modifications cite specific guidelines** (AHA, ADA, USDA Dietary Guidelines) for each ingredient swap. The citations come from Claude's training; recheck against the live source for clinical use.

## How it uses Claude

- Model: `claude-opus-4-7` for all calls (high reasoning quality)
- Adaptive thinking enabled for nutrition estimation and recipe modification (intelligence-sensitive tasks)
- Prompt caching on stable system prompts to reduce repeat-call cost
- Structured outputs via Pydantic models (`client.messages.parse`)

## Costs

Each full flow (parse + baseline nutrition + modification + modified nutrition) is roughly **3-5 cents** at Opus 4.7 prices. Cache hits on the system prompt reduce subsequent calls. If costs become an issue, swap `MODEL = "claude-opus-4-7"` to `"claude-sonnet-4-6"` in `claude_calls.py` — Sonnet handles this task well at ~60% lower cost.
