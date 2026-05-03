"""Streamlit UI for the recipe modifier."""

import os
from io import BytesIO

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from claude_calls import (
    compute_meal_gi_gl,
    estimate_nutrition,
    modify_recipe,
    parse_recipe_from_image,
    parse_recipe_from_text,
)
from models import NutritionFacts, Recipe

load_dotenv()

st.set_page_config(page_title="Scooty Recipes", page_icon="🥗", layout="wide")


# ---------- Path Forward Marketing brand styling ----------
# Tokens mirror ~/.claude/skills/path-forward-marketing-design/colors_and_type.css

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700&family=Barlow+Condensed:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
      :root {
        --pfm-forest: #2D5A3D;
        --pfm-forest-deep: #234A31;
        --pfm-forest-soft: #3E6F50;
        --pfm-cream: #D9D7C2;
        --pfm-cream-soft: #E8E6D6;
        --pfm-cream-deep: #C2BFA6;
        --pfm-ink: #1A1F1C;
        --pfm-ink-muted: #4A524C;
        --pfm-ink-subtle: #7D847F;
        --pfm-line: #D8D6CC;
        --pfm-paper: #FAF8F0;
        --pfm-paper-alt: #F3F0E4;
        --pfm-accent: #C38A3E;
      }

      html, body, [class*="css"] {
        font-family: 'Poppins', ui-sans-serif, system-ui, sans-serif;
        color: var(--pfm-ink);
      }

      /* Headlines: italic Poppins with tight letter-spacing */
      h1, h2 {
        font-family: 'Poppins', sans-serif !important;
        font-style: italic !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--pfm-ink) !important;
        text-wrap: balance;
      }
      h3, h4 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        font-style: normal !important;
        color: var(--pfm-ink) !important;
      }

      /* Eyebrow + section-title — Barlow Condensed all-caps echoes the FORWARD bar */
      .pfm-eyebrow {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--pfm-forest);
        margin-bottom: 0.25rem;
      }
      .pfm-section-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-weight: 700;
        font-size: 28px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--pfm-ink);
        line-height: 1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
      }
      .pfm-lede {
        font-size: 19px;
        line-height: 1.5;
        color: var(--pfm-ink-muted);
        max-width: 70ch;
      }
      .pfm-rule {
        height: 1px;
        background: var(--pfm-line);
        margin: 1.5rem 0;
        border: 0;
      }

      /* Buttons: sharp corners, forest fill */
      .stButton > button {
        border-radius: 4px !important;
        border: 1px solid var(--pfm-forest) !important;
        background: var(--pfm-forest) !important;
        color: var(--pfm-cream) !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em;
        transition: background 140ms ease;
      }
      .stButton > button:hover {
        background: var(--pfm-forest-soft) !important;
        border-color: var(--pfm-forest-soft) !important;
      }
      .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--pfm-forest) !important;
      }

      /* Inputs / cards: light rounding, hairline borders */
      .stTextInput input, .stTextArea textarea, .stFileUploader,
      [data-testid="stExpander"], [data-testid="stMetric"] {
        border-radius: 4px !important;
        border-color: var(--pfm-line) !important;
      }

      /* Toggles: forest accent when on */
      .stCheckbox [data-baseweb="checkbox"] [aria-checked="true"] {
        background-color: var(--pfm-forest) !important;
      }

      /* Sidebar */
      [data-testid="stSidebar"] {
        background: var(--pfm-paper-alt);
        border-right: 1px solid var(--pfm-line);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- API key handling ----------

def get_api_key() -> str | None:
    # Streamlit secrets first
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    # Env var second
    if os.getenv("ANTHROPIC_API_KEY"):
        return os.getenv("ANTHROPIC_API_KEY")
    return None


# ---------- Helpers ----------

def format_recipe_markdown(recipe: Recipe) -> str:
    lines = [f"### {recipe.title}", f"**Servings:** {recipe.servings}", "", "**Ingredients:**"]
    for ing in recipe.ingredients:
        qty_str = f"{ing.quantity:g}" if ing.quantity else ""
        unit_str = f" {ing.unit}" if ing.unit and ing.unit != "count" else ""
        notes_str = f" *({ing.notes})*" if ing.notes else ""
        if ing.quantity == 0:
            lines.append(f"- {ing.name}{notes_str}")
        else:
            lines.append(f"- {qty_str}{unit_str} {ing.name}{notes_str}")
    lines.append("")
    lines.append("**Instructions:**")
    for i, step in enumerate(recipe.instructions, 1):
        lines.append(f"{i}. {step}")
    return "\n".join(lines)


def nutrition_comparison_df(orig: NutritionFacts, mod: NutritionFacts) -> pd.DataFrame:
    rows = [
        ("Calories", "kcal", orig.calories, mod.calories),
        ("Total Carbs", "g", orig.carbs_g, mod.carbs_g),
        ("Total Sugar", "g", orig.sugar_g, mod.sugar_g),
        ("Added Sugar", "g", orig.added_sugar_g, mod.added_sugar_g),
        ("Fiber", "g", orig.fiber_g, mod.fiber_g),
        ("Protein", "g", orig.protein_g, mod.protein_g),
        ("Total Fat", "g", orig.fat_g, mod.fat_g),
        ("Saturated Fat", "g", orig.saturated_fat_g, mod.saturated_fat_g),
        ("Cholesterol", "mg", orig.cholesterol_mg, mod.cholesterol_mg),
        ("Sodium", "mg", orig.sodium_mg, mod.sodium_mg),
    ]
    if orig.glycemic_index is not None or mod.glycemic_index is not None:
        rows.append(("Glycemic Index (est.)", "0-100+", orig.glycemic_index or 0, mod.glycemic_index or 0))
        rows.append(("Glycemic Load (est.)", "per serving", orig.glycemic_load or 0, mod.glycemic_load or 0))

    data = []
    for nutrient, unit, o, m in rows:
        delta = m - o
        pct = ((delta / o) * 100) if o else 0
        data.append({
            "Nutrient": nutrient,
            "Unit": unit,
            "Original": round(o, 1),
            "Modified": round(m, 1),
            "Change": f"{'+' if delta >= 0 else ''}{round(delta, 1)}",
            "% Change": f"{'+' if pct >= 0 else ''}{round(pct, 1)}%" if o else "—",
        })
    return pd.DataFrame(data)


def get_client() -> anthropic.Anthropic | None:
    key = get_api_key()
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


# ---------- UI ----------

st.markdown(
    """
    <div class="pfm-eyebrow">Scooty Recipes</div>
    <h1 style="margin-top:0; font-size: clamp(36px, 4vw, 56px);">Your Recipes, Only Healthier</h1>
    <p class="pfm-lede">
      Drop in any recipe — text or photo — and get a version that's lower in cholesterol,
      diabetic-friendly, or both. Modifications grounded in USDA Dietary Guidelines, AHA, and ADA standards.
    </p>
    <hr class="pfm-rule" />
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="pfm-eyebrow">Scooty Recipes</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0;">About</h3>', unsafe_allow_html=True)
    st.markdown(
        "- **Recipe parsing:** Claude (text or image)\n"
        "- **Nutrition:** estimated from USDA FoodData Central reference values *(via Claude — not a live API call)*\n"
        "- **Glycemic Index/Load:** Atkinson 2021 GI table (peer-reviewed) with Claude fallback for foods not in table\n"
        "- **Modifications:** Claude, citing AHA/ADA/USDA guidelines"
    )
    st.markdown("---")
    st.markdown(
        "⚠️ **Not medical advice.** Estimates only. "
        "For clinical use, consult a registered dietitian."
    )
    st.markdown("---")

    if not get_api_key():
        st.warning("No API key configured.")
        manual_key = st.text_input("Claude API key", type="password",
                                    help="Get one at console.anthropic.com")
        if manual_key:
            st.session_state["manual_api_key"] = manual_key

# Override with manual key if entered
if "manual_api_key" in st.session_state and not get_api_key():
    os.environ["ANTHROPIC_API_KEY"] = st.session_state["manual_api_key"]

client = get_client()

if not client:
    st.error("⚠️ Set your Claude API key in `.streamlit/secrets.toml`, the `ANTHROPIC_API_KEY` env var, "
             "or paste it in the sidebar to begin.")
    st.stop()


# ---------- Step 1: input ----------

st.markdown('<div class="pfm-eyebrow">Step 01</div><div class="pfm-section-title">Provide a recipe</div>', unsafe_allow_html=True)
input_method = st.radio("Input method", ["Paste text", "Upload image"], horizontal=True)

col_input, col_parse = st.columns([4, 1])

if input_method == "Paste text":
    with col_input:
        recipe_text = st.text_area(
            "Paste recipe (ingredients + instructions)",
            height=250,
            placeholder="Title: Classic Chocolate Chip Cookies\nServings: 24\n\n"
                       "1 cup butter, softened\n3/4 cup sugar\n3/4 cup brown sugar\n2 large eggs\n"
                       "..."
        )
    parse_clicked = col_parse.button("Parse recipe", type="primary", use_container_width=True)
    if parse_clicked and recipe_text.strip():
        with st.spinner("Parsing recipe..."):
            try:
                st.session_state["recipe"] = parse_recipe_from_text(client, recipe_text)
                st.session_state.pop("baseline_nutrition", None)
                st.session_state.pop("modified", None)
            except Exception as e:
                st.error(f"Parse failed: {e}")

else:
    with col_input:
        image_file = st.file_uploader(
            "Upload a recipe image (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
    parse_clicked = col_parse.button("Parse image", type="primary", use_container_width=True)
    if parse_clicked and image_file:
        with st.spinner("Reading recipe from image..."):
            try:
                media_type = f"image/{image_file.type.split('/')[-1]}"
                # Normalize jpg → jpeg
                if media_type == "image/jpg":
                    media_type = "image/jpeg"
                st.session_state["recipe"] = parse_recipe_from_image(
                    client, image_file.read(), media_type
                )
                st.session_state.pop("baseline_nutrition", None)
                st.session_state.pop("modified", None)
            except Exception as e:
                st.error(f"Parse failed: {e}")


# ---------- Step 2: parsed recipe + baseline nutrition ----------

if "recipe" in st.session_state:
    recipe: Recipe = st.session_state["recipe"]
    st.markdown('<hr class="pfm-rule" /><div class="pfm-eyebrow">Step 02</div><div class="pfm-section-title">Original recipe</div>', unsafe_allow_html=True)
    st.markdown(format_recipe_markdown(recipe))

    if "baseline_nutrition" not in st.session_state:
        if st.button("Estimate baseline nutrition", type="primary"):
            with st.spinner("Estimating nutrition (USDA FDC reference values)..."):
                try:
                    nutrition = estimate_nutrition(client, recipe)
                    nutrition = compute_meal_gi_gl(client, recipe, nutrition)
                    st.session_state["baseline_nutrition"] = nutrition
                except Exception as e:
                    st.error(f"Nutrition estimate failed: {e}")

    if "baseline_nutrition" in st.session_state:
        n: NutritionFacts = st.session_state["baseline_nutrition"]
        st.subheader("Baseline nutrition (per serving)")

        cols = st.columns(4)
        cols[0].metric("Calories", f"{n.calories:.0f}")
        cols[1].metric("Carbs", f"{n.carbs_g:.1f} g")
        cols[2].metric("Sugar", f"{n.sugar_g:.1f} g")
        cols[3].metric("Sat. Fat", f"{n.saturated_fat_g:.1f} g")

        cols = st.columns(4)
        cols[0].metric("Fiber", f"{n.fiber_g:.1f} g")
        cols[1].metric("Cholesterol", f"{n.cholesterol_mg:.0f} mg")
        cols[2].metric("Glycemic Index", f"{n.glycemic_index:.0f}" if n.glycemic_index else "—")
        cols[3].metric("Glycemic Load", f"{n.glycemic_load:.1f}" if n.glycemic_load else "—")

        with st.expander("Sources & caveats"):
            st.markdown("**Nutrition citations (USDA FDC):**")
            for c in n.citations:
                st.markdown(f"- {c}")
            st.markdown(f"**GI/GL source:** {n.gi_source}")
            if n.caveats:
                st.markdown(f"**Caveats:** {n.caveats}")


# ---------- Step 3: modifications ----------

if "baseline_nutrition" in st.session_state:
    st.markdown('<hr class="pfm-rule" /><div class="pfm-eyebrow">Step 03</div><div class="pfm-section-title">Choose modifications</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    low_chol = col1.toggle("🫀 Lower cholesterol", help="AHA-aligned: reduce sat fat, dietary cholesterol")
    diabetic = col2.toggle("🩸 Diabetic-friendly", help="ADA-aligned: lower glycemic load, less added sugar")

    if (low_chol or diabetic) and st.button("Generate modified recipe", type="primary"):
        with st.spinner("Generating modified recipe..."):
            try:
                modified = modify_recipe(client, st.session_state["recipe"], low_chol, diabetic)
                mod_nutrition = estimate_nutrition(client, modified.recipe)
                mod_nutrition = compute_meal_gi_gl(client, modified.recipe, mod_nutrition)
                st.session_state["modified"] = modified
                st.session_state["modified_nutrition"] = mod_nutrition
            except Exception as e:
                st.error(f"Modification failed: {e}")


# ---------- Step 4: side-by-side comparison ----------

if "modified" in st.session_state and "modified_nutrition" in st.session_state:
    st.markdown('<hr class="pfm-rule" /><div class="pfm-eyebrow">Step 04</div><div class="pfm-section-title">Comparison</div>', unsafe_allow_html=True)

    col_orig, col_mod = st.columns(2)
    with col_orig:
        st.subheader("Original")
        st.markdown(format_recipe_markdown(st.session_state["recipe"]))
    with col_mod:
        st.subheader("Modified")
        st.markdown(format_recipe_markdown(st.session_state["modified"].recipe))

    st.subheader("Nutrition variance (per serving)")
    df = nutrition_comparison_df(
        st.session_state["baseline_nutrition"],
        st.session_state["modified_nutrition"],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Substitutions made")
    for swap in st.session_state["modified"].swaps:
        with st.container(border=True):
            st.markdown(f"**{swap.original_ingredient}** → **{swap.replacement}**")
            st.markdown(f"*Rationale:* {swap.rationale}")
            st.caption(f"📚 {swap.citation}")

    if st.session_state["modified"].notes:
        st.info(st.session_state["modified"].notes)
