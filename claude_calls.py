"""All Claude API calls for the recipe modifier app.

Uses Claude Opus 4.7 with adaptive thinking and prompt caching on stable system prompts.
"""

import base64
from typing import Optional

import anthropic

from models import ModifiedRecipe, NutritionFacts, Recipe
from gi_lookup import resolve_gi


MODEL = "claude-sonnet-4-6"


# ---------- System prompts (stable — cached) ----------

PARSE_SYSTEM_PROMPT = """\
You are a recipe parser. Extract structured data from recipes provided as text or images.

Guidelines:
- Extract title, ingredients (with quantity, unit, name, prep notes), instructions, and serving count.
- For ranges like "2-3 cloves", use the average (2.5).
- For "a pinch", "to taste", "as needed": set quantity to 0 and note in the notes field.
- Normalize units to common forms: tbsp, tsp, cup, oz, lb, g, kg, ml, l, count.
- For unitless counts (e.g., "2 eggs"), use unit "count".
- Preserve prep notes like "chopped", "minced", "optional", "divided".
- If servings aren't stated, estimate from quantities (typical recipe = 4) and note it.
- Keep the ingredient name normalized (e.g., "all-purpose flour", not "flour, AP")."""


NUTRITION_SYSTEM_PROMPT = """\
You are a nutrition analysis assistant. Estimate per-serving nutrition for recipes based on \
USDA FoodData Central (FDC) reference values from your training data.

Methodology:
1. For each ingredient, identify the closest USDA FDC food entry.
2. Compute total nutrient contribution using ingredient quantity and standard density/conversions:
   - 1 cup all-purpose flour ≈ 120g
   - 1 cup granulated sugar ≈ 200g
   - 1 cup brown sugar (packed) ≈ 220g
   - 1 tbsp butter = 14g
   - 1 tbsp oil = 14g
   - 1 large egg ≈ 50g
   - 1 cup milk ≈ 244g
   - 1 cup water ≈ 240g
3. Sum across all ingredients, divide by servings.

Required output fields:
- calories, protein_g, carbs_g, sugar_g, added_sugar_g, fiber_g, fat_g, saturated_fat_g, \
  cholesterol_mg, sodium_mg
- Leave glycemic_index and glycemic_load as null — those are computed separately from a GI table.

Citations field:
- List the USDA FDC categories you drew on (e.g., "USDA FDC: Butter, salted (FDC 173410)").
- Mark this as estimated, not directly queried from the live USDA API.

Caveats field:
- Note ingredient-level uncertainty (e.g., "fat content varies by butter brand").

Important: This is a training-data estimate, not a live USDA API call. Phrase citations accordingly. \
For clinical/medical use, consult a registered dietitian."""


MODIFY_SYSTEM_PROMPT = """\
You are a recipe modification assistant. Modify recipes to be lower in cholesterol and/or \
diabetic-friendly while preserving the dish's identity and flavor profile.

Reference standards (cite specifically in each swap):
- USDA Dietary Guidelines for Americans 2020-2025
- American Heart Association (AHA) 2021 Dietary Guidance
- American Diabetes Association (ADA) Standards of Care 2024 — Nutrition Therapy

For LOWER CHOLESTEROL:
- AHA: limit saturated fat to <6% of total calories
- AHA: <300mg dietary cholesterol/day (lower if at risk)
- Replace butter with olive oil, avocado oil, or plant-based fats
- Replace whole eggs with egg whites or flax/chia "eggs" where structural role allows
- Reduce or substitute red meat with fish, poultry, legumes
- Add soluble fiber sources (oats, beans, citrus pectin, psyllium)

For DIABETIC-FRIENDLY:
- ADA: emphasize carbohydrate quality and quantity; lower glycemic load
- Replace refined grains with whole grains (whole wheat, oats, quinoa, barley)
- Reduce added sugars; small amounts of sugar alcohols (erythritol) or stevia for sweetness if needed
- Add fiber and protein to slow glucose response
- Use legumes, non-starchy vegetables, and lean proteins to balance carb load
- Avoid high-GI ingredients (white rice, white bread, mashed potatoes); prefer low-GI alternatives

If BOTH goals are selected, choose substitutions that satisfy both — e.g., olive oil + whole grain \
swaps simultaneously address sat fat and glycemic load.

Constraints:
- Preserve the dish's culinary identity. Don't turn lasagna into salad.
- Substitutions must be available in typical US grocery stores.
- For structural ingredients (e.g., gluten in bread, eggs in custard), note any texture/flavor tradeoffs.
- Update instructions if the substitution requires different cooking technique (e.g., olive oil burns at lower temp than butter).

For each substitution, output:
- original_ingredient: the ingredient being replaced
- replacement: the new ingredient with quantity, e.g. "1/2 cup unsweetened applesauce"
- rationale: why this swap helps the goal
- citation: specific guideline (e.g., "AHA 2021 Dietary Guidance — saturated fat reduction")"""


# ---------- API call functions ----------


def parse_recipe_from_text(client: anthropic.Anthropic, recipe_text: str) -> Recipe:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": PARSE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": f"Parse this recipe:\n\n{recipe_text}"}
        ],
        output_format=Recipe,
    )
    return response.parsed_output


def parse_recipe_from_image(
    client: anthropic.Anthropic, image_bytes: bytes, media_type: str
) -> Recipe:
    """media_type: 'image/jpeg' or 'image/png' or 'image/webp' or 'image/gif'."""
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": PARSE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "Parse the recipe in this image."},
                ],
            }
        ],
        output_format=Recipe,
    )
    return response.parsed_output


def estimate_nutrition(client: anthropic.Anthropic, recipe: Recipe) -> NutritionFacts:
    """Get baseline nutrition from Claude's USDA FDC reference values.

    GI/GL is filled in separately via gi_lookup."""
    ingredient_lines = "\n".join(
        f"- {ing.quantity} {ing.unit} {ing.name}"
        + (f" ({ing.notes})" if ing.notes else "")
        for ing in recipe.ingredients
    )
    user_msg = (
        f"Recipe: {recipe.title}\n"
        f"Servings: {recipe.servings}\n\n"
        f"Ingredients:\n{ingredient_lines}\n\n"
        "Estimate the per-serving nutrition. Set glycemic_index and glycemic_load to null."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[
            {
                "type": "text",
                "text": NUTRITION_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
        output_format=NutritionFacts,
    )
    return response.parsed_output


def compute_meal_gi_gl(
    client: anthropic.Anthropic, recipe: Recipe, nutrition: NutritionFacts
) -> NutritionFacts:
    """Compute meal-level GI and GL from per-ingredient lookups, mutating nutrition in place.

    Strategy:
    - For each ingredient, resolve a GI value (table or Claude fallback).
    - Approximate ingredient carb contribution using nutrition.carbs_g per serving prorated by ingredient mass share.
    - Since we don't have per-ingredient mass here, use a simpler approach: weighted average GI of the
      carb-contributing ingredients, then GL = (GI * carbs_per_serving) / 100.

    This is a rough estimate; for precise GL you'd need per-ingredient carb breakdown.
    """
    ingredient_gis = []
    sources = []
    any_estimated = False

    for ing in recipe.ingredients:
        if ing.quantity == 0:
            continue
        try:
            resolved = resolve_gi(client, ing.name)
        except Exception:
            continue
        # Skip ingredients that contribute negligible carbs (oil, salt, herbs)
        # using a heuristic: if name contains low-carb keyword, skip
        low_carb_keywords = ("oil", "salt", "pepper", "herb", "spice", "vinegar", "vanilla",
                             "extract", "yeast", "baking soda", "baking powder", "water")
        if any(kw in ing.name.lower() for kw in low_carb_keywords):
            continue
        ingredient_gis.append(resolved.gi)
        sources.append(resolved.source)
        if not resolved.in_table:
            any_estimated = True

    if not ingredient_gis:
        nutrition.glycemic_index = None
        nutrition.glycemic_load = None
        nutrition.gi_source = "no carb-bearing ingredients identified"
        return nutrition

    avg_gi = sum(ingredient_gis) / len(ingredient_gis)
    nutrition.glycemic_index = round(avg_gi, 1)
    nutrition.glycemic_load = round((avg_gi * nutrition.carbs_g) / 100, 1)

    if any_estimated:
        nutrition.gi_source = "mixed (Atkinson 2021 table + Claude estimates for foods not in table)"
    else:
        nutrition.gi_source = "Atkinson 2021 GI tables"

    return nutrition


def modify_recipe(
    client: anthropic.Anthropic,
    recipe: Recipe,
    low_cholesterol: bool,
    diabetic_friendly: bool,
) -> ModifiedRecipe:
    goals = []
    if low_cholesterol:
        goals.append("LOWER CHOLESTEROL (AHA-aligned)")
    if diabetic_friendly:
        goals.append("DIABETIC-FRIENDLY (ADA-aligned)")
    goals_str = " AND ".join(goals)

    ingredient_lines = "\n".join(
        f"- {ing.quantity} {ing.unit} {ing.name}"
        + (f" ({ing.notes})" if ing.notes else "")
        for ing in recipe.ingredients
    )
    instruction_lines = "\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe.instructions))

    user_msg = (
        f"Modify this recipe to be {goals_str}.\n\n"
        f"Original recipe:\n"
        f"Title: {recipe.title}\n"
        f"Servings: {recipe.servings}\n\n"
        f"Ingredients:\n{ingredient_lines}\n\n"
        f"Instructions:\n{instruction_lines}\n\n"
        "Return the modified recipe (full ingredient list and instructions, with substitutions applied) "
        "plus a list of swaps with rationale and guideline citations."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[
            {
                "type": "text",
                "text": MODIFY_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
        output_format=ModifiedRecipe,
    )
    return response.parsed_output
