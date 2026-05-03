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
- Reduce or substitute red meat with fish, poultry, legumes
- Add soluble fiber sources (oats, beans, citrus pectin, psyllium)

For DIABETIC-FRIENDLY:
- ADA: emphasize carbohydrate quality and quantity; lower glycemic load
- Replace refined grains with whole grains (whole wheat, oats, quinoa, barley)
- Reduce added sugars
- Add fiber and protein to slow glucose response
- Use legumes, non-starchy vegetables, and lean proteins to balance carb load
- Avoid high-GI ingredients (white rice, white bread, mashed potatoes); prefer low-GI alternatives

PREFERRED INGREDIENT SUBSTITUTIONS — reach for these well-established home-cook swaps when the recipe contains the source ingredient:

For lower saturated fat / cholesterol:
- Butter → unsweetened applesauce (in baking; 1:1 by volume — preserves moisture, dramatically cuts saturated fat)
- Butter → mashed avocado (1:1 in baked goods like brownies, cookies)
- Butter → olive oil or avocado oil (use 3/4 cup oil per 1 cup butter; for sautéing or non-baking applications)
- Whole eggs → 2 egg whites per whole egg (or flax egg: 1 tbsp ground flax + 3 tbsp water, in baked goods)
- Sour cream → plain nonfat Greek yogurt (1:1)
- Mayo → plain Greek yogurt or mashed avocado (1:1)
- Heavy cream → evaporated skim milk, cashew cream, or pureed silken tofu
- Cheese (when adding for flavor) → nutritional yeast (cheesy flavor, no saturated fat)
- Bacon → turkey bacon, smoked tempeh, or crisped mushrooms
- Ground beef → ground turkey, lean chicken, or lentils (½ meat + ½ lentils works well)
- Frying → baking, air-frying, or pan-searing in a small amount of olive oil

For diabetic-friendly / lower glycemic load:
- White rice → cauliflower rice, brown rice, or quinoa
- Mashed potatoes → mashed cauliflower (or 50/50 blend)
- Regular pasta → zucchini noodles, spaghetti squash, chickpea/lentil pasta, or whole wheat pasta
- White flour → whole wheat, oat, almond, or chickpea flour (note any texture changes)
- White bread → whole grain or sprouted-grain bread
- Refined sugar → mashed banana or unsweetened applesauce (in baked goods, where fruit sweetness works)
- Refined sugar → small amounts of date paste (1 cup pitted dates + 1 cup hot water, blended) — adds fiber and slows glucose response
- Dried fruit → fresh fruit (much less concentrated sugar)

ARTIFICIAL SWEETENER POLICY:
- The user provides a per-request preference in the user message (sweetener policy: ALLOWED or NOT ALLOWED).
- If ALLOWED: stevia, erythritol, monk fruit, and allulose are acceptable when sweetness is structurally needed and reduction alone won't achieve the goal.
- If NOT ALLOWED: do NOT suggest stevia, erythritol, monk fruit, allulose, sucralose, aspartame, or saccharin. Rely instead on natural sweetness reduction, fresh/mashed fruit, dates, or flavor compensation (vanilla, cinnamon, almond extract, citrus zest).

If BOTH goals are selected, choose substitutions that satisfy both — e.g., olive oil + whole grain \
swaps simultaneously address sat fat and glycemic load.

Constraints:
- Preserve the dish's culinary identity. Don't turn lasagna into salad.
- Substitutions must be available in typical US grocery stores.
- For structural ingredients (e.g., gluten in bread, eggs in custard), note any texture/flavor tradeoffs.
- Update instructions if the substitution requires different cooking technique (e.g., olive oil burns at lower temp than butter; cauliflower rice cooks faster than white rice).

For each substitution, output:
- original_ingredient: the ingredient being replaced
- replacement: the new ingredient with quantity, e.g. "1/2 cup unsweetened applesauce"
- rationale: why this swap helps the goal
- citation: specific guideline (e.g., "AHA 2021 Dietary Guidance — saturated fat reduction")"""


# ---------- API call functions ----------


def parse_recipe_from_text(client: anthropic.Anthropic, recipe_text: str) -> Recipe:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,
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
    if response.parsed_output is None:
        text_blocks = [b.text for b in response.content if b.type == "text"]
        preview = ("\n".join(text_blocks))[:300] if text_blocks else "(no text in response)"
        raise RuntimeError(
            f"Could not parse recipe (stop_reason={response.stop_reason}). "
            f"Response preview: {preview}"
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
    if response.parsed_output is None:
        raise RuntimeError(
            f"Could not parse recipe from image (stop_reason={response.stop_reason}). "
            "Try a clearer image or paste the recipe as text."
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
        max_tokens=8192,
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
    if response.parsed_output is None:
        text_blocks = [b.text for b in response.content if b.type == "text"]
        preview = ("\n".join(text_blocks))[:300] if text_blocks else "(no text in response)"
        raise RuntimeError(
            f"Could not parse nutrition response (stop_reason={response.stop_reason}). "
            f"Try again or simplify the recipe. Response preview: {preview}"
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
    allow_artificial_sweeteners: bool = False,
) -> ModifiedRecipe:
    goals = []
    if low_cholesterol:
        goals.append("LOWER CHOLESTEROL (AHA-aligned)")
    if diabetic_friendly:
        goals.append("DIABETIC-FRIENDLY (ADA-aligned)")
    goals_str = " AND ".join(goals)

    sweetener_policy = (
        "ALLOWED — stevia, erythritol, monk fruit, and allulose may be used when sweetness is structurally needed."
        if allow_artificial_sweeteners
        else "NOT ALLOWED — do not suggest stevia, erythritol, monk fruit, allulose, sucralose, aspartame, or saccharin. Use natural sweetness reduction, fresh/mashed fruit, dates, or flavor compensation (vanilla, cinnamon, almond extract, citrus zest) instead."
    )

    ingredient_lines = "\n".join(
        f"- {ing.quantity} {ing.unit} {ing.name}"
        + (f" ({ing.notes})" if ing.notes else "")
        for ing in recipe.ingredients
    )
    instruction_lines = "\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe.instructions))

    user_msg = (
        f"Modify this recipe to be {goals_str}.\n\n"
        f"Sweetener policy: {sweetener_policy}\n\n"
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
        max_tokens=12288,
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
    if response.parsed_output is None:
        text_blocks = [b.text for b in response.content if b.type == "text"]
        preview = ("\n".join(text_blocks))[:300] if text_blocks else "(no text in response)"
        raise RuntimeError(
            f"Could not generate modified recipe (stop_reason={response.stop_reason}). "
            f"Try again or pick fewer modifications. Response preview: {preview}"
        )
    return response.parsed_output
