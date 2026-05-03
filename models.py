from pydantic import BaseModel, Field
from typing import Optional


class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name, e.g. 'all-purpose flour', 'olive oil'")
    quantity: float = Field(description="Numerical quantity. Use 0 if 'to taste' or 'pinch'.")
    unit: str = Field(description="Unit: tbsp, tsp, cup, oz, lb, g, kg, ml, l, count, or empty string")
    notes: str = Field(default="", description="Prep notes: chopped, diced, optional, etc.")


class Recipe(BaseModel):
    title: str
    ingredients: list[Ingredient]
    instructions: list[str]
    servings: int = Field(description="Number of servings the recipe yields. Use 4 if not stated.")


class NutritionFacts(BaseModel):
    """Per-serving nutrition estimate."""
    calories: float
    protein_g: float
    carbs_g: float
    sugar_g: float = Field(description="Total sugars (added + naturally occurring)")
    added_sugar_g: float = Field(default=0, description="Added sugars only")
    fiber_g: float
    fat_g: float
    saturated_fat_g: float
    cholesterol_mg: float
    sodium_mg: float
    glycemic_index: Optional[float] = Field(default=None, description="Estimated meal-level GI (0-100+)")
    glycemic_load: Optional[float] = Field(default=None, description="Estimated meal-level GL per serving")
    gi_source: str = Field(default="", description="Source of GI/GL: 'Atkinson 2021 table', 'Claude estimate', or 'mixed'")
    citations: list[str] = Field(default_factory=list, description="Reference sources for nutrient values")
    caveats: str = Field(default="", description="Notes on uncertainty or limitations")


class IngredientSwap(BaseModel):
    original_ingredient: str
    replacement: str = Field(description="New ingredient with quantity, e.g. '1/2 cup unsweetened applesauce'")
    rationale: str = Field(description="Why this swap helps the chosen goal(s)")
    citation: str = Field(description="Specific guideline reference (e.g. 'AHA 2021 Dietary Guidance', 'ADA Standards of Care 2024')")


class ModifiedRecipe(BaseModel):
    recipe: Recipe
    swaps: list[IngredientSwap]
    notes: str = Field(default="", description="General notes on the modification approach")
