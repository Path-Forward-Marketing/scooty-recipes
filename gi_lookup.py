"""GI/GL lookup with Atkinson 2021 starter table + Claude fallback for missing foods."""

import csv
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic
from pydantic import BaseModel, Field


GI_TABLE_PATH = Path(__file__).parent / "data" / "gi_table.csv"


@dataclass
class GIRecord:
    food_key: str
    display_name: str
    gi: float
    carbs_per_100g: float
    reference: str


class GIEstimate(BaseModel):
    """Claude-generated GI estimate when a food isn't in the table."""
    gi: float = Field(description="Estimated glycemic index, 0-110")
    confidence: str = Field(description="low | medium | high")
    similar_foods: list[str] = Field(description="Reference foods used to estimate")
    rationale: str = Field(description="One sentence explanation")


def load_table() -> dict[str, GIRecord]:
    table: dict[str, GIRecord] = {}
    with open(GI_TABLE_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[row["food_key"]] = GIRecord(
                food_key=row["food_key"],
                display_name=row["display_name"],
                gi=float(row["gi"]),
                carbs_per_100g=float(row["carbs_per_100g"]),
                reference=row["reference"],
            )
    return table


_TABLE: Optional[dict[str, GIRecord]] = None


def get_table() -> dict[str, GIRecord]:
    global _TABLE
    if _TABLE is None:
        _TABLE = load_table()
    return _TABLE


def lookup_in_table(ingredient_name: str) -> Optional[GIRecord]:
    """Direct + fuzzy match against the static table. Returns None if no good match."""
    table = get_table()
    name = ingredient_name.lower().strip()

    if name in table:
        return table[name]

    # Substring match — many ingredients are descriptive ("organic white rice")
    for key, record in table.items():
        if key in name or name in key:
            return record

    # Fuzzy match
    matches = difflib.get_close_matches(name, table.keys(), n=1, cutoff=0.7)
    if matches:
        return table[matches[0]]

    return None


def estimate_gi_via_claude(client: anthropic.Anthropic, ingredient_name: str) -> GIEstimate:
    """Fallback: ask Claude to estimate GI from training data when not in table."""
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        output_config={"effort": "low"},
        system=[
            {
                "type": "text",
                "text": (
                    "You estimate glycemic index (GI) values for foods that aren't in published GI tables. "
                    "Base your estimate on the food's carbohydrate composition (refined vs whole grain, "
                    "simple vs complex), fiber content, processing level, and published GI values for "
                    "similar foods. Reference Atkinson 2021 (International Tables of GI) when relevant. "
                    "Be conservative — when in doubt, lean toward the average of similar foods rather than extremes."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": f"Estimate the GI for: {ingredient_name}"}
        ],
        output_format=GIEstimate,
    )
    return response.parsed_output


class GIBatchItem(BaseModel):
    """One food's GI estimate within a batch response."""
    food_name: str = Field(description="The food name — use the exact name from the input list")
    gi: float = Field(description="Estimated glycemic index, 0-110")
    confidence: str = Field(description="low | medium | high")
    similar_foods: list[str] = Field(default_factory=list, description="Up to 3 reference foods used to estimate")


class GIBatchResult(BaseModel):
    estimates: list[GIBatchItem]


def estimate_gi_batch_via_claude(
    client: anthropic.Anthropic, food_names: list[str]
) -> dict[str, GIEstimate]:
    """Estimate GI for many foods in a single Claude call.

    Returns a dict mapping each input food name to a GIEstimate. If parsing fails
    or a name isn't returned by Claude, that name is omitted from the result.
    """
    if not food_names:
        return {}

    food_list = "\n".join(f"- {name}" for name in food_names)
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        output_config={"effort": "low"},
        system=[
            {
                "type": "text",
                "text": (
                    "You estimate glycemic index (GI) values for foods that aren't in published GI tables. "
                    "For each food in the input list, estimate its GI based on carbohydrate composition "
                    "(refined vs whole grain, simple vs complex), fiber content, processing level, and "
                    "published GI values for similar foods (Atkinson 2021 International Tables of GI). "
                    "Return one estimate per input food, using the EXACT food name as provided. "
                    "Be conservative — when in doubt, use the average of similar foods rather than extremes."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Estimate the glycemic index for each of these foods (use the exact names provided):\n\n{food_list}",
            }
        ],
        output_format=GIBatchResult,
    )

    if response.parsed_output is None:
        return {}

    estimates = response.parsed_output.estimates
    result: dict[str, GIEstimate] = {}
    for i, name in enumerate(food_names):
        match = next((e for e in estimates if e.food_name.lower() == name.lower()), None)
        if match is None and i < len(estimates):
            match = estimates[i]
        if match is not None:
            result[name] = GIEstimate(
                gi=match.gi,
                confidence=match.confidence,
                similar_foods=match.similar_foods,
                rationale="",
            )
    return result


@dataclass
class ResolvedGI:
    gi: float
    source: str  # human-readable source description
    in_table: bool


def resolve_gi(client: anthropic.Anthropic, ingredient_name: str) -> ResolvedGI:
    """Try table first, fall back to Claude. Always returns a value with its source."""
    record = lookup_in_table(ingredient_name)
    if record:
        return ResolvedGI(
            gi=record.gi,
            source=f"{record.reference} ({record.display_name})",
            in_table=True,
        )

    estimate = estimate_gi_via_claude(client, ingredient_name)
    refs = ", ".join(estimate.similar_foods[:3])
    return ResolvedGI(
        gi=estimate.gi,
        source=f"Claude estimate ({estimate.confidence} confidence; based on: {refs})",
        in_table=False,
    )
