# llm.py
from typing import Literal, Optional, Tuple
import ast
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================
# Types
# ======================

IntentType = Literal[
    "get_events",
    "get_goals",
    "get_highlights",
]

CompetitionType = Literal[
    "spain_laliga",
    "italy_serie-a",
    "germany_bundesliga",
    "france_ligue-1",
    "europe_uefa-champions-league",
    "england_epl",
]

IntentTuple = Tuple[
    str,
    str,
    IntentType,
    CompetitionType,
    Optional[str],
    Optional[str],
]


# ======================
# Prompt (statique, safe)
# ======================

PROMPT_TEMPLATE = """
You are a deterministic parser.

Return ONLY a Python tuple with this exact format:
(team_a, team_b, intent, competition, season, date)

Rules:
- exactly 2 teams, sorted alphabetically
- intent ∈ ("get_events","get_goals","get_highlights")
- competition ∈ (
  "spain_laliga",
  "italy_serie-a",
  "germany_bundesliga",
  "france_ligue-1",
  "europe_uefa-champions-league",
  "england_epl"
)
- season format "YYYY-YYYY"
- date format "YYYY-MM-DD" or None
- NO explanation, NO text, ONLY the tuple

User query:
{query}

Output:
"""


# ======================
# LLM Local Parser
# ======================

class LocalLLMParser:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cuda",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.device = device

    def parse(self, query: str) -> IntentTuple:
        prompt = PROMPT_TEMPLATE.format(query=query)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
            )

        decoded = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        start = decoded.find("(")
        end = decoded.find(")", start)

        if start == -1 or end == -1:
            raise ValueError(f"No tuple found in LLM output: {decoded}")

        decoded = decoded[start : end + 1]


        try:
            parsed = ast.literal_eval(decoded)
        except Exception as e:
            raise ValueError(f"Invalid tuple from LLM: {decoded}") from e

        self._validate(parsed)
        return parsed

    def _validate(self, t: tuple) -> None:
        if not isinstance(t, tuple) or len(t) != 6:
            raise ValueError("Output must be a 6-element tuple")

        team_a, team_b, intent, competition, season, date = t

        if team_a > team_b:
            raise ValueError("Teams must be alphabetically sorted")

        if intent not in ("get_events", "get_goals", "get_highlights"):
            raise ValueError("Invalid intent")

        if competition not in (
            "spain_laliga",
            "italy_serie-a",
            "germany_bundesliga",
            "france_ligue-1",
            "europe_uefa-champions-league",
            "england_epl",
        ):
            raise ValueError("Invalid competition")

if __name__ == "__main__":
    parser = LocalLLMParser(device="cpu")

    result = parser.parse(
        "Salut, je veux voir les buts du match Manchester United Arsenal du 8 mars 2015"
    )

    print(result)
