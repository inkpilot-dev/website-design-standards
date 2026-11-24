"""Topics the teacher is asked to write new questions about.

These cover reasoning that the procedural generators cannot produce: everyday,
causal, scientific and verbal reasoning. They are briefs for the teacher, not
training text; the questions and the answers both come from the teacher.
"""
from __future__ import annotations

import json
import re

from .verify import number_text, numeric_value

TOPICS = [
    ("commonsense", "physical_commonsense", "everyday physical situations (objects, containers, weight, heat, motion) where the outcome must be worked out from the facts given"),
    ("commonsense", "social_commonsense", "everyday social situations where a person's likely knowledge or belief must be inferred from what they saw and did not see"),
    ("commonsense", "practical_planning", "practical planning under constraints such as time, budget, capacity or opening hours"),
    ("causal", "cause_and_effect", "chains of cause and effect in a described system, asking what happens after a change"),
    ("causal", "counterfactuals", "counterfactual questions about a described situation: what would have happened had one fact been different"),
    ("causal", "experiment_design", "small described experiments, asking which conclusion the results support or which variable was not controlled"),
    ("science", "physics_reasoning", "school-level physics reasoning (speed, density, levers, circuits, energy) with all needed values given"),
    ("science", "chemistry_reasoning", "school-level chemistry reasoning (ratios, concentrations, dilution, balancing simple quantities) with all needed values given"),
    ("science", "biology_reasoning", "inheritance, population or food-web reasoning from rules stated in the question"),
    ("science", "units_and_estimation", "unit conversion and proportional reasoning across several steps, with every conversion factor given"),
    ("maths", "everyday_finance", "money problems: interest, budgets, discounts, splitting bills, exchange at a stated rate"),
    ("maths", "ratio_and_proportion", "ratio, proportion and mixture problems that need several steps"),
    ("maths", "statistics", "mean, median, mode, range and weighted averages of small data sets given in the question"),
    ("maths", "probability_stories", "probability in story form with small, fully specified sample spaces"),
    ("maths", "algebra_stories", "word problems that reduce to one or two equations"),
    ("maths", "geometry_stories", "geometry word problems about lengths, areas, angles or volumes with all dimensions given"),
    ("maths", "work_and_motion", "combined work, travel and meeting-point problems"),
    ("logic", "conditional_reasoning", "if-then rules applied to a described case, including contrapositives and invalid inferences"),
    ("logic", "argument_flaws", "short arguments where the reader must pick which option names the flaw in the reasoning"),
    ("logic", "necessary_sufficient", "situations that turn on the difference between necessary and sufficient conditions"),
    ("logic", "set_reasoning", "membership and category reasoning with stated rules, using invented categories"),
    ("logic", "seating_and_ranking", "small arrangement or ranking puzzles with enough clues for one solution"),
    ("logic", "rule_following", "applying a stated multi-clause policy or rulebook to a specific case"),
    ("verbal", "reading_inference", "a short original passage of three to five sentences followed by a question whose answer must be inferred, not copied"),
    ("verbal", "analogies", "analogies between relationships, asking which option completes the pattern"),
    ("verbal", "word_logic", "letter and word puzzles: anagram counts, alphabetical order, letter positions, code words under a stated rule"),
    ("verbal", "instructions", "following a sequence of written instructions that transform a list, a word or a number"),
    ("code", "code_output", "a short Python snippet of 5 to 12 lines, asking what it prints"),
    ("code", "bug_finding", "a short Python function with one described intended behaviour, asking which option identifies the bug"),
    ("code", "algorithm_tracing", "tracing a named simple algorithm (binary search, bubble sort, stack or queue operations) on given data and reporting a count or final state"),
    ("code", "complexity_reasoning", "counting how many times a described loop or operation runs for given sizes"),
    ("data", "table_questions", "a small table given as text, with a question that needs several lookups and a calculation"),
    ("data", "schedules_and_timetables", "timetables or calendars given as text, asking for a duration, a clash or the earliest possible time"),
    ("data", "chart_descriptions", "a described chart or series of readings, asking for a trend comparison that needs calculation"),
    ("strategy", "game_reasoning", "simple two-player games with stated rules, asking for the winning move or the winner under best play"),
    ("strategy", "optimisation", "small optimisation problems: the cheapest, fastest or largest option under stated constraints"),
    ("strategy", "resource_allocation", "dividing limited resources among competing needs under stated rules"),
    ("spatial", "spatial_reasoning", "positions, directions, rotations and reflections of described objects"),
    ("spatial", "paper_folding_and_nets", "folding, cutting and stacking described step by step, asking for a count or a final arrangement"),
    ("temporal", "time_reasoning", "ordering events and working out durations, ages and dates from stated relations"),
]

LEVELS = {1: ("easy", "2"), 2: ("moderate", "3"), 3: ("medium", "4"), 4: ("hard", "5 or 6"),
          5: ("very hard", "7 or more")}
ANSWER_TYPES = ("number", "choice", "yesno", "text")

PROMPT = """Write {n} new reasoning questions for a training set.

Topic: {brief}.
Difficulty: {level}. Solving each one should take about {steps} distinct reasoning steps.

Every question must:
- be self-contained, with all the facts it needs, and require no specialist or obscure knowledge;
- need reasoning, not recall;
- have exactly one correct answer that is short and can be checked;
- differ from the others in setting, names and structure.{avoid}

Use one of these answer types for each question:
- "number": the answer is a single number with no units;
- "choice": the question lists four options labelled A) to D) and the answer is one letter;
- "yesno": the answer is Yes or No;
- "text": the answer is one word or a short name.

Reply with only a JSON array in this form, and nothing else:
[{{"question": "...", "answer": "...", "answer_type": "..."}}]"""


def author_prompt(brief: str, difficulty: int, n: int, recent: list[str]) -> str:
    level, steps = LEVELS[difficulty]
    avoid = ""
    if recent:
        avoid = "\n- not resemble these existing questions: " + " | ".join(q[:90] for q in recent[-4:])
    return PROMPT.format(n=n, brief=brief, level=level, steps=steps, avoid=avoid)


def parse_authored(content: str) -> list[dict]:
    """Questions from the teacher's reply. Anything malformed is dropped, never repaired."""
    start, end = content.find("["), content.rfind("]")
    items = None
    if 0 <= start < end:
        try:
            items = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            items = None
    if items is None:  # fall back to object-by-object so one bad item does not lose the batch
        items = []
        for m in re.finditer(r"\{[^{}]*\}", content):
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        q, a, t = it.get("question"), it.get("answer"), it.get("answer_type")
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            a = str(a)
        if not (isinstance(q, str) and isinstance(a, str) and t in ANSWER_TYPES):
            continue
        q, a = q.strip(), a.strip()
        if len(q) < 40 or not a or len(a) > 60:
            continue
        if t == "number":
            value = numeric_value(a)
            if value is None:
                continue
            a = number_text(value)  # "2.5 cups" and "25%" become "2.5" and "25"
        if t == "yesno" and a.lower().rstrip(".") not in ("yes", "no"):
            continue
        if t == "choice" and not (re.fullmatch(r"\(?[A-Da-d]\)?", a) and all(
                re.search(rf"(?m)(?:^|\s)\(?{c}[).:]", q) for c in "ABCD")):
            continue
        out.append({"question": q, "answer": a, "answer_type": t})
    return out
