"""HotpotQA distractor: local paragraph retrieval and answer-token F1."""

from collections import Counter
import json
from pathlib import Path
import random
import re
import string

import dspy
from rank_bm25 import BM25Okapi

from experiments.theory_driven.benchmarks.base import BaseBenchmark, Splits


class HotpotQAAnswer(dspy.Signature):
    """Answer the question using the supplied context. In answer, return only
    the minimal short answer (a name, date, number, or yes/no as appropriate);
    no explanations, reasoning, citations, or restatement of the question.
    """

    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class HotpotQAProgram(dspy.Module):
    """BM25 over supplied paragraphs only, followed by one answer predictor."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(HotpotQAAnswer)

    def forward(self, question: str, paragraphs: list[dict[str, str]]):
        tokens = [
            re.findall(r"\w+", (p["title"] + " " + p["text"]).lower()) or [""]
            for p in paragraphs
        ]
        if tokens:
            scores = BM25Okapi(tokens).get_scores(re.findall(r"\w+", question.lower()))
            indices = sorted(
                range(len(paragraphs)), key=lambda i: (-float(scores[i]), i)
            )[:4]
        else:
            indices = []
        selected = [paragraphs[i] for i in indices]
        context = "\n\n".join(f"{p['title']}: {p['text']}" for p in selected)
        result = self.answer(context=context, question=question)
        return dspy.Prediction(
            answer=result.answer, retrieved_titles=[p["title"] for p in selected]
        )


def normalize_answer(answer: str) -> str:
    """Use the standard SQuAD lowercase/punctuation/article normalization."""
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_f1(gold: str, prediction: str) -> float:
    expected = normalize_answer(gold).split()
    actual = normalize_answer(prediction).split()
    if not expected or not actual:
        return float(expected == actual)
    common = sum((Counter(expected) & Counter(actual)).values())
    return 2.0 * common / (len(expected) + len(actual))


def hotpotqa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Return answer F1 and feedback suitable for GEPA reflection."""
    score = answer_f1(gold.answer, getattr(pred, "answer", ""))
    retrieved = getattr(pred, "retrieved_titles", None)
    supporting = getattr(gold, "supporting_titles", None)
    if retrieved is None or supporting is None:
        retrieval_feedback = "Support coverage unknown: retrieval titles or support annotations unavailable."
    else:
        missing = [title for title in supporting if title not in set(retrieved)]
        retrieval_feedback = (
            f"Missed supporting-fact titles: {', '.join(missing)}. "
            "Potential retrieval limitation, not proof of an instruction defect."
            if missing else
            "Missed supporting-fact titles: none. No annotated support titles missing."
        )
    return dspy.Prediction(
        score=score,
        feedback=f"Gold answer: {gold.answer}. Answer F1: {score:.3f}. "
        "Metric: normalized answer token-F1 (harmonic mean of token precision and recall; "
        "lowercase and remove punctuation and articles). Extra explanation tokens can "
        "lower precision even when the short answer is included. Token overlap is not a "
        "semantic correctness judgment, and a low score alone does not establish a "
        "reasoning failure. Task contract: return only the minimal short answer in answer, "
        "with no explanations or citations. "
        f"{retrieval_feedback} Title coverage does not establish evidence sufficiency; "
        "inspect the actual context for missing facts or ambiguity before attributing "
        "the mismatch to the instruction.",
    )


class HotpotQABenchmark(BaseBenchmark):
    """HotpotQA distractor, CC BY-SA 4.0 (https://hotpotqa.github.io/).

    Production loads HF hotpotqa/hotpot_qa, distractor, validation and caches
    only the seeded subset. fixture_path is explicit test-only injection;
    download failures propagate and never fall back to the checked-in fixture.
    Examples expose question and paragraphs (list of title/text dicts) as inputs;
    answer and supporting_titles are gold-only fields, never program inputs.
    """

    def __init__(
        self,
        name: str = "hotpotqa",
        *,
        cache_dir: str | Path | None = None,
        fixture_path: str | Path | None = None,
    ):
        super().__init__(name)
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "diaspy" / "hotpotqa"
        )
        self.fixture_path = Path(fixture_path) if fixture_path is not None else None

    def setup(self) -> tuple[bool, str]:
        return True, "HotpotQA uses per-question local BM25; no global index required."

    def load_data(
        self,
        seed: int = 0,
        train_size: int = 20,
        val_size: int = 20,
        test_size: int = 50,
    ) -> Splits:
        if min(train_size, val_size, test_size) < 0:
            raise ValueError("Split sizes must be non-negative")
        count = train_size + val_size + test_size
        if self.fixture_path is not None:
            rows = json.loads(self.fixture_path.read_text())["rows"]
            random.Random(seed).shuffle(rows)
            rows = rows[:count]
        else:
            cache_path = (
                self.cache_dir
                / f"validation-v1-seed{seed}-{train_size}-{val_size}-{test_size}.json"
            )
            if cache_path.exists():
                rows = json.loads(cache_path.read_text())["rows"]
            else:
                from datasets import load_dataset

                dataset = load_dataset(
                    "hotpotqa/hotpot_qa", "distractor", split="validation"
                )
                indices = list(range(len(dataset)))
                random.Random(seed).shuffle(indices)
                rows = [dataset[index] for index in indices[:count]]
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "dataset": "hotpotqa/hotpot_qa",
                            "config": "distractor",
                            "split": "validation",
                            "license": "CC BY-SA 4.0",
                            "seed": seed,
                            "rows": rows,
                        }
                    )
                    + "\n"
                )
        if len(rows) != count:
            raise ValueError(
                f"Requested {count} examples, but data source contains {len(rows)}"
            )
        examples = [
            dspy.Example(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                paragraphs=[
                    {"title": title, "text": "".join(sentences)}
                    for title, sentences in zip(
                        row["context"]["title"],
                        row["context"]["sentences"],
                        strict=True,
                    )
                ],
                supporting_titles=list(dict.fromkeys(row["supporting_facts"]["title"])),
            ).with_inputs("question", "paragraphs")
            for row in rows[: train_size + val_size + test_size]
        ]
        return (
            examples[:train_size],
            examples[train_size : train_size + val_size],
            examples[train_size + val_size :],
        )

    def get_program(self) -> dspy.Module:
        return HotpotQAProgram()

    def get_metric(self):
        return hotpotqa_metric
