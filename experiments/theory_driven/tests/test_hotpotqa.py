"""Offline regression tests using attributed, real HotpotQA examples."""

import json
from pathlib import Path
import random

import dspy
import pytest
from dspy.utils import DummyLM

from experiments.theory_driven import benchmarks

FIXTURE = Path(__file__).parent / "fixtures" / "hotpotqa.json"


def test_hotpotqa_export_and_answer_f1_feedback():
    benchmark_type = getattr(benchmarks, "HotpotQABenchmark", None)
    assert benchmark_type is not None, "HotpotQABenchmark must be exported"
    benchmark = benchmark_type("hotpotqa")
    metric = benchmark.get_metric()
    gold = dspy.Example(
        answer="The New York City", supporting_titles=["New York City", "History"]
    )
    pred = dspy.Prediction(answer="new york", retrieved_titles=["New York City"])
    result = metric(gold, pred, None, "answer", None)
    assert isinstance(result, dspy.Prediction)
    assert result.score == pytest.approx(0.8)
    assert "The New York City" in result.feedback
    assert "History" in result.feedback
    assert "HotpotQABenchmark" in benchmarks.__all__


def test_real_fixture_splits_are_deterministic_disjoint_and_input_only():
    benchmark = benchmarks.HotpotQABenchmark(fixture_path=FIXTURE)
    state = random.getstate()
    splits = benchmark.load_data(seed=17, train_size=1, val_size=1, test_size=2)
    ids = [[example.id for example in split] for split in splits]
    assert [len(split) for split in splits] == [1, 1, 2]
    assert len({identifier for split in ids for identifier in split}) == 4
    assert ids == [
        [ex.id for ex in split] for split in benchmark.load_data(17, 1, 1, 2)
    ]
    assert ids != [
        [ex.id for ex in split] for split in benchmark.load_data(18, 1, 1, 2)
    ]
    assert random.getstate() == state
    assert benchmark.setup()[0] is True
    for split in splits:
        for example in split:
            assert set(example.inputs().keys()) == {"question", "paragraphs"}
            assert len(example.paragraphs) == 10
            assert example.supporting_titles
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["attribution"]["license"] == "CC BY-SA 4.0"
    assert {ex.id for split in splits for ex in split} == {
        row["id"] for row in fixture["rows"]
    }


def test_program_retrieves_only_current_question_paragraphs_then_answers():
    benchmark = benchmarks.HotpotQABenchmark(fixture_path=FIXTURE)
    examples = [ex for split in benchmark.load_data(0, 1, 1, 2) for ex in split]
    example = next(ex for ex in examples if ex.id == "5a8b57f25542995d1e6f1371")
    program = benchmark.get_program()
    lm = DummyLM([{"answer": "yes"}, {"answer": "sentinel"}])
    with dspy.context(lm=lm):
        prediction = program(**example.inputs())
        isolated = program(
            question="unique sentinel",
            paragraphs=[{"title": "Only local", "text": "unique sentinel"}],
        )
    assert prediction.answer == "yes"
    assert len(prediction.retrieved_titles) == 4
    assert set(prediction.retrieved_titles) <= {p["title"] for p in example.paragraphs}
    assert "Scott Derrickson" in prediction.retrieved_titles
    assert isolated.retrieved_titles == ["Only local"]
    assert len(lm.history) == 2
    first_prompt = str(lm.history[0]["messages"])
    assert "Scott Derrickson" in first_prompt
    assert "American director" in first_prompt
    second_prompt = str(lm.history[1]["messages"])
    assert "Only local" in second_prompt
    assert "Scott Derrickson" not in second_prompt


def test_dataset_subset_cached_and_reused_without_network(tmp_path, monkeypatch):
    import datasets

    rows = json.loads(FIXTURE.read_text())["rows"]
    calls = []

    def load_dataset(path, name, *, split):
        calls.append((path, name, split))
        return rows

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    benchmark = benchmarks.HotpotQABenchmark(cache_dir=tmp_path)
    first = benchmark.load_data(7, 1, 1, 1)
    assert calls == [("hotpotqa/hotpot_qa", "distractor", "validation")]
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    assert len(json.loads(cache_files[0].read_text())["rows"]) == 3

    def offline(*args, **kwargs):
        pytest.fail("Cached subset must not download data")

    monkeypatch.setattr(datasets, "load_dataset", offline)
    second = benchmarks.HotpotQABenchmark(cache_dir=tmp_path).load_data(7, 1, 1, 1)
    assert [[ex.id for ex in s] for s in first] == [[ex.id for ex in s] for s in second]


@pytest.mark.parametrize("sizes", [(-1, 1, 1), (2, 2, 2)])
def test_invalid_or_insufficient_splits_fail_instead_of_silent_truncation(sizes):
    benchmark = benchmarks.HotpotQABenchmark(fixture_path=FIXTURE)
    with pytest.raises(ValueError):
        benchmark.load_data(0, *sizes)


@pytest.mark.parametrize(
    "benchmark_type,name",
    [(benchmarks.HoVerBenchmark, "hover"), (benchmarks.PUPABenchmark, "pupa")],
)
def test_pilot_benchmarks_share_interface_but_cannot_run_fake_data(
    benchmark_type, name
):
    benchmark = benchmark_type(name)
    assert isinstance(benchmark, benchmarks.BaseBenchmark)
    ok, message = benchmark.setup()
    assert ok is False
    assert "pilot" in message.lower()
    with pytest.raises(NotImplementedError, match="pilot"):
        benchmark.load_data(seed=7, train_size=1, val_size=1, test_size=1)
    gold = dspy.Example(answer="gold", gold_passages=["A", "B"])
    pred = dspy.Prediction(answer="gold", retrieved_passages=["A"])
    result = benchmark.get_metric()(gold, pred, None, "qa", None)
    assert isinstance(result.score, float)
    assert isinstance(result.feedback, str) and result.feedback


def test_hotpotqa_implements_the_common_base_interface():
    assert isinstance(benchmarks.HotpotQABenchmark(), benchmarks.BaseBenchmark)


def test_download_failure_never_uses_fixture_fallback(tmp_path, monkeypatch):
    import datasets

    def unavailable(*args, **kwargs):
        raise OSError("dataset unavailable")

    monkeypatch.setattr(datasets, "load_dataset", unavailable)
    with pytest.raises(OSError, match="dataset unavailable"):
        benchmarks.HotpotQABenchmark(cache_dir=tmp_path).load_data(0, 1, 1, 1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "gold,answer,score",
    [
        ("The Paris!", "paris", 1.0),
        ("red red blue", "red blue blue", 2 / 3),
        ("yes", "no", 0.0),
        ("Tokyo", "", 0.0),
        ("", "", 1.0),
    ],
)
def test_squad_f1_known_answers(gold, answer, score):
    result = benchmarks.HotpotQABenchmark().get_metric()(
        dspy.Example(answer=gold), dspy.Prediction(answer=answer)
    )
    assert result.score == pytest.approx(score)
    assert result.feedback
