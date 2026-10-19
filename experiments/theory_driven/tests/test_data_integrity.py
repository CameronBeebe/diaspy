"""Fixture splits must be deterministic, disjoint, and keep gold out of inputs."""
from pathlib import Path
from experiments.theory_driven.benchmarks import HotpotQABenchmark


def test_hotpot_fixture_splits():
    fixture = Path(__file__).parent / "fixtures" / "hotpotqa.json"
    benchmark = HotpotQABenchmark(fixture_path=fixture)
    first = benchmark.load_data(seed=0, train_size=1, val_size=1, test_size=1)
    repeated = benchmark.load_data(seed=0, train_size=1, val_size=1, test_size=1)
    other = benchmark.load_data(seed=1, train_size=1, val_size=1, test_size=1)
    def questions(splits):
        return [[e.question for e in split] for split in splits]
    assert questions(first) == questions(repeated)
    assert questions(first) != questions(other)
    sets = [set(e.question for e in split) for split in first]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    for split in first:
        for example in split:
            assert "answer" not in example.inputs()
            assert "supporting_facts" not in example.inputs()
