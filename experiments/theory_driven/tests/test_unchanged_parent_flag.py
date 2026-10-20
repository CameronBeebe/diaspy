"""B flags returned_unchanged_parent only on the final failed attempt of a round."""
from experiments.theory_driven.tests.test_attempt_feedback import make_b


def test_b_fail_all_flags_only_the_final_attempt():
    proposer, *_, batch = make_b(["first", "second"], task_scores=(0.2, 0.2, 0.2))
    assert proposer({"answer": "parent"}, batch, ["answer"]) == {}
    assert [(r["attempt"], r["survived"], r["returned_unchanged_parent"]) for r in proposer.log] == [
        (1, False, False), (2, False, True)]


def test_b_success_on_later_attempt_never_flags_unchanged_parent():
    proposer, *_, batch = make_b(["first", "second"], task_scores=(0.2, 0.2, 0.8))
    assert proposer({"answer": "parent"}, batch, ["answer"]) == {"answer": "second"}
    assert [(r["attempt"], r["survived"], r["returned_unchanged_parent"]) for r in proposer.log] == [
        (1, False, False), (2, True, False)]
