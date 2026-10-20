import dspy
from dspy.utils import DummyLM

from diaspy.agents import ThesisAgent, AntithesisAgent, SynthesisAgent, CriticAgent


def assert_single_request(lm, **inputs):
    assert len(lm.history) == 1
    messages = lm.history[0]["messages"]
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    assert len(user_messages) == 1
    for field, value in inputs.items():
        assert f"[[ ## {field} ## ]]\n{value}\n" in user_messages[0]


def test_thesis_agent():
    adapter = dspy.JSONAdapter()
    lm = DummyLM(
        [{"reasoning": "Consider the query.", "thesis": "Test thesis"}],
        adapter=adapter,
    )
    with dspy.context(lm=lm, adapter=adapter):
        result = ThesisAgent()("Test query")

    assert result == "Test thesis"
    assert_single_request(lm, query="Test query")


def test_antithesis_agent():
    adapter = dspy.JSONAdapter()
    lm = DummyLM(
        [{"reasoning": "Challenge the thesis.", "antithesis": "Test antithesis"}],
        adapter=adapter,
    )
    with dspy.context(lm=lm, adapter=adapter):
        result = AntithesisAgent()("Test query", "Test thesis")

    assert result == "Test antithesis"
    assert_single_request(lm, query="Test query", thesis="Test thesis")


def test_synthesis_agent():
    adapter = dspy.JSONAdapter()
    lm = DummyLM(
        [{"reasoning": "Reconcile the positions.", "synthesis": "Test synthesis"}],
        adapter=adapter,
    )
    with dspy.context(lm=lm, adapter=adapter):
        result = SynthesisAgent()("Test query", "Test thesis", "Test antithesis")

    assert result == "Test synthesis"
    assert_single_request(
        lm, query="Test query", thesis="Test thesis", antithesis="Test antithesis"
    )


def test_critic_agent():
    adapter = dspy.JSONAdapter()
    lm = DummyLM(
        [{"reasoning": "Evaluate the synthesis.", "critique": "Test critique", "score": 0.8}],
        adapter=adapter,
    )
    with dspy.context(lm=lm, adapter=adapter):
        critique, score = CriticAgent()(
            "Test query", "Test thesis", "Test antithesis", "Test synthesis"
        )

    assert critique == "Test critique"
    assert score == 0.8
    assert_single_request(
        lm,
        query="Test query",
        thesis="Test thesis",
        antithesis="Test antithesis",
        synthesis="Test synthesis",
    )
