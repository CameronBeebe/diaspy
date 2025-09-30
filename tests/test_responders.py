import pytest
import dspy
from diaspy.responders import DialecticResponder
from diaspy.agents import ThesisAgent, AntithesisAgent, SynthesisAgent, CriticAgent
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_agents():
    return {
        'thesis': MagicMock(return_value='Mock thesis'),
        'antithesis': MagicMock(return_value='Mock antithesis'),
        'synthesis': MagicMock(return_value='Mock synthesis'),
        'critic': MagicMock(return_value=('Mock critique', 0.9)),
        'pro_debate': MagicMock(return_value='Mock pro'),
        'con_debate': MagicMock(return_value='Mock con'),
        'expert': MagicMock(return_value='Mock opinion')
    }

def test_dialectic_responder_binary(mock_agents):
    responder = DialecticResponder(**mock_agents)
    prediction = responder('Test query', mode='binary')
    assert hasattr(prediction, 'thesis')
    assert hasattr(prediction, 'antithesis')
    assert hasattr(prediction, 'synthesis')
    assert hasattr(prediction, 'critiques')

def test_dialectic_responder_debate(mock_agents):
    responder = DialecticResponder(**mock_agents)
    prediction = responder('Test query', mode='debate')
    assert hasattr(prediction, 'debate_history')
    assert hasattr(prediction, 'synthesis')

def test_dialectic_responder_experts(mock_agents):
    responder = DialecticResponder(**mock_agents)
    prediction = responder('Test query', mode='experts', domains=['test'])
    assert hasattr(prediction, 'expert_opinions')
    assert hasattr(prediction, 'synthesis')
