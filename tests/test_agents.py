import pytest
import dspy
from diaspy.agents import ThesisAgent, AntithesisAgent, SynthesisAgent, CriticAgent
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_lm():
    with patch('dspy.settings') as mock_settings:
        mock_lm = MagicMock()
        mock_settings.lm = mock_lm
        yield mock_lm

def test_thesis_agent(mock_lm):
    agent = ThesisAgent()
    mock_lm.return_value = MagicMock(completions=[{'thesis': 'Test thesis'}])
    result = agent('Test query')
    assert result == 'Test thesis'
    mock_lm.assert_called_once()

def test_antithesis_agent(mock_lm):
    agent = AntithesisAgent()
    mock_lm.return_value = MagicMock(completions=[{'antithesis': 'Test antithesis'}])
    result = agent('Test query', 'Test thesis')
    assert result == 'Test antithesis'
    mock_lm.assert_called_once()

def test_synthesis_agent(mock_lm):
    agent = SynthesisAgent()
    mock_lm.return_value = MagicMock(completions=[{'synthesis': 'Test synthesis'}])
    result = agent('Test query', 'Test thesis', 'Test antithesis')
    assert result == 'Test synthesis'
    mock_lm.assert_called_once()

def test_critic_agent(mock_lm):
    agent = CriticAgent()
    mock_lm.return_value = MagicMock(completions=[{'critique': 'Test critique', 'score': 0.8}])
    critique, score = agent('Test query', 'Test thesis', 'Test antithesis', 'Test synthesis')
    assert critique == 'Test critique'
    assert score == 0.8
    mock_lm.assert_called_once()
