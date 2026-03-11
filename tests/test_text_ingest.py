from radar.core.ingest import TextIngestAgent
from radar.main import app
from typer.testing import CliRunner
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

runner = CliRunner()


@pytest.fixture
def mock_db_session():
    with patch("radar.db.engine.async_session") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session_cls.return_value = mock_session
        # Mock context manager
        mock_session.__aenter__.return_value = mock_session
        mock_session.__aexit__.return_value = None

        # Mock execute result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = (
            None  # Assume no existing entities
        )
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        yield mock_session


@pytest.mark.asyncio
async def test_text_ingest_agent():
    agent = TextIngestAgent()
    text = "This is a test signal.\nIt has multiple lines."
    signal, kg = await agent.ingest(text)

    assert signal.title == "This is a test signal."
    assert signal.content == text
    assert signal.source == "local_nlp"


@patch("radar.main.run_ingest", new_callable=AsyncMock)
def test_ingest_command_stdin(mock_run_ingest, mock_db_session):
    # Simulate stdin input
    result = runner.invoke(app, ["ingest"], input="Stdin content here")
    if result.exit_code != 0:
        print(result.stdout)
        print(result.exception)
    assert result.exit_code == 0
    mock_run_ingest.assert_called_once()
    assert "Stdin content here" in mock_run_ingest.call_args[0][0]


@patch("radar.main.run_ingest", new_callable=AsyncMock)
def test_ingest_command_hyphen(mock_run_ingest, mock_db_session):
    # Simulate 'radar ingest -'
    result = runner.invoke(app, ["ingest", "-"], input="Hyphen content")
    if result.exit_code != 0:
        print(result.stdout)
    assert result.exit_code == 0
    mock_run_ingest.assert_called_once()
    assert "Hyphen content" in mock_run_ingest.call_args[0][0]


def test_ingest_command_empty():
    result = runner.invoke(app, ["ingest"], input="")
    assert result.exit_code == 1
    assert "Error" in result.stdout
