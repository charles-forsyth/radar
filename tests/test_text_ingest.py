import pytest
from radar.core.ingest import TextIngestAgent
from radar.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.asyncio
async def test_text_ingest_agent():
    agent = TextIngestAgent()
    text = "This is a test signal.\nIt has multiple lines."
    signal = await agent.ingest(text)

    assert signal.title == "This is a test signal."
    assert signal.content == text
    assert signal.source == "stdin"


def test_ingest_command_stdin():
    # Simulate stdin input
    result = runner.invoke(app, ["ingest"], input="Stdin content here")
    assert result.exit_code == 0
    assert "ingested:" in result.stdout
    assert "Stdin content here" in result.stdout or "Length" in result.stdout


def test_ingest_command_hyphen():
    # Simulate 'radar ingest -'
    result = runner.invoke(app, ["ingest", "-"], input="Hyphen content")
    assert result.exit_code == 0
    assert "ingested:" in result.stdout
    assert "Hyphen content" in result.stdout or "Length" in result.stdout


def test_ingest_command_empty():
    result = runner.invoke(app, ["ingest"], input="")
    assert result.exit_code == 1
    assert "Error" in result.stdout
