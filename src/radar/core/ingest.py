import httpx
from bs4 import BeautifulSoup
from radar.db.models import Signal
from datetime import datetime
import logging
from google import genai
from radar.config import settings
from typing import List, Tuple
from radar.core.models import KnowledgeGraphExtraction

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def get_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for the given text."""
        result = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text,
        )
        return result.embeddings[0].values

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts."""
        if not texts:
            return []

        # We process embeddings sequentially or concurrently here
        # Optimizing with `batch_embed_contents` is better but we use a loop for type safety for now

        embeddings = []
        for t in texts:
            result = self.client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=t,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    async def extract_knowledge(self, text: str) -> KnowledgeGraphExtraction:
        """Extract entities, relationships, and trends from text using structured generation."""
        prompt = """
        Analyze the following text and extract a Knowledge Graph.
        
        1. **Entities:** Identify key players (Companies, People) and Technologies.
        2. **Connections:** Map the relationships between them (who is competing with whom, who supports what).
        3. **Trends:** Identify broader market trends or patterns (e.g., "AI Consolidation", "Green Tech Surge"). Estimate their velocity.
        
        Text:
        {text}
        """

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt.format(text=text),
            config={
                "response_mime_type": "application/json",
                "response_schema": KnowledgeGraphExtraction,
            },
        )

        try:
            # The response.text should be JSON conforming to the schema
            return KnowledgeGraphExtraction.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Failed to parse knowledge extraction: {e}")
            return KnowledgeGraphExtraction(entities=[], connections=[], trends=[])

    async def answer_question(
        self, question: str, context_signals: List[Signal]
    ) -> str:
        """Synthesize an answer using RAG."""
        context_text = "\n\n".join(
            [f"--- Signal: {s.title} ---\n{s.content}" for s in context_signals]
        )

        prompt = f"""
You are RADAR, an Industry Intelligence Brain. Use the following ingested signals to answer the user's question.
If the answer is not in the context, say you don't know based on current signals.

CONTEXT:
{context_text}

QUESTION: {question}
"""
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text


class WebIngestAgent:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.intel = IntelligenceAgent()

    async def fetch(self, url: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, headers=self.headers, follow_redirects=True
                )
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                logger.error(f"HTTP Error fetching {url}: {e}")
                raise

    async def parse(
        self, html: str, url: str
    ) -> Tuple[Signal, KnowledgeGraphExtraction]:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string if soup.title else "No Title"

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        content = text[:5000]

        # Generate vector embedding
        vector = await self.intel.get_embedding(content)

        # Extract Knowledge Graph
        kg = await self.intel.extract_knowledge(content)

        signal = Signal(
            title=title,
            url=url,
            content=content,
            raw_text=text,
            date=datetime.now(),
            source="web",
            vector=vector,
        )
        return signal, kg

    async def ingest(self, url: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        logger.info(f"Ingesting {url}")
        html = await self.fetch(url)
        signal, kg = await self.parse(html, url)
        logger.info(f"Parsed signal: {signal.title}")
        return signal, kg


class TextIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()

    async def parse(
        self, text: str, title: str = "Raw Input"
    ) -> Tuple[Signal, KnowledgeGraphExtraction]:
        # Generate a generic title if not provided or just use the first line
        if title == "Raw Input" and text.strip():
            first_line = text.strip().split("\n")[0][:50]
            if first_line:
                title = first_line

        content = text
        # Generate vector embedding
        vector = await self.intel.get_embedding(content[:5000])

        # Extract Knowledge Graph
        kg = await self.intel.extract_knowledge(content[:5000])

        signal = Signal(
            title=title,
            content=content,
            raw_text=text,
            date=datetime.now(),
            source="stdin",
            vector=vector,
        )
        return signal, kg

    async def ingest(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        logger.info("Ingesting raw text from stdin")
        signal, kg = await self.parse(text)
        logger.info(f"Parsed signal: {signal.title}")
        return signal, kg
