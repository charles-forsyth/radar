# Radar Roadmap: Future Feature Improvements

This document tracks planned and proposed enhancements for the Radar Personal Industry Intelligence Brain.

## 📡 Radar Live: SIGINT Tactical HUD (High Priority)
Transform Radar from a strategic research brain into a real-time tactical dashboard for signal intelligence.

- **ADS-B Integration:** Real-time tracking of aircraft overhead using `dump1090-fa`/`readsb` data.
- **APRS-IS Listener:** Live stream of HAM radio packets (position reports, weather) within a 100km radius of Tioga County.
- **Meshtastic Hub:** Real-time mesh status monitoring, node tracking, and channel message logging via the Meshtastic Python API.
- **WebSDR & Frequency Hunter:** Automated lookup of local emergency frequencies (P25/Analog) with direct links to the nearest public WebSDR receivers.

## 🤖 Enhanced Research Agents
- **Deep Research Access:** Transition from the Search Grounding fallback to the native Gemini Deep Research Interactions API once access is fully provisioned.
- **Multi-Agent Swarms:** Parallelize research tasks across multiple specialized agents (e.g., one for technical architecture, one for market finance).
- **Source Verification:** Cross-reference extracted claims across multiple independent search results to assign confidence scores.

## 📊 Knowledge Graph & UI
- **Temporal Analysis:** Visualize how trends and entity relationships have evolved over multiple intelligence sweeps.
- **Automated Alerts:** Trigger notifications (Voice/Email) when a new signal significantly changes the competitive landscape for a high-priority interest.
- **Interactive TUI Filters:** Add keyboard navigation to the dashboard to filter signals and entities by category (e.g., just show SIGINT/RF signals).

## ⚙️ Core Infrastructure
- **Matryoshka Embedding Tuning:** Optimize retrieval by experimenting with different dimensions (from 256 to 3072) supported by the `gemini-embedding-001` model.
- **Semantic Deduplication:** Automatically merge highly similar signals or entities to maintain a clean knowledge graph.
- **Offline Mode:** Implement local caching for LLM responses to enable basic querying of the graph without an active API connection.
