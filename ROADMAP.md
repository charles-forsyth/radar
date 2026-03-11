# Radar Roadmap: Future Feature Improvements

This document tracks planned and proposed enhancements for the Radar Personal Industry Intelligence Brain.

## 📡 Radar Live: SIGINT Tactical HUD (High Priority)
Transform Radar from a strategic research brain into a real-time tactical dashboard for signal intelligence.

- **ADS-B Integration:** Real-time tracking of aircraft overhead using `dump1090-fa`/`readsb` data.
- **APRS-IS Listener:** Live stream of HAM radio packets (position reports, weather) within a 100km radius of Tioga County.
- **Meshtastic Hub:** Real-time mesh status monitoring, node tracking, and channel message logging via the Meshtastic Python API.
- **WebSDR & Frequency Hunter:** Automated lookup of local emergency frequencies (P25/Analog) with direct links to the nearest public WebSDR receivers.

## 🤖 Enhanced Research Agents
- **Local Web Crawler:** Build a custom Rust-based crawler that scrapes technical technical wires and search results without external AI APIs.
- **Multi-Agent Swarms:** Parallelize research tasks across multiple specialized local agents (e.g., one for technical architecture, one for market finance).
- **Source Verification:** Cross-reference extracted claims across multiple independent search results to assign confidence scores locally.

## 📊 Knowledge Graph & UI
- **Temporal Analysis:** Visualize how trends and entity relationships have evolved over multiple intelligence sweeps.
- **Automated Alerts:** Trigger notifications (Voice/Email) when a new signal significantly changes the competitive landscape for a high-priority interest.
- **Interactive TUI Filters:** Add keyboard navigation to the dashboard to filter signals and entities by category (e.g., just show SIGINT/RF signals).

## ⚙️ Core Infrastructure
- **Local Hashing Embeddings:** Refine the custom C-based hashing embedding engine to support higher dimensions and better collision resistance.
- **Semantic Deduplication:** Automatically merge highly similar signals or entities using local string-distance algorithms (Levenshtein/Jaro-Winkler).
- **Extractive Summarization:** Enhance the local TextRank engine to handle multi-document summarization for long-term intelligence briefing.
