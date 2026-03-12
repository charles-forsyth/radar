# Radar v0.14.0: The "Sovereign Intelligence" Update

This plan outlines the integration of four major, 100% offline capabilities into the Radar toolchain.

## 1. Advanced Local RAG (BM25s)
**Goal:** Replace the naive C-based hashing embedder with a state-of-the-art sparse retrieval algorithm.
*   **Dependency:** `uv add bm25s`
*   **Implementation:** 
    *   Update `IntelligenceAgent` in `src/radar/core/ingest.py` to use `bm25s`.
    *   Create a local BM25 index on disk (`.radar_index/`).
    *   Whenever a signal is ingested, tokenize and add it to the BM25 index.
    *   Update `radar ask` to query the BM25 index for exact semantic/keyword matches, eliminating the "loose" matching of the hash embedder.

## 2. Tactical Audio Interception (Local Whisper)
**Goal:** Intercept and transcribe live police/fire scanner audio using local AI.
*   **Dependency:** `uv add faster-whisper pydub` and system `ffmpeg`.
*   **Implementation:**
    *   Create `AudioIngestAgent` in `src/radar/core/ingest.py`.
    *   Use `ffmpeg` (via subprocess) to record a 15-30 second chunk from an ICECAST/HTTP audio stream (e.g., Broadcastify).
    *   Run the audio through a quantized `faster-whisper` (tiny.en) model running purely on the CPU.
    *   Ingest the transcribed text as a `Signal`.

## 3. Statistical Anomaly Detection (Isolation Forest)
**Goal:** Mathematically detect anomalies in data flow rather than relying on regex keywords.
*   **Dependency:** `uv add scikit-learn numpy`
*   **Implementation:**
    *   Update `TacticalAgent.detect_anomalies`.
    *   Query the last 7 days of signals to build a feature matrix (e.g., Signal Frequency, Sentiment/Length of text, Listener counts from Broadcastify).
    *   Train an `IsolationForest` on the baseline.
    *   Evaluate the current SITREP/data point against the forest to calculate an anomaly score. Flag as CRITICAL if the score is -1.

## 4. The Mission Control TUI (Textual)
**Goal:** Build a persistent, interactive terminal dashboard.
*   **Dependency:** `uv add textual`
*   **Implementation:**
    *   Create `src/radar/ui/dashboard.py`.
    *   Build a Textual `App` with panes for:
        *   **Feed:** Scrolling list of latest Signals/News.
        *   **Tactical:** Live alerts and anomaly detection outputs.
        *   **Search:** An interactive input box to query the BM25 index and see summaries.
    *   Update `src/radar/main.py` to launch the TUI via `radar dash`.
