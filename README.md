<<<<<<< HEAD
# Local Chat (ChatGPT-style UI + Free Local AI)

This project runs a **ChatGPT-like desktop interface** (sidebar chat history + main conversation) on top of a **free local LLM backend (Ollama)**.

## What’s included

### UI
- **ChatGPT-style layout**
  - left sidebar with **New chat** + chat history
  - main chat area with **full-width message bands** (assistant/user styling like ChatGPT)
  - multiline composer at the bottom
- **Streaming generation**
  - assistant row appears immediately with a **typing indicator (· · ·)**
  - updates live as tokens stream
- **Input fixes**
  - typed characters always visible (correct text color + input is never disabled)

### AI agent (free)
- Uses **Ollama** locally — **no OpenAI credits / API key** required.

### Voice
- Offline **TTS** via `pyttsx3`
- Optional **Speak while generating** (sentence chunking) to keep speech consistent with what’s shown on screen.
- Mic input uses `SpeechRecognition` default Google recognizer (free, online). If you want fully offline STT, swap to Vosk.

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> If `pyaudio` fails: see platform notes in `requirements.txt`.

### 2) Install Ollama and pull a model

```bash
ollama pull llama3.1:8b
```

Ollama typically runs at: `http://localhost:11434`

### 3) Run

```bash
python app.py
```

Legacy ttk UI is kept as `app_ttk.py`.

>>>>>>>
