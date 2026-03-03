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

## Changing Ollama model storage drive (Windows)

Set:

- `OLLAMA_MODELS=D:\Ollama`

Then pull models again.
=======
# ChatGPTPython
Overview

This program creates a graphical user interface for the OpenAI ChatGPT API. The GUI allows the user to input a prompt and receive a response from the API.
Dependencies

    tkinter: Standard GUI library for Python
    openai: API client library for OpenAI API
    messagebox: Provides a set of dialogues for tkinter GUI

Files

    dist folder: Contains the built and compiled version of the program.
    build folder: Contains the build files and resources used to build the program.

Program Structure

    GUI Initialization: The tkinter GUI is created and given a title "ChatGPT".

    Icons: Two icons are added, one for the user and one for ChatGPT.

    Chat History: A tkinter Text widget is created to display the chat history between the user and ChatGPT.

    Prompt Frame: A tkinter Frame widget is created to hold the user's prompt. The frame contains an icon and a tkinter Entry widget for the user to input their prompt.

    Generate Response: The generate_response function is called when the user inputs their prompt and presses the "Generate Response" button or the "Return" key. The function retrieves the user's prompt, sends it to the OpenAI API, and displays the response in the chat history.

How to Use

    Install the dependencies: tkinter, openai, and messagebox.
    Clone or download the repository.
    Open a terminal in the directory containing the program files.
    On CLI write python3 chatgpt.py or use any IDE

Notes

    Make sure to input your own OpenAI API key in the program before running it.
    The program requires internet connectivity to access the OpenAI API.
    The program assumes that the user-png.png and chatgpt2-icon.png image files are in the same directory as the program file.
>>>>>>> 4fc605c5c6a59547bbedc28f2941c4b27efec91f
