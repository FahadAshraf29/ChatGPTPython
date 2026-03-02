"""Legacy entry point.

Run:
    python chatgpt.py

This launches the new ChatGPT-style desktop UI (customtkinter) backed by a free local
Ollama model (no OpenAI credits / API key).

Or run directly:
    python app.py

If you prefer the older ttk variant, run:
    python app_ttk.py
"""

from app import main


if __name__ == "__main__":
    main()
