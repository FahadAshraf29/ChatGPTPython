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
    Go to dist\chatgpt.exe for running the executable file.

Notes

    Make sure to input your own OpenAI API key in the program before running it.
    The program requires internet connectivity to access the OpenAI API.
    The program assumes that the user-png.png and chatgpt2-icon.png image files are in the same directory as the program file.
