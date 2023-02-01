import tkinter as tk
from tkinter import messagebox
import openai

#You need to add your own Key
openai.api_key = "sk-YvxGal2i94TicBX62......"

def generate_response(event=None):
    prompt = entry.get()
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=1,
    ).choices[0].text

    chat_history.configure(state="normal")
    chat_history.image_create("end", image=user_icon)
    chat_history.insert("end", "You: " + prompt + "\n")
    chat_history.image_create("end", image=chatgpt_icon)
    chat_history.insert("end", "ChatGPT: " + response + "\n")
    chat_history.configure(state="disabled")
    entry.delete(0, "end")


root = tk.Tk()
root.title("ChatGPT")




# Add user icon
user_icon = tk.PhotoImage(file="./user-png.png").subsample(2,2)

# Add ChatGPT icon
chatgpt_icon = tk.PhotoImage(file="./chatgpt2-icon.png").subsample(2,2)

chat_history = tk.Text(root, state="disabled")
chat_history.tag_configure("font", font=("Sans", 14))
chat_history.pack(fill="both", expand=True)


# Frame to hold the prompt
frame = tk.Frame(root)
frame.pack(fill="x")

# Add icon
icon = tk.Label(frame, image=user_icon)
icon.pack(side="left", padx=5)

# Add prompt text
entry = tk.Entry(frame)
entry.pack(side="left", fill="x", expand=True)
entry.focus_set()
entry.bind("<Return>", generate_response)

button = tk.Button(root, text="Generate Response", command=generate_response)
button.pack(fill="x")

root.mainloop()
