import tkinter as tk
from tkinter import messagebox
from parse import Parse
import speech_recognition as sr

# ---- Back4App Setup (Replace with your credentials) ----
APPLICATION_ID = "K1fT0DIi85jzZt89YiMZOxjSuDgakdjbEOfiqOZe"      # Replace with your Back4App Application ID
CLIENT_KEY = "FJBOvvBmTQ93tHsE4Fd5B4XKuzviQi0GIyq0cnQB"      # Replace with your Back4App Client Key

Parse.initialize(APPLICATION_ID, CLIENT_KEY)

# Define a Parse Object subclass for the shopping items
class ShoppingItem(Parse.Object):
    pass

# -------- App Logic --------
def add_item():
    item_name = entry.get().strip()
    if item_name:
        item = ShoppingItem()
        item.set("name", item_name)
        item.set("status", "to_bring")
        item.save()
        entry.delete(0, tk.END)
        update_lists()
    else:
        messagebox.showwarning("Input Error", "Please enter an item name.")

def update_lists():
    listbox_to_bring.delete(0, tk.END)
    listbox_done.delete(0, tk.END)
    global items_cache
    items_cache = {}

    # Query items with "to_bring" status
    to_bring_items = ShoppingItem.Query.filter(status="to_bring").find()
    for item in to_bring_items:
        listbox_to_bring.insert(tk.END, item.get("name"))
        items_cache[item.get("name")] = item

    # Query items with "done" status
    done_items = ShoppingItem.Query.filter(status="done").find()
    for item in done_items:
        listbox_done.insert(tk.END, item.get("name"))

def mark_done(item_name):
    item = items_cache.get(item_name)
    if item:
        item.set("status", "done")
        item.save()
        update_lists()

def mark_selected():
    try:
        idx = listbox_to_bring.curselection()[0]
        item_name = listbox_to_bring.get(idx)
        mark_done(item_name)
    except IndexError:
        messagebox.showwarning("Error", "Please select an item to mark as done.")

def listen_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        messagebox.showinfo("Voice Command", "Say 'I have brought [item]'")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        process_voice_command(text)
    except Exception as e:
        messagebox.showerror("Voice Error", f"Could not understand audio.\n{e}")

def process_voice_command(command):
    words = command.lower().split()
    if "brought" in words:
        idx = words.index("brought")
        item_name = " ".join(words[idx + 1:])
        for candidate in items_cache.keys():
            if item_name.strip() in candidate.lower():
                mark_done(candidate)
                messagebox.showinfo("Voice Command", f"Marked '{candidate}' as done!")
                return
        messagebox.showinfo("Voice Command", f"No matching item found for '{item_name}'.")
    else:
        messagebox.showinfo("Voice Command", f"Say 'I have brought [item]' to mark an item done.")

# ---- Tkinter UI ----
app = tk.Tk()
app.title("Back4App Shopping List")
app.geometry("600x400")

entry = tk.Entry(app, width=30)
entry.grid(row=0, column=0, padx=10, pady=10)
btn_add = tk.Button(app, text="Add Item", command=add_item)
btn_add.grid(row=0, column=1, padx=10, pady=10)

tk.Label(app, text="To Bring:").grid(row=1, column=0, padx=10, pady=5)
listbox_to_bring = tk.Listbox(app, height=15, width=30, selectmode=tk.SINGLE)
listbox_to_bring.grid(row=2, column=0, padx=10, pady=5)
btn_mark_done = tk.Button(app, text="Mark as Done", command=mark_selected)
btn_mark_done.grid(row=3, column=0, padx=10, pady=5)

tk.Label(app, text="Done:").grid(row=1, column=1, padx=10, pady=5)
listbox_done = tk.Listbox(app, height=15, width=30)
listbox_done.grid(row=2, column=1, padx=10, pady=5)

btn_voice = tk.Button(app, text="Speak 'I have brought ...'", command=listen_voice)
btn_voice.grid(row=4, column=0, columnspan=2, pady=15)

items_cache = {}

update_lists()
app.mainloop()
