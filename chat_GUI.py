import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import chatbot  # Assuming chatbot.py contains all necessary functions

class ChatGUI:
    last_bot_response = ""  # Class variable to keep track of the last bot response

    def __init__(self, master):
        self.master = master
        master.title("Chatbot")

        self.chat_log = ScrolledText(master, state='disabled', width=80, height=20)
        self.chat_log.pack(padx=10, pady=10)

        self.msg_entry = tk.Entry(master, width=50)
        self.msg_entry.pack(padx=10, pady=5)
        self.msg_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(pady=5)

        self.user_id = simpledialog.askstring("User ID", "Enter your user ID or 'new' for a new user:", parent=master)
        if not self.user_id:
            messagebox.showinfo("No User ID", "No user ID provided. Exiting application.")
            master.destroy()
        else:
            self.init_chat()

    # def init_chat(self):
    #     # Initialize or load the user model here
    #     self.user_model = chatbot.load_user_model(self.user_id)  # Adjust according to your chatbot.py's function
    #     if self.user_model is None:  # Assuming load_user_model returns None if the user model doesn't exist
    #         self.user_model = {'name': '', 'personal_info': {}, 'likes': [], 'dislikes': []}
    #         chatbot.save_user_model(self.user_id, self.user_model)  # And a function to save the model
    #     self.user_name = self.user_model.get('name', 'there')  # Default to 'there' if name not in model
    def init_chat(self):
        self.user_model = chatbot.load_user_model(self.user_id)  # Load or initialize the user model
        if self.user_model is None:  # If the user model doesn't exist, create a new one
            self.user_model = {'name': '', 'personal_info': {}, 'likes': [], 'dislikes': []}
            self.user_name = simpledialog.askstring("Name", "What's your name?", parent=self.master)  # Ask for the user's name
            self.user_model['name'] = self.user_name
            chatbot.save_user_model(self.user_id, self.user_model)  # Save the new user model
        else:
            self.user_name = self.user_model.get('name', 'there')  # Use the name from the user model

        self.update_chat_log(f"Chatbot: Welcome back, {self.user_name}! How can I help you today?\n")  # Display the welcome message


    def send_message(self, event=None):
        user_input = self.msg_entry.get()
        if user_input:
            self.msg_entry.delete(0, tk.END)
            self.update_chat_log(f"You: {user_input}\n")
            threading.Thread(target=self.process_user_input, args=(user_input,)).start()

    def process_user_input(self, user_input):
        # Process user input for greetings, likes, dislikes, and general queries
        if user_input.lower() == 'quit':
            self.chat_log.after(100, self.update_chat_log, f"Chatbot: Bye {self.user_name}! Have a great day!\n")
            return

        response = ""
        greeting = chatbot.get_greeting(user_input)
        if greeting:
            response += f"{greeting} {self.user_name}!\n"

        # Handling likes and dislikes
        if "dislike" in user_input.lower() or "disliked" in user_input.lower() or "don't like" in user_input.lower():
            disliked_items = chatbot.extract_nouns_and_adjectives(ChatGUI.last_bot_response)
            for item in disliked_items:
                chatbot.update_user_model(self.user_id, 'dislikes', item)
            response += f"Noted. You don't like {', '.join(disliked_items)}.\n"
        elif "like" in user_input.lower() or "liked" in user_input.lower():
            liked_items = chatbot.extract_nouns_and_adjectives(ChatGUI.last_bot_response)
            for item in liked_items:
                chatbot.update_user_model(self.user_id, 'likes', item)
            response += f"Noted. You like {', '.join(liked_items)}.\n"

        # If no specific handling was done, get a general response
        if not response.strip():
            response = chatbot.get_response(self.user_id, user_input)
        chatbot.update_last_bot_response(response)  # Update the last bot response
        self.chat_log.after(100, self.update_chat_log, f"Chatbot: {response}\n")

    def update_chat_log(self, message):
        self.chat_log.config(state='normal')
        self.chat_log.insert(tk.END, message)
        self.chat_log.config(state='disabled')
        self.chat_log.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatGUI(root)
    root.mainloop()
