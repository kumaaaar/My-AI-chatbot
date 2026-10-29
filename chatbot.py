
import json
import ollama

# Load saved conversation history
try:
    with open("chat_history.json", "r", encoding="utf-8") as file:
        messages = json.load(file)

    print("Previous conversation loaded!")

except FileNotFoundError:
    messages = []
    print("No previous conversation found. Starting fresh.")

except json.JSONDecodeError:
    messages = []
    print("Saved history is empty or corrupted. Starting fresh.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    if user_input.lower() == "clear":
        messages = []

        with open("chat_history.json", "w", encoding="utf-8") as file:
            json.dump(messages, file, indent=4)

        print("Conversation cleared!")
        continue

    if user_input.strip() == "":
        continue

    # Add the user's message to conversation history
    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=messages
        )

    except Exception as e:
        print("Sorry, something went wrong:", e)

        # Remove the failed user message
        messages.pop()

        continue

    # Get the assistant's reply
    assistant_reply = response["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

    # Save conversation after a successful response
    with open("chat_history.json", "w", encoding="utf-8") as file:
        json.dump(messages, file, indent=4)

    print("Bot:", assistant_reply)
