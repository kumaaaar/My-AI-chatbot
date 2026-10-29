
import json

messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hey wassup!"}
]

# Step 1: Save the conversation
with open("chat_history.json", "w", encoding="utf-8") as file:
    json.dump(messages, file, indent=4)

print("Conversation saved successfully!")

# Step 2: Read the conversation back
with open("chat_history.json", "r", encoding="utf-8") as file:
    loaded_messages = json.load(file)

print(loaded_messages)
print(type(loaded_messages))
print(loaded_messages[0]["content"])