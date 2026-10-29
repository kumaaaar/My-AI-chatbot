
import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="Streamlit helps build web apps using Python."
)

embedding = response["embeddings"][0]

print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])
