
import ollama
import math

# 1. Read the document
with open("notes.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Split the document into chunks
chunks = [
    sentence.strip()
    for sentence in text.replace("\n", " ").split(".")
    if sentence.strip()
]

if not chunks:
    print("No text found in notes.txt.")
    raise SystemExit

# 3. Create embeddings for each chunk
chunk_embeddings = []

for chunk in chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )
    chunk_embeddings.append(response["embeddings"][0])


# 4. Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# 5. Ask the user a question
question = input("Ask a question: ")

question_response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = question_response["embeddings"][0]


# 6. Find and rank the most relevant chunks
scores = []

for chunk, embedding in zip(chunks, chunk_embeddings):
    score = cosine_similarity(question_embedding, embedding)
    scores.append((score, chunk))

# Sort from highest similarity to lowest
scores.sort(reverse=True)

print("\nTop 3 matches:")

for score, chunk in scores[:3]:
    print(f"{score:.4f} -> {chunk}")


# 7. Check the best similarity score
threshold = 0.60
best_score, relevant_chunk = scores[0]

if best_score < threshold:
    print("\nAI Answer:")
    print("I don't know based on the provided notes.")

else:
    print("\nRelevant information:", relevant_chunk)

    # 8. Generate an answer using the retrieved context
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the "
                    "provided context. You may paraphrase the context, "
                    "but do not add outside facts. If the context "
                    "does not contain enough information to answer, "
                    "reply exactly: "
                    "'I don't know based on the provided notes.'"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context: {relevant_chunk}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    print("\nAI Answer:")
    print(response["message"]["content"])
