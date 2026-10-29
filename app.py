
import math
import re
from io import BytesIO

import ollama
import streamlit as st
from pypdf import PdfReader


# -----------------------------------------
# 1. Page setup
# -----------------------------------------
st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 My AI Chatbot")
st.caption("Chat normally or attach documents to ask questions about them.")


# -----------------------------------------
# 2. Models
# -----------------------------------------
CHAT_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 3
THRESHOLD = 0.40


# -----------------------------------------
# 3. Initialize session state
# -----------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# -----------------------------------------
# 4. Extract text from documents
# -----------------------------------------
def extract_pages(filename, file_bytes):
    filename = filename.lower()

    if filename.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="replace")
        return [{"page": None, "text": text}]

    if filename.endswith(".pdf"):
        pdf = PdfReader(BytesIO(file_bytes))
        pages = []

        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text
                })

        return pages

    return []


# -----------------------------------------
# 5. Split text into chunks
# -----------------------------------------
def split_into_chunks(pages):
    chunks = []

    for page_data in pages:
        text = re.sub(
            r"\s+",
            " ",
            page_data["text"]
        ).strip()

        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            # Prefer a word boundary when possible.
            if end < len(text):
                boundary = text.rfind(" ", start, end)

                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_data["page"],
                    "filename": page_data.get("filename", "")
                })

            if end >= len(text):
                break

            start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


# -----------------------------------------
# 6. Create embeddings
# -----------------------------------------
@st.cache_data(show_spinner=False)
def embed_texts(texts):
    embeddings = []

    for text in texts:
        result = ollama.embed(
            model=EMBED_MODEL,
            input=text
        )

        embeddings.append(result["embeddings"][0])

    return embeddings


# -----------------------------------------
# 7. Cosine similarity
# -----------------------------------------
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# -----------------------------------------
# 8. Retrieve relevant document chunks
# -----------------------------------------
def retrieve_context(question, chunks, embeddings):
    result = ollama.embed(
        model=EMBED_MODEL,
        input=question
    )

    question_embedding = result["embeddings"][0]
    scores = []

    for chunk, embedding in zip(chunks, embeddings):
        score = cosine_similarity(
            question_embedding,
            embedding
        )

        scores.append((score, chunk))

    scores.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scores[:TOP_K]


# -----------------------------------------
# 9. Process uploaded documents
# -----------------------------------------
uploaded_prompt = None

# Display currently attached documents.
if st.session_state.documents:
    with st.expander(
        f"📎 Attached documents ({len(st.session_state.documents)})",
        expanded=True
    ):
        for document in st.session_state.documents:
            st.write(f"📄 {document['name']}")

        if st.button("Remove all documents"):
            st.session_state.documents = []
            st.rerun()


# -----------------------------------------
# 10. Build the RAG knowledge base
# -----------------------------------------
all_chunks = []
embeddings = []

if st.session_state.documents:
    try:
        for document in st.session_state.documents:
            pages = extract_pages(
                document["name"],
                document["bytes"]
            )

            for page_data in pages:
                page_data["filename"] = document["name"]

            all_chunks.extend(
                split_into_chunks(pages)
            )

        if all_chunks:
            with st.spinner("Preparing your documents..."):
                embeddings = embed_texts([
                    chunk["text"] for chunk in all_chunks
                ])

            st.caption(
                f"RAG ready · {len(all_chunks)} chunks indexed"
            )

        else:
            st.warning(
                "No readable text found in the attached documents. "
                "Scanned PDFs may require OCR."
            )

    except Exception as e:
        st.error(f"Could not prepare documents: {e}")
        st.stop()


# -----------------------------------------
# 11. Display chat history
# -----------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.write(source)


# -----------------------------------------
# 12. Chat input with built-in attachment button
# -----------------------------------------
prompt = st.chat_input(
    "Message your AI...",
    accept_file="multiple",
    file_type=["pdf", "txt"]
)

if prompt:
    # Save newly attached files in session state.
    if prompt.files:
        for uploaded_file in prompt.files:
            new_document = {
                "name": uploaded_file.name,
                "bytes": uploaded_file.getvalue()
            }

            # Replace an existing document with the same filename.
            st.session_state.documents = [
                doc for doc in st.session_state.documents
                if doc["name"] != new_document["name"]
            ]

            st.session_state.documents.append(new_document)

        # Keep the text question, if one was submitted with the files.
        user_input = prompt.text.strip()

        if not user_input:
            st.rerun()

    else:
        user_input = prompt.text.strip()

    # Ignore blank messages.
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        try:
            with st.chat_message("assistant"):
                sources = []

                # ---------------------------------
                # MODE A: RAG when documents exist
                # ---------------------------------
                if st.session_state.documents and all_chunks:
                    with st.spinner("Searching your documents..."):
                        matches = retrieve_context(
                            user_input,
                            all_chunks,
                            embeddings
                        )

                        sources = [
                            chunk
                            for score, chunk in matches
                            if score >= THRESHOLD
                        ]

                        if not sources:
                            assistant_reply = (
                                "I couldn't find sufficiently relevant "
                                "information in your attached documents. "
                                "Try rephrasing your question."
                            )

                        else:
                            context_parts = []

                            for index, chunk in enumerate(
                                sources,
                                start=1
                            ):
                                page = chunk["page"]

                                if page is None:
                                    location = chunk["filename"]
                                else:
                                    location = (
                                        f"{chunk['filename']}, "
                                        f"page {page}"
                                    )

                                context_parts.append(
                                    f"Source {index} ({location}):\n"
                                    f"{chunk['text']}"
                                )

                            context = "\n\n".join(context_parts)

                            response = ollama.chat(
                                model=CHAT_MODEL,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a document question-answering "
                                            "assistant. Answer using the supplied "
                                            "context. Read it carefully. If the "
                                            "answer is present, provide it directly. "
                                            "Do not claim information is missing "
                                            "when it is present. Do not invent "
                                            "facts. If the answer genuinely cannot "
                                            "be found in the context, say you don't "
                                            "know based on these documents. "
                                            "Keep your answer concise."
                                        )
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            f"DOCUMENT CONTEXT:\n{context}\n\n"
                                            f"QUESTION:\n{user_input}\n\n"
                                            "Answer using the document context."
                                        )
                                    }
                                ]
                            )

                            assistant_reply = (
                                response["message"]["content"]
                            )

                # ---------------------------------
                # MODE B: Normal AI chat
                # ---------------------------------
                else:
                    with st.spinner("Thinking..."):
                        chat_messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a helpful AI assistant. "
                                    "Answer clearly and naturally."
                                )
                            }
                        ]

                        # Include recent conversation history.
                        for message in st.session_state.messages[-11:]:
                            chat_messages.append({
                                "role": message["role"],
                                "content": message["content"]
                            })

                        response = ollama.chat(
                            model=CHAT_MODEL,
                            messages=chat_messages
                        )

                        assistant_reply = (
                            response["message"]["content"]
                        )

                st.write(assistant_reply)

                # Display retrieved document sources.
                if sources:
                    with st.expander("📚 View retrieved sources"):
                        for source in sources:
                            page = source["page"]

                            if page is None:
                                location = source["filename"]
                            else:
                                location = (
                                    f"{source['filename']} "
                                    f"— Page {page}"
                                )

                            st.markdown(f"**{location}**")
                            st.write(source["text"])
                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_reply,
                "sources": [
                    (
                        f"{source['filename']} "
                        + (
                            f"(Page {source['page']})"
                            if source["page"] is not None
                            else ""
                        )
                        + f": {source['text']}"
                    )
                    for source in sources
                ]
            })

        except Exception as e:
            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] == "user"
            ):
                st.session_state.messages.pop()

            st.error(f"Something went wrong: {e}")
