from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are PaperLens, a research assistant that answers questions STRICTLY based on the provided context from uploaded research papers.

Rules you must follow:
1. Answer ONLY using the context provided. Do NOT use any outside knowledge.
2. If the answer is not found in the context, respond exactly with: "I couldn't find this in the uploaded papers."
3. Be concise and precise — this is academic Q&A, not a general chatbot.
4. At the end of every answer, cite which paper(s) you drew from using: [Source: filename.pdf]
5. Never make up citations. Only cite papers that are actually in the context.
"""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[Chunk {i} | {chunk['filename']} | score: {chunk['score']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(lines)


def generate_answer(
    query: str,
    chunks: list[dict],
    chat_history: list[dict]
) -> str:
    """
    Generate an answer from Groq using:
    - retrieved chunks as context
    - last 5 messages as conversation history
    - strict system prompt to prevent hallucination

    Args:
        query: current user question
        chunks: top relevant chunks from Qdrant (already filtered by score)
        chat_history: last N messages [{"role": ..., "content": ...}]

    Returns:
        LLM answer string
    """
    context = build_context(chunks)

    # Inject context into the current user message
    user_message_with_context = f"""Context from uploaded papers:

{context}

---

Question: {query}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *chat_history,  # last 5 turns for conversational memory
        {"role": "user", "content": user_message_with_context}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
        temperature=0.2  # low temp = more faithful, less creative hallucination
    )

    return response.choices[0].message.content.strip()
