import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load API Key


load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)


# Load FAISS Database


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db",
    embedding_model,
    allow_dangerous_deserialization=True
)


# Generate Response


def generate_response(emotion):

    query = f"I am feeling {emotion}"

    docs = db.similarity_search(query, k=3)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are an AI Mental Wellness Assistant.

Detected Emotion:
{emotion}

Relevant Knowledge:
{context}

Instructions:
- Respond empathetically.
- Use the retrieved knowledge.
- Suggest one practical activity.
- Suggest one breathing exercise if appropriate.
- End with one motivational sentence.
- Keep the response under 500 words.
"""

    response = client.chat.completions(
        model="sarvam-105b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content