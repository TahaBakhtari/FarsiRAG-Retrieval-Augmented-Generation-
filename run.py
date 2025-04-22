from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import re
from hazm import Normalizer
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="api-key"
)

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    normalizer = Normalizer()
    text = "\n".join(normalizer.normalize(page.extract_text()) for page in reader.pages if page.extract_text())
    return text

def split_text(text, size=300):
    sentences = re.split(r'(?<=[.!؟])\s+', text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < size:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def retrieve_top_k(query, chunks, index, embedder, k=3):
    query_vector = embedder.encode([query])
    faiss.normalize_L2(query_vector)
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

def answer_with_openrouter(context, question):
    prompt = f"""Persian Textbook:

{context}

Question: {question}

Please provide a scientific, accurate, and clear answer:"""

    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "rag-persian-pdf-demo"
        },
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Persian-speaking AI assistant that provides accurate and clear answers to educational questions."},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    pdf_path = "math.pdf"  # Path to the Persian PDF file
    question = input("Enter Your Question :")

    print(" Reading PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    print(" Splitting text...")
    chunks = split_text(text)

    print(" Creating embedding...")
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embed_chunks(chunks, embedder)

    print(" Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(" Retrieving related chunks...")
    top_chunks = retrieve_top_k(question, chunks, index, embedder)
    context = "\n".join(top_chunks)

    print(" Sending question to GPT-4o-mini via OpenRouter...")
    answer = answer_with_openrouter(context, question)

    print("\n Final answer:")
    print(answer)
