# evaluation.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load resources
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("builder_index.faiss")
with open("chunk_lookup.json", "r") as f:
    chunk_lookup = json.load(f)

# LLM query function
def query_local_llm(context, question):
    prompt = f"""Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}],
        options={"temperature": 0}
    )
    return response['message']['content']

# Sample evaluation questions and expected keywords
evaluation_data = [
    {
        "question": "Which builder had the most problems?",
        "expected_keywords": ["ABC Homes", "most problems"]
    },
    {
        "question": "What issues did H123 have?",
        "expected_keywords": ["Cracked", "Wall"]
    },
    {
         "question": "Which houses passed inspection?",
         "expected_keywords": ["H125", "H126"]
    }

]

# Run evaluation
total = len(evaluation_data)
score = 0

for entry in evaluation_data:
    query = entry["question"]
    expected = entry["expected_keywords"]

    # Get chunks and context
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    valid_chunks = [chunk_lookup[str(i)] for i in I[0] if str(i) in chunk_lookup]
    context = "\n".join(valid_chunks)

    # Query LLM
    answer = query_local_llm(context, query)
    print(f"\nQ: {query}\nA: {answer.strip()}")

    # Keyword check
    match = any(kw.lower() in answer.lower() for kw in expected)
    print("✅ Passed" if match else "❌ Failed")
    if match:
        score += 1

print(f"\n🧪 Evaluation Complete: {score}/{total} correct responses")
