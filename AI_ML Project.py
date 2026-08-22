import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# 1. READ & CLEAN PDF
reader = PdfReader(r"d:\C++\2608.20123v1.pdf")
text = ""

for page in reader.pages[:8]:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"


def clean_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


text = clean_text(text)

start = text.find("ABSTRACT")
if start != -1:
    text = text[start:]


# 2. CHUNKING FUNCTIONS
def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def create_chunks(sentences, max_words=200, overlap_sentences=1):
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if (
            current_word_count + sentence_word_count > max_words
            and current_chunk
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_word_count = sum(
                len(s.split()) for s in current_chunk
            )

        current_chunk.append(sentence)
        current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


sentences = split_sentences(text)
chunks = create_chunks(sentences, max_words=200, overlap_sentences=1)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print("\n" + "=" * 50)
    print(f"CHUNK {i + 1}")
    print("=" * 50)
    print(chunk)
    print(f"\nWord count: {len(chunk.split())}")


# 3. EMBEDDING GENERATION & SIMILARITY
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
chunk_embeddings = model.encode(chunks)

print("\nNumber of embeddings:", len(chunk_embeddings))
print("Embedding dimension:", len(chunk_embeddings[0]))


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search(query, top_k=5):
    query_embedding = model.encode(query)
    results = []

    for i, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        results.append(
            {"chunk_id": i + 1, "score": float(score), "text": chunks[i]}
        )

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# 4. SINGLE SEARCH RUN
query = "What is Nested Sequential Monte Carlo?"
results = search(query, top_k=3)

print("\n" + "=" * 60)
print("SEARCH QUERY:", query)
print("=" * 60)

for result in results:
    print(f"\nCHUNK {result['chunk_id']}")
    print(f"Cosine Similarity: {result['score']:.4f}")
    print("\nText:")
    print(result["text"])
    print("\n" + "-" * 60)


# 5. BENCHMARKING RELEVANT VS IRRELEVANT QUERIES
relevant_queries = [
    "What is Nested Sequential Monte Carlo?",
    "How do discrete diffusion language models generate text?",
    "What are the limitations of bootstrap sequential Monte Carlo?",
    "What is the purpose of inference-time steering?",
    "How does NSMC improve the proposal distribution?",
    "What is the difference between gradient-based and gradient-free methods?",
]

irrelevant_queries = [
    "What is the capital of India?",
    "Why do humans exist?",
    "What is Newton's 4th law?",
    "Which language is better - C++ or Python?",
    "How is the Maths Association performing relative to the other associations?",
]

print("\n" + "=" * 70)
print("RELEVANT QUERY RESULTS")
print("=" * 70)

relevant_scores = []
for query in relevant_queries:
    results = search(query, top_k=1)
    top_result = results[0]
    relevant_scores.append(top_result["score"])

    print(f"\nQuery: {query}")
    print(f"Top Chunk: {top_result['chunk_id']}")
    print(f"Top Score: {top_result['score']:.4f}")


print("\n" + "=" * 70)
print("IRRELEVANT QUERY RESULTS")
print("=" * 70)

irrelevant_scores = []
for query in irrelevant_queries:
    results = search(query, top_k=1)
    top_result = results[0]
    irrelevant_scores.append(top_result["score"])

    print(f"\nQuery: {query}")
    print(f"Top Chunk: {top_result['chunk_id']}")
    print(f"Top Score: {top_result['score']:.4f}")


print("\n" + "=" * 70)
print("SCORE SUMMARY")
print("=" * 70)

print("\nRelevant query scores:")
for score in relevant_scores:
    print(f"{score:.4f}")

print("\nIrrelevant query scores:")
for score in irrelevant_scores:
    print(f"{score:.4f}")

print("\nRelevant statistics:")
print(f"Minimum: {np.min(relevant_scores):.4f}")
print(f"Maximum: {np.max(relevant_scores):.4f}")
print(f"Mean:    {np.mean(relevant_scores):.4f}")

print("\nIrrelevant statistics:")
print(f"Minimum: {np.min(irrelevant_scores):.4f}")
print(f"Maximum: {np.max(irrelevant_scores):.4f}")
print(f"Mean:    {np.mean(irrelevant_scores):.4f}")

THRESHOLD = 0.23


def search_with_threshold(query, top_k=3):
    results = search(query, top_k=len(chunks))
    valid_results = [
        result
        for result in results
        if result["score"] >= THRESHOLD
    ]

    valid_results = valid_results[:top_k]

    print("\n" + "=" * 60)
    print("SEARCH QUERY:", query)
    print("=" * 60)

    if not valid_results:
        print("\nNo valid match found.")
        print(f"Threshold: {THRESHOLD}")
        print(
            f"Highest similarity: "
            f"{results[0]['score']:.4f}"
        )
        return []
    for result in valid_results:
        print(f"\nCHUNK {result['chunk_id']}")
        print(
            f"Cosine Similarity: "
            f"{result['score']:.4f}"
        )
        print("\nStatus: VALID MATCH")
        print("\nText:")
        print(result["text"])
        print("\n" + "-" * 60)
    return valid_results
while True:
    query = input(
        "\nEnter a search query "
        "(or type 'exit' to quit): "
    )
    if query.lower() == "exit":
        print("\nProgram ended.")
        break
    search_with_threshold(query)
