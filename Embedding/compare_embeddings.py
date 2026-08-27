"""
compare_embeddings.py

GOAL OF THIS FILE
------------------
Show what a "text embedding" actually is, and how to compare two pieces
of text to see how similar their MEANING is (not just their spelling).

WHAT IS AN EMBEDDING?
----------------------
A neural network model can read a sentence and turn it into a list of
numbers, e.g. [0.12, -0.44, 0.98, ... ] (usually 300-1000+ numbers long).
This list of numbers is called a "vector", and the process of creating
it is called "embedding".

The key property: sentences with SIMILAR MEANING end up with vectors
that point in SIMILAR DIRECTIONS in space, even if they don't share any
of the same words. That's the whole point of embeddings -- they capture
meaning, not just exact words.

HOW DO WE COMPARE TWO VECTORS?
--------------------------------
We use "cosine similarity". Imagine each vector as an arrow starting at
the origin. Cosine similarity measures the ANGLE between two arrows:

    1.0   -> arrows point in exactly the same direction (identical meaning)
    0.0   -> arrows are perpendicular (unrelated meaning)
   -1.0   -> arrows point in opposite directions (opposite meaning)

In practice, for sentence embeddings, most scores you'll see fall
somewhere between 0 and 1.
"""

# sentence-transformers is a library that wraps pretrained embedding models
# and gives us a simple .encode() method to turn text into vectors.
from sentence_transformers import SentenceTransformer

# util.cos_sim is a ready-made cosine similarity function so we don't have
# to implement the math by hand.
from sentence_transformers import util


def load_model() -> SentenceTransformer:
    """
    Downloads (first run only, then cached locally) and loads a small,
    fast embedding model called "all-MiniLM-L6-v2".

    It's a good default for learning because it:
      - is small (~80MB) and runs fast on a CPU (no GPU needed)
      - produces good-quality general-purpose sentence embeddings
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def compare_sentences(model: SentenceTransformer, sentence_a: str, sentence_b: str) -> float:
    """
    Turns two sentences into embedding vectors and returns how similar
    they are, as a single number between -1 and 1 (higher = more similar).
    """
    # model.encode() takes text in and returns a vector (a list of numbers)
    # representing that text's meaning.
    embedding_a = model.encode(sentence_a)
    embedding_b = model.encode(sentence_b)

    # Compare the two vectors using cosine similarity.
    # The result is wrapped in a small tensor/array, so .item() pulls out
    # the plain Python float we actually want to print.
    similarity_score = util.cos_sim(embedding_a, embedding_b).item()

    return similarity_score


def main():
    model = load_model()

    # A handful of example sentence pairs to show different similarity levels:
    # 1. Same meaning, totally different words -> should score HIGH
    # 2. Same topic, different meaning          -> should score MEDIUM
    # 3. Unrelated topics                        -> should score LOW
    sentence_pairs = [
        ("The cat sat on the mat.", "A feline was resting on the rug."),
        ("The cat sat on the mat.", "The dog ran across the yard."),
        ("The cat sat on the mat.", "The stock market crashed today."),
        ("The cat sat on the mat.", "I fell from my bike."),
        ("The kids play in the park.", "The play was for kids in the park.")
    ]

    print(f"Model loaded: all-MiniLM-L6-v2\n")
    print("Comparing sentence pairs by MEANING (cosine similarity):\n")

    for sentence_a, sentence_b in sentence_pairs:
        score = compare_sentences(model, sentence_a, sentence_b)
        print(f'"{sentence_a}"')
        print(f'"{sentence_b}"')
        print(f"-> similarity score: {score:.4f}\n")


# This check means "only run main() if this file is executed directly",
# e.g. `python compare_embeddings.py`. If this file were imported into
# another script instead, main() would NOT run automatically.
if __name__ == "__main__":
    main()
