"""
outliers.py

GOAL OF THIS FILE
------------------
Test whether we can automatically catch a question that doesn't
belong to any of our 4 known languages (python/r/html/css) --
something classify.py structurally CANNOT do (it always confidently
picks one of the 4, no matter how unrelated the question actually is).

THIS FILE COMPARES TWO DIFFERENT APPROACHES ON PURPOSE
-----------------------------------------------------------
We tried Isolation Forest first and it underperformed badly -- it
barely flagged pure GIBBERISH text as unusual (ranked it 38th out of
401, not even close to the top). Verifying that with a simpler
approach -- distance to the nearest real questions -- showed the
embeddings themselves had a clean, strong signal all along; Isolation
Forest just wasn't the right tool for this shape of data (dense,
384-dimensional, continuous embeddings). Isolation Forest isolates
points by looking at ONE random dimension at a time, which works well
when outliers are extreme in a single feature (like age=999 in a
spreadsheet), but poorly when "unusual" means moderately different
across MANY dimensions at once, cumulatively -- which is exactly what
an off-topic embedding looks like. Distance naturally combines all 384
dimensions into one number, so it catches that cumulative signal where
Isolation Forest can't.

Both are kept here, side by side, specifically so you can see that
contrast for yourself rather than just take our word for it -- and as
a reminder to verify a technique is actually working before trusting
its conclusions about the data.

THE THRESHOLD (DISTANCE_THRESHOLD BELOW)
--------------------------------------------
1.1 was picked by eyeballing our own real test results: on-topic
questions consistently land under ~1.0, off-topic/gibberish ones
consistently land over ~1.2. This is a rough, manually-tuned cutoff
from a handful of examples, not a statistically rigorous one -- treat
the raw distance number as more informative than the yes/no verdict.
"""

import json
import sys

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

DISTANCE_THRESHOLD = 1.1

# A handful of built-in test cases: some genuinely off-topic, one gibberish,
# one real on-topic question for contrast.
BUILT_IN_TEST_CASES = [
    ("gibberish", "zzz qqq xxx flerbnorp glorptastic 12345"),
    ("cooking", "How do I make garlic butter shrimp pasta?"),
    ("hiking", "What is the best hiking trail in Colorado?"),
    ("flat tire", "How do I fix my flat tire?"),
    ("real python question", "How do I merge two dictionaries in one line?"),
]


def load_questions() -> list[dict]:
    with open("questions.json") as f:
        return json.load(f)


def check_question(question: str, model, iso_forest, nearest_neighbors) -> dict:
    """
    Runs one question through BOTH outlier-detection approaches and
    returns a dict with both results, so they can be compared directly.
    """
    embedding = model.encode([question])

    # Isolation Forest: score_samples gives a continuous anomaly score
    # (lower = more anomalous). predict gives a -1/1 verdict using its
    # own internal threshold.
    iso_score = iso_forest.score_samples(embedding)[0]
    iso_verdict = "outlier" if iso_forest.predict(embedding)[0] == -1 else "normal"

    # Distance-based: average distance to the 5 closest REAL questions.
    distances, _ = nearest_neighbors.kneighbors(embedding)
    avg_distance = distances[0].mean()
    distance_verdict = "outlier" if avg_distance > DISTANCE_THRESHOLD else "normal"

    return {
        "question": question,
        "iso_score": iso_score,
        "iso_verdict": iso_verdict,
        "avg_distance": avg_distance,
        "distance_verdict": distance_verdict,
    }


def print_result(result: dict):
    print(f'"{result["question"]}"')
    print(f'  Isolation Forest : score={result["iso_score"]:7.4f}  -> {result["iso_verdict"]}')
    print(f'  NN distance      : dist={result["avg_distance"]:7.4f}  -> {result["distance_verdict"]}')
    print()


def main():
    questions = load_questions()
    real_titles = [q["title"] for q in questions]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(real_titles)} real questions...")
    real_embeddings = model.encode(real_titles)

    # Both detectors are fit ONLY on the real 400 questions -- they
    # learn what "normal" looks like from those, then we check new
    # questions against that baseline.
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    iso_forest.fit(real_embeddings)

    nearest_neighbors = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nearest_neighbors.fit(real_embeddings)

    print("\nBuilt-in test cases:\n")
    for label, question in BUILT_IN_TEST_CASES:
        result = check_question(question, model, iso_forest, nearest_neighbors)
        print(f"[{label}]")
        print_result(result)

    # Also accept a question of your own from the command line, same
    # pattern as classify.py and full_pipeline.py.
    if len(sys.argv) > 1:
        custom_question = sys.argv[1]
        print("Your question:\n")
        result = check_question(custom_question, model, iso_forest, nearest_neighbors)
        print_result(result)
    else:
        print("(Pass your own question as a command-line argument to test it, e.g.:")
        print('  python outliers.py "some question here")')


if __name__ == "__main__":
    main()
