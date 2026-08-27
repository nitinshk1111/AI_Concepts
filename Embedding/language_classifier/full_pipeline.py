"""
full_pipeline.py

GOAL OF THIS FILE
------------------
cluster.py and classify.py each independently compared themselves
against the REAL StackOverflow tags. They never actually fed into each
other. This file builds the real, chained, production-style pipeline:

    1. CLUSTER raw embeddings -- no labels involved at all
    2. NAME each cluster ONCE -- simulates a human glancing at a few
       example questions per cluster and deciding what to call it
    3. Train a CLASSIFIER on those cluster-derived names (NOT the real
       tags)
    4. Check the whole chain against the real tags at the very end,
       purely to grade ourselves -- same as before, this comparison
       plays no role in how the pipeline actually works

IMPORTANT HONESTY NOTE ABOUT STEP 2
---------------------------------------
In a genuine real-world project, a human would open cluster 0, read a
handful of its questions, and decide "this looks like Python" -- a
manual, one-time judgment call. We don't have a human in the loop
here, so we SIMULATE that judgment by taking the most common real tag
within each cluster (majority vote). This is a stand-in for a human
decision, not a shortcut that lets the classifier see real labels --
the classifier below only ever sees the SIMULATED name, which can be
wrong if clustering made a mistake for a particular question.

Any error introduced by imperfect clustering, or by a "human" naming a
cluster slightly wrong, will now flow downstream into the classifier
-- which is exactly what would happen in reality too. That's the point
of running this end to end instead of grading each stage in isolation.
"""

import json
import sys
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_questions() -> list[dict]:
    with open("questions.json") as f:
        return json.load(f)


def name_clusters(discovered_clusters, real_tags) -> dict[int, str]:
    """
    Simulates a human naming each cluster once, by taking the most
    common real tag among the questions that ended up in that cluster.

    Returns e.g. {0: "python", 1: "css", 2: "r", 3: "html"} -- a
    one-time mapping from cluster NUMBER to a human-meaningful NAME.
    """
    tags_per_cluster: dict[int, list[str]] = {}
    for cluster_id, real_tag in zip(discovered_clusters, real_tags):
        tags_per_cluster.setdefault(cluster_id, []).append(real_tag)

    cluster_to_name = {}
    for cluster_id, tags_in_cluster in tags_per_cluster.items():
        most_common_tag, count = Counter(tags_in_cluster).most_common(1)[0]
        cluster_to_name[cluster_id] = most_common_tag
        print(
            f"  Cluster {cluster_id} -> named '{most_common_tag}' "
            f"({count}/{len(tags_in_cluster)} of its questions were actually '{most_common_tag}')"
        )
    return cluster_to_name


def main():
    questions = load_questions()
    titles = [q["title"] for q in questions]
    real_tags = [q["tag"] for q in questions]  # used ONLY for naming + final grading, never for training

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(titles)} questions...")
    embeddings = model.encode(titles)

    # ---- STEP 1: CLUSTER -- identical to cluster.py, no labels involved ----
    print("\nStep 1: clustering (no labels used)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    discovered_clusters = kmeans.fit_predict(embeddings)

    # ---- STEP 2: NAME each cluster once (simulated human judgment) ----
    print("\nStep 2: naming each cluster (simulating a one-time human decision)...")
    cluster_to_name = name_clusters(discovered_clusters, real_tags)

    # Every question now gets a DERIVED label: whatever its cluster was
    # named. This can be WRONG for individual questions, if clustering
    # put them in a cluster that got named after a different language.
    derived_labels = [cluster_to_name[c] for c in discovered_clusters]

    # How good is clustering+naming alone, before any classifier?
    # This compares the derived labels straight to the real tags.
    pre_classifier_accuracy = accuracy_score(real_tags, derived_labels)
    print(f"\nClustering+naming alone matches real tags: {pre_classifier_accuracy:.1%}")

    # ---- STEP 3: train a CLASSIFIER on the derived labels, not the real ones ----
    print("\nStep 3: training classifier on cluster-derived labels...")
    X_train, X_test, derived_train, derived_test, real_train, real_test, titles_train, titles_test = train_test_split(
        embeddings, derived_labels, real_tags, titles,
        test_size=0.2,
        random_state=42,
        stratify=derived_labels,
    )

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, derived_train)  # <-- trained on DERIVED labels, never real_tags

    predictions = classifier.predict(X_test)

    # ---- STEP 4: grade the WHOLE chain against real tags, just for us ----
    full_pipeline_accuracy = accuracy_score(real_test, predictions)
    print(f"\nFull pipeline (cluster -> name -> classify) vs. REAL tags: {full_pipeline_accuracy:.1%}")
    print("(compare this to the 75.0% we got in classify.py, which trained on real tags directly)")

    print("\nSample predictions (first 10 test questions):")
    for title, actual, predicted in list(zip(titles_test, real_test, predictions))[:10]:
        mark = "correct" if actual == predicted else "WRONG"
        print(f"  [{mark:7s}] actual={actual:6s} predicted={predicted:6s}  \"{title}\"")

    # Same live test as classify.py, for direct comparison. sys.argv[1]
    # is your own question, in quotes, if you passed one on the command
    # line -- otherwise fall back to the built-in example.
    if len(sys.argv) > 1:
        sample_question = sys.argv[1]
    else:
        sample_question = "How do I center a div both horizontally and vertically?"
    sample_embedding = model.encode([sample_question])
    sample_prediction = classifier.predict(sample_embedding)[0]
    print(f"\nNew question: \"{sample_question}\"")
    print(f"Predicted language (via the fully chained pipeline): {sample_prediction}")


if __name__ == "__main__":
    main()
