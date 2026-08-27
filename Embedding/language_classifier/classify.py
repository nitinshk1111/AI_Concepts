"""
classify.py

GOAL OF THIS FILE
------------------
Train an actual classifier that guesses which language (python, r,
html, css) a NEW, never-before-seen question is about -- using the
question's embedding as its input.

CLUSTERING (cluster.py) VS. CLASSIFICATION (this file)
-----------------------------------------------------------
Clustering: no labels given at all, algorithm invents its own groups.
Classification: we DO give the algorithm labels to learn from (this is
called "supervised learning"), and afterward it can predict the label
of new, unlabeled examples it has never seen.

THE TRAIN/TEST SPLIT -- WHY WE NEVER TEST ON DATA WE TRAINED ON
--------------------------------------------------------------------
If we let the model learn from ALL 400 questions and then quiz it on
those exact same 400 questions, it could theoretically just memorize
answers rather than actually learning the pattern -- like grading a
student using the exact questions they already saw the answer key for.

So we split our 400 questions into two separate piles BEFORE training:
  - TRAIN set (80%, ~320 questions): the model learns from these
  - TEST set (20%, ~80 questions): held back, the model NEVER sees
    these during training. Afterward we quiz it on this set and check
    how many it gets right. That accuracy is a fair, honest estimate
    of how well it'll do on genuinely new questions in the future.

THE CLASSIFIER: K-NEAREST NEIGHBORS (KNN)
----------------------------------------------
KNN's idea is simple: to classify a new question, look at its K
closest neighbors (by embedding distance) among the TRAINING questions,
and predict whichever language is most common among those neighbors.
E.g. with K=3: find the 3 most similar training questions to this new
one; if 2 of them are "python" and 1 is "r", predict "python".
"""

import json
import sys

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_questions() -> list[dict]:
    with open("questions.json") as f:
        return json.load(f)


def main():
    questions = load_questions()
    titles = [q["title"] for q in questions]
    labels = [q["tag"] for q in questions]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(titles)} questions...")
    embeddings = model.encode(titles)

    # Split BEFORE training. random_state fixes which questions land in
    # which pile, so this is reproducible instead of random every run.
    # stratify=labels keeps the same 25/25/25/25 language balance in
    # both the train and test piles, instead of risking e.g. a test set
    # that accidentally has zero "r" questions in it.
    X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(
        embeddings, labels, titles,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"Train set: {len(X_train)} questions   Test set: {len(X_test)} questions")

    # Train the classifier: "study" the training embeddings + their
    # known labels. n_neighbors=5 means: look at the 5 closest training
    # questions when making each guess.
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)

    # Quiz it on the test set -- questions it has NEVER seen labels for.
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy on {len(X_test)} held-out test questions: {accuracy:.1%}")

    # Per-language breakdown: precision/recall for each of the 4 tags,
    # so we can see if e.g. html and css get confused with each other
    # more than python and r do (the concern we predicted earlier).
    print("\nPer-language breakdown:")
    print(classification_report(y_test, predictions))

    # Show a handful of individual test examples: question, real answer,
    # model's guess -- so mistakes (if any) are visible, not just a
    # single summary number.
    print("Sample predictions (first 10 test questions):")
    for title, actual, predicted in list(zip(titles_test, y_test, predictions))[:10]:
        mark = "correct" if actual == predicted else "WRONG"
        print(f"  [{mark:7s}] actual={actual:6s} predicted={predicted:6s}  \"{title}\"")

    # Finally: try it on a brand new, made-up question that was never
    # part of the dataset at all. sys.argv is just the list of words you
    # typed on the command line; sys.argv[0] is always the script name
    # itself, so sys.argv[1] is the first thing typed AFTER the filename
    # -- here, your own question in quotes, if you gave one.
    if len(sys.argv) > 1:
        sample_question = sys.argv[1]
    else:
        sample_question = "How do I center a div both horizontally and vertically?"
    sample_embedding = model.encode([sample_question])
    sample_prediction = classifier.predict(sample_embedding)[0]
    print(f"\nNew question: \"{sample_question}\"")
    print(f"Predicted language: {sample_prediction}")


if __name__ == "__main__":
    main()
