"""
cluster.py

GOAL OF THIS FILE
------------------
Take the 400 real questions.json questions, embed them, and see if an
UNSUPERVISED algorithm (KMeans) can automatically rediscover the 4
language groups (python/r/html/css) -- WITHOUT ever being told the
real tags. We only use the real tags afterward, to check its work.

CLUSTERING VS. WHAT WE DID BEFORE
------------------------------------
In visualization/, we already knew the topic of every sentence, and
just used PCA/t-SNE to CHECK whether the embeddings visually agreed
with labels we already had.

Clustering is different: the algorithm gets ONLY the embeddings, no
labels at all, and has to invent its own groups from scratch, purely
based on which points are close together in 384D space. We use
KMeans, which works like this:
  1. Pick K random points to be initial "cluster centers"
  2. Assign every question to whichever center is nearest
  3. Move each center to the actual middle (average) of the questions
     now assigned to it
  4. Repeat steps 2-3 until the centers stop moving much

K = 4 here because we expect 4 language groups -- but note that in a
REAL unsupervised scenario you often don't know K in advance. We're
allowed to "cheat" a little here because we already know there should
be 4 groups, purely to make this easier to evaluate.

AFTER clustering, we bring the real tags back JUST to draw two plots
side by side: one colored by what KMeans guessed, one colored by the
real tag. If clustering worked well, the two pictures should look
similar.
"""

import json

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


def load_questions() -> list[dict]:
    with open("questions.json") as f:
        return json.load(f)


def main():
    questions = load_questions()
    titles = [q["title"] for q in questions]
    real_tags = [q["tag"] for q in questions]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(titles)} questions...")
    embeddings = model.encode(titles)

    # KMeans: discover 4 groups using ONLY the embeddings, no labels.
    # random_state fixes the initial random cluster centers, so results
    # are reproducible instead of slightly different every run.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    discovered_clusters = kmeans.fit_predict(embeddings)  # one cluster number (0-3) per question
    # discovered_clusters is now e.g. [2, 0, 0, 3, 1, ...] -- just cluster
    # NUMBERS with no inherent meaning. Cluster "0" isn't necessarily
    # "python" -- KMeans doesn't know what these groups represent, only
    # that they're 4 groups of similar things.

    # Reduce to 2D purely so we can plot -- same PCA step as before.
    reduced = PCA(n_components=2).fit_transform(embeddings)
    x_coords = reduced[:, 0]
    y_coords = reduced[:, 1]

    # Two plots side by side: what KMeans discovered vs. reality.
    figure = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("What KMeans discovered (no labels given)", "Real tags (ground truth)"),
    )

    discovered_fig = px.scatter(
        x=x_coords, y=y_coords,
        color=[str(c) for c in discovered_clusters],
        hover_name=titles,
    )
    real_fig = px.scatter(
        x=x_coords, y=y_coords,
        color=real_tags,
        hover_name=titles,
    )

    for trace in discovered_fig.data:
        figure.add_trace(trace, row=1, col=1)
    for trace in real_fig.data:
        figure.add_trace(trace, row=1, col=2)

    figure.update_layout(title="Clustering StackOverflow questions: discovered vs. real language")

    output_path = "cluster_plot.html"
    figure.write_html(output_path)
    print(f"Saved comparison plot to: {output_path}")


if __name__ == "__main__":
    main()
