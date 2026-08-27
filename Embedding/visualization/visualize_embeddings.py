"""
visualize_embeddings.py

GOAL OF THIS FILE
------------------
compare_embeddings.py showed similarity as a single NUMBER (cosine score).
This file shows similarity as a PICTURE instead -- a scatter plot where
sentences with similar meaning land near each other, and unrelated
sentences land far apart.

THE PROBLEM: WE CAN'T SEE IN 384 DIMENSIONS
----------------------------------------------
Each sentence becomes a vector of 384 numbers (from all-MiniLM-L6-v2).
Humans can only look at 2D (flat) or 3D (a cube) plots. So before we can
draw anything, we need to squash 384 numbers down to just 2, PER
SENTENCE, while trying to keep sentences that were close in 384D still
close in 2D.

This squashing step is called DIMENSIONALITY REDUCTION. We use a
technique called PCA (Principal Component Analysis):

  - Imagine all our 384-dimensional sentence vectors as a cloud of
    points floating in space.
  - PCA finds the 2 directions along which that cloud is MOST spread
    out (these are called "principal components").
  - It then projects every point onto just those 2 directions.

This is lossy (we throw away 382 of the 384 numbers!), but it's a
genuinely useful approximation -- sentences that were similar in the
full 384D space usually still end up visually close in the 2D result.

THE PLOT
---------
We use Plotly (interactive charts) instead of a static image, so you
can hover over each dot in your browser and see exactly which sentence
it represents, and zoom/pan around.
"""

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import plotly.express as px


def load_model() -> SentenceTransformer:
    """Same small, fast embedding model used in compare_embeddings.py."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_sentences() -> list[tuple[str, str]]:
    """
    Returns a list of (topic, sentence) pairs.

    The "topic" label is ONLY used to color the dots in the plot so we
    can visually check whether the model's sense of "similar meaning"
    lines up with our own human idea of these being related sentences.
    The embedding model itself never sees the topic label -- it only
    ever looks at the sentence text.
    """
    return [
        # -- animals --
        ("animals", "The cat sat on the mat."),
        ("animals", "A dog barked loudly in the yard."),
        ("animals", "The bird flew across the sky."),
        ("animals", "My rabbit loves eating carrots."),
        # -- sports --
        ("sports", "She scored a goal in the final minute."),
        ("sports", "The team won the championship this year."),
        ("sports", "He trains every morning for the marathon."),
        ("sports", "The basketball game went into overtime."),
        # -- finance --
        ("finance", "The stock market crashed today."),
        ("finance", "Investors are worried about inflation."),
        ("finance", "The company reported record profits."),
        ("finance", "Interest rates rose again this quarter."),
        # -- weather --
        ("weather", "It rained heavily all afternoon."),
        ("weather", "The forecast predicts sunny skies tomorrow."),
        ("weather", "A storm is approaching the coast."),
        ("weather", "Temperatures dropped below freezing overnight."),
    ]


def main():
    model = load_model()
    topic_sentence_pairs = get_sentences()

    topics = [topic for topic, _ in topic_sentence_pairs]
    sentences = [sentence for _, sentence in topic_sentence_pairs]

    # Encode ALL sentences in one call (a "batch"), instead of looping and
    # calling .encode() once per sentence. This is both faster (the model
    # can process many sentences at once more efficiently) and gives us
    # back a single array of shape (16, 384): 16 sentences, 384 numbers each.
    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}  (sentences x dimensions)")

    # PCA: squash 384 dimensions down to 2, so we can plot them.
    # n_components=2 means "give me back only 2 numbers per sentence".
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)  # shape becomes (16, 2)
    print(f"Reduced shape:    {reduced.shape}  (sentences x 2)")

    # explained_variance_ratio_ tells us how much of the ORIGINAL
    # information (spread/variance) survived the squash from 384 -> 2
    # dimensions. E.g. [0.35, 0.20] means the 1st axis of our plot
    # captures 35% of the original structure, the 2nd axis another 20%
    # -- so together this 2D picture represents ~55% of the full picture.
    # (The rest was thrown away by compressing 384 numbers into 2.)
    variance_captured = pca.explained_variance_ratio_.sum() * 100
    print(f"Variance captured by the 2D plot: {variance_captured:.1f}%\n")

    x_coords = reduced[:, 0]  # first PCA axis, for every sentence
    y_coords = reduced[:, 1]  # second PCA axis, for every sentence

    # Build the interactive scatter plot:
    #   - one dot per sentence
    #   - color = topic label (so same-topic sentences share a color)
    #   - hovering over a dot shows the full original sentence text
    figure = px.scatter(
        x=x_coords,
        y=y_coords,
        color=topics,
        hover_name=sentences,
        labels={"x": "PCA component 1", "y": "PCA component 2", "color": "Topic"},
        title="Sentence embeddings, reduced from 384D to 2D with PCA",
    )

    output_path = "embedding_plot.html"
    figure.write_html(output_path)
    print(f"Saved interactive plot to: {output_path}")
    print("Open that file in a browser to explore it.")


if __name__ == "__main__":
    main()
