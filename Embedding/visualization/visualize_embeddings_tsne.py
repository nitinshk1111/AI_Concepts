"""
visualize_embeddings_tsne.py

GOAL OF THIS FILE
------------------
This is the same idea as visualize_embeddings.py (turn sentences into
384D vectors, then squash them down to 2D so we can plot them) but
using a DIFFERENT dimensionality-reduction technique: t-SNE, instead
of PCA. Compare the two plots side by side once you run both scripts.

PCA vs t-SNE -- WHAT'S ACTUALLY DIFFERENT
--------------------------------------------
PCA (see visualize_embeddings.py) does ONE simple geometric operation:
it finds the 2 directions with the most overall spread in the 384D
data, and projects every point onto just those 2 directions. Think of
it like shining a light through a 3D object to cast a flat shadow.

t-SNE works differently:
  1. In the original 384D space, it looks at each sentence and figures
     out which other sentences are its close neighbors.
  2. It then iteratively rearranges points in 2D so that those SAME
     neighbor relationships are preserved -- close things stay close,
     far things stay far -- even if that means bending and stretching
     space to make it fit.

This usually produces more visually separated, crisper-looking
clusters than PCA. The tradeoff: it's slower, results can look
slightly different each time you run it (unless you fix random_state,
which we do below), and -- importantly -- the DISTANCE BETWEEN two
separate clusters doesn't mean anything with t-SNE. Only "are these
points close neighbors or not" is meaningful. (With PCA, by contrast,
the axes have real mathematical meaning: how much of the original
spread/variance each axis captures.)
"""

from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.express as px


def load_model() -> SentenceTransformer:
    """Same small, fast embedding model used in the other scripts."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_sentences() -> list[tuple[str, str]]:
    """
    Same (topic, sentence) pairs as visualize_embeddings.py, kept as a
    duplicate here (rather than imported) so this file can be read top
    to bottom on its own, without needing to open the other script too.

    As before: the "topic" label is ONLY used to color dots in the plot.
    The embedding model never sees it -- it only ever reads the
    sentence text.
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

    # Same batch-encoding step as before: 16 sentences -> (16, 384) array.
    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}  (sentences x dimensions)")

    # t-SNE: squash 384 dimensions down to 2, but by preserving NEIGHBOR
    # relationships instead of overall variance (see docstring above).
    #
    # perplexity: roughly "how many neighbors should I consider for each
    # point when deciding what counts as close?" It MUST be smaller than
    # the number of sentences we have. The common default (30) assumes
    # much larger datasets; with only 16 sentences we use a small value.
    #
    # random_state: t-SNE starts from a random layout and iteratively
    # improves it, so results vary slightly run to run. Fixing
    # random_state makes this script produce the exact same plot every
    # time, which is easier to learn from.
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = tsne.fit_transform(embeddings)  # shape becomes (16, 2)
    print(f"Reduced shape:    {reduced.shape}  (sentences x 2)")

    # NOTE: there is no equivalent of PCA's explained_variance_ratio_
    # here. t-SNE doesn't preserve "variance" in a measurable way --
    # it only tries to preserve local neighbor relationships, so there's
    # no single number that captures "how much information survived."

    x_coords = reduced[:, 0]
    y_coords = reduced[:, 1]

    figure = px.scatter(
        x=x_coords,
        y=y_coords,
        color=topics,
        hover_name=sentences,
        labels={"x": "t-SNE dimension 1", "y": "t-SNE dimension 2", "color": "Topic"},
        title="Sentence embeddings, reduced from 384D to 2D with t-SNE",
    )

    output_path = "embedding_plot_tsne.html"
    figure.write_html(output_path)
    print(f"Saved interactive plot to: {output_path}")
    print("Open that file in a browser to explore it.")


if __name__ == "__main__":
    main()
