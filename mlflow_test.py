import mlflow
import json
import numpy as np

from Clusterize import (
    comment_flattening,
    data_prep,
    data_clustering_tsne,
    make_dataframe_and_clusters,
    plot_clusters
)


def count_clusters(labels):
    """Count clusters excluding noise (-1)."""
    return len(set(labels)) - (1 if -1 in labels else 0)


def optimize_perplexity(
    embeddings,
    clean_entries,
    eps,
    min_samples,
    perplexity_values
):
    mlflow.set_experiment("tsne_perplexity_optimization")

    best_result = None
    best_cluster_count = float("inf")

    for perplexity in perplexity_values:
        with mlflow.start_run(run_name=f"perplexity_{perplexity}"):

            print(f"\n🔍 Testing perplexity = {perplexity}")

            emb_2d, labels = data_clustering_tsne(
                embeddings=embeddings,
                eps=eps,
                min_samples=min_samples,
                perplexity=perplexity
            )

            n_clusters = count_clusters(labels)

            # ---- MLflow logging ----
            mlflow.log_param("perplexity", perplexity)
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)
            mlflow.log_metric("num_clusters", n_clusters)

            print(f"➡️ clusters = {n_clusters}")

            if n_clusters < best_cluster_count:
                best_cluster_count = n_clusters
                best_result = {
                    "perplexity": perplexity,
                    "emb_2d": emb_2d,
                    "labels": labels
                }

    print(
        f"\n✅ Best perplexity: {best_result['perplexity']} "
        f"with {best_cluster_count} clusters"
    )

    return best_result


def main():
    EPS = 1.5
    MIN_SAMPLES = 5

    PERPLEXITY_GRID = [5, 10, 20, 30, 40, 50, 70, 100]

    entries = comment_flattening("data/comments_with_embeddings.json")
    embeddings, clean_entries = data_prep(entries)

    best = optimize_perplexity(
        embeddings=embeddings,
        clean_entries=clean_entries,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        perplexity_values=PERPLEXITY_GRID
    )

    df, clusters = make_dataframe_and_clusters(
        best["emb_2d"],
        best["labels"],
        clean_entries
    )

    plot_clusters(df, clusters)


if __name__ == "__main__":
    main()