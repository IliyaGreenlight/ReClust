import mlflow
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from Clusterize import (
    comment_flattening,
    data_prep,
    data_clustering_tsne,
)

from Visualize import (
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

def extract_topic_with_model(model, tokenizer, comments):
    prompt = f"""
    Context:
    You are given several YouTube comments from the same discussion topic. 
    Comments are provided in the following format: ["comment 1", ..., "comment 5"]
    Youtube video has the title: All Best Cat Memes

    Comments:
    {comments}

    Task:
    Return a concise topic label (1 to 4 words)
    No punctuation.
    No explanation.
    Only the label.
    """.strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.2,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean output (VERY important)
    result = result.split("Task:")[-1].strip()
    return result

def optimize_model(model_names, clusters, experiment_name="topic_extraction"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    for model_name in model_names:
        with mlflow.start_run(run_name=model_name):

            try:
                # Load model ONCE per run
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )

                all_results = {}
                total_length = 0
                valid_format_count = 0

                for cluster_id, comments in clusters.items():
                    prediction = extract_topic_with_model(model, tokenizer, comments)

                    word_count = len(prediction.split())

                    # Heuristic checks
                    is_valid = (
                        1 <= word_count <= 4 and
                        prediction.replace(" ", "").isalnum()
                    )

                    if is_valid:
                        valid_format_count += 1

                    total_length += word_count

                    all_results[cluster_id] = {
                        "prediction": prediction,
                        "word_count": word_count,
                        "valid_format": is_valid
                    }

                # 📊 Log useful metrics
                avg_length = total_length / len(clusters)
                format_ratio = valid_format_count / len(clusters)

                mlflow.log_metric("avg_topic_length", avg_length)
                mlflow.log_metric("format_compliance", format_ratio)

                # 🧾 Log parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("num_clusters", len(clusters))

                # 💾 Save predictions
                mlflow.log_dict(all_results, "predictions.json")

            except Exception as e:
                mlflow.log_param("error", str(e))
                continue

def main():


    MODELS = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "bigscience/bloomz-7b1-mt"
    ]

    with open("data/cluster_representatives.json", "r", encoding="utf-8") as f:
        clusters = json.load(f)

    optimize_model(MODELS, clusters)

if __name__ == "__main__":
    main()