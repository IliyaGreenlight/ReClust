import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

def comment_flattening(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        comments = json.load(f)
    all_entries = []
    for c in comments:
        if "embedding" in c:
            all_entries.append({
                "type": "comment",
                "body": c.get("body", ""),
                "embedding": c["embedding"],
                "author": c["author_name"],
            })
        # if "replies" in c:
        #     for r in c["replies"]:
        #         if "embedding" in r:
        #             all_entries.append({
        #                 "type": "reply",
        #                 "body": r.get("body", ""),
        #                 "embedding": r["embedding"]
        #             })
        
    print(f"Loaded {len(all_entries)} entries (comments + replies).")
    return all_entries

def data_prep(all_entries):
    clean_entries = []
    clean_embeddings = []

    for x in all_entries:
        emb = x.get("embedding")
        if isinstance(emb, list) and len(emb) > 0 and all(isinstance(v, (int, float)) for v in emb):
            clean_entries.append(x)
            clean_embeddings.append(np.array(emb, dtype=np.float32))

    # Check for consistent dimensionality
    dims = [len(e) for e in clean_embeddings]
    unique_dims = set(dims)
    if len(unique_dims) > 1:
        print(f"Warning: inconsistent embedding sizes found: {unique_dims}")
        # Keep only the most common dimension
        from collections import Counter
        common_dim = Counter(dims).most_common(1)[0][0]
        clean_entries = [x for x, e in zip(clean_entries, clean_embeddings) if len(e) == common_dim]
        clean_embeddings = [e for e in clean_embeddings if len(e) == common_dim]

    embeddings = np.vstack(clean_embeddings)
    print(f"Using {len(clean_entries)} valid entries (shape = {embeddings.shape})")
    return embeddings, clean_entries

def data_clustering_tsne(embeddings, eps, min_samples, perplexity):
     # === t-SNE REDUCTION ===
    print("Running t-SNE for visualization (this may take a few minutes)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    # === DBSCAN ON 2D POINTS ===
    print("Running DBSCAN on t-SNE results...")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(emb_2d)
    labels = db.labels_
    return emb_2d, labels