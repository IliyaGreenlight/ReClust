import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def get_embedding(text: str, model):
    if not text or not text.strip():
        return []
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()

def calculate_embedding(model_name: str):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("Loading comments.json...")
    with open("data/comments.json", "r", encoding="utf-8") as f:
        comments = json.load(f)


    print("Generating embeddings for comments and replies...")
    for comment in tqdm(comments, desc="Processing comments"):
        # Comment body
        comment["embedding"] = get_embedding(comment.get("body", ""), model=model)

        # Replies
        if "replies" in comment and isinstance(comment["replies"], list):
            for reply in comment["replies"]:
                reply["embedding"] = get_embedding(reply.get("body", ""), model=model)

    print(f"Saving embedings...")
    with open("data/comments_with_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    return comments

if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    calculate_embedding(model_name=MODEL_NAME)

