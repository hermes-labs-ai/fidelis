"""
Step 4: Train a learned router to replace regex classification.
Uses Phase 0 per-question data as training labels.
Label: use_llm = 1 if s2_hit > s1_hit, else 0.
Features: nomic query embedding + query length + heuristic features.

5-fold CV. Compare logistic regression vs random forest.
"""
import json
import math
import urllib.request
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

BENCH = Path(__file__).parent.parent
OLLAMA_URL = "http://localhost:11434"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed via nomic-embed-text. Returns (N, 768) array."""
    all_vecs = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i+50]
        sanitized = [t[:2000] if len(t) > 2000 else t for t in batch]
        body = json.dumps({"model": "nomic-embed-text", "input": sanitized}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        all_vecs.extend(data["embeddings"])
    return np.array(all_vecs)


def extract_features(questions: list[str], embeddings: np.ndarray) -> np.ndarray:
    """Build feature matrix: embeddings + query-level heuristics."""
    n = len(questions)
    # Heuristic features
    heuristics = np.zeros((n, 6))
    for i, q in enumerate(questions):
        ql = q.lower()
        heuristics[i, 0] = len(q)  # query length
        heuristics[i, 1] = sum(1 for p in ["how many", "total", "count"] if p in ql)  # counting signal
        heuristics[i, 2] = sum(1 for p in ["first", "before", "after", "order", "earliest"] if p in ql)  # temporal signal
        heuristics[i, 3] = sum(1 for p in ["you told", "you said", "you suggested", "you mentioned"] if p in ql)  # assistant ref
        heuristics[i, 4] = sum(1 for p in ["what is my", "where did i", "what did i"] if p in ql)  # direct factual
        heuristics[i, 5] = q.count("?")  # question marks

    return np.hstack([embeddings, heuristics])


def main():
    # Load Phase 0 baseline data
    pq = json.load(open(BENCH / "runs/baseline/per_question.json"))
    data = json.load(open(Path.home() / "Documents/projects/LongMemEval/data/longmemeval_s_cleaned.json"))
    data = [e for e in data if "_abs" not in e["question_id"]]

    questions = [data[q["qi"]]["question"] for q in pq]

    # Labels: 1 if LLM helps (s2 hit, s1 miss), 0 if LLM hurts or neutral
    # More precisely: 1 if s2 > s1 for this question
    labels = np.array([
        1 if (q["s2_hit_at_1"] and not q["s1_hit_at_1"]) else 0
        for q in pq
    ])

    print(f"Embedding {len(questions)} queries...")
    embeddings = embed_texts(["search_query: " + q for q in questions])
    X = extract_features(questions, embeddings)
    y = labels

    print(f"Features: {X.shape[1]} (768 embedding + 6 heuristic)")
    print(f"Labels: {sum(y)}/{len(y)} use_llm ({sum(y)/len(y)*100:.0f}%)")
    print()

    # Also compute: what would regex router get (for comparison baseline)
    from collections import Counter
    skip_patterns = ["you told me", "you suggested", "you recommended",
                     "you mentioned", "you said", "you explained",
                     "our previous conversation", "our last chat", "remind me what you"]
    temporal_patterns = ["which happened first", "which did i do first",
                         "how many days", "how many weeks", "how many months",
                         "before or after", "what order", "what was the date",
                         "which event", "which trip", "order of the", "from earliest",
                         "from first", "most recently", "a week ago", "two weeks ago",
                         "a month ago", "last saturday", "last sunday", "last weekend",
                         "last monday", "last tuesday", "graduated first", "started first",
                         "finished first", "did i do first", "did i attend first",
                         "who did i go with to the"]
    counting_patterns = ["how many", "total number", "how much total",
                         "in total", "altogether", "combined"]

    regex_preds = []
    for q in questions:
        ql = q.lower()
        if any(p in ql for p in skip_patterns):
            regex_preds.append(0)
        elif any(p in ql for p in temporal_patterns) or any(p in ql for p in counting_patterns):
            regex_preds.append(1)
        else:
            regex_preds.append(0)
    regex_preds = np.array(regex_preds)

    # Compute regex router R@1
    regex_r1_hits = 0
    for i, q in enumerate(pq):
        if regex_preds[i] == 1:
            regex_r1_hits += q["s2_hit_at_1"]
        else:
            regex_r1_hits += q["s1_hit_at_1"]
    regex_r1 = regex_r1_hits / len(pq)
    print(f"Regex router R@1: {regex_r1:.1%}")

    # 5-fold CV for both models
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, C=1.0)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]:
        fold_r1s = []
        fold_accs = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X[train_idx], y[train_idx])
            preds = model_clone.predict(X[test_idx])

            # Compute R@1 for this fold
            r1_hits = 0
            for i, idx in enumerate(test_idx):
                if preds[i] == 1:
                    r1_hits += pq[idx]["s2_hit_at_1"]
                else:
                    r1_hits += pq[idx]["s1_hit_at_1"]
            fold_r1 = r1_hits / len(test_idx)
            fold_r1s.append(fold_r1)

            # Classification accuracy
            fold_acc = np.mean(preds == y[test_idx])
            fold_accs.append(fold_acc)

        mean_r1 = np.mean(fold_r1s)
        std_r1 = np.std(fold_r1s)
        mean_acc = np.mean(fold_accs)

        print(f"\n{name}:")
        print(f"  CV R@1: {mean_r1:.1%} ± {std_r1:.1%}")
        print(f"  CV acc: {mean_acc:.1%}")
        print(f"  vs regex: {mean_r1 - regex_r1:+.1%}")
        print(f"  Ship? {'YES' if mean_r1 >= regex_r1 and std_r1 < 0.02 else 'NO'} (criterion: R@1 >= regex AND std < 2pp)")

    # Train final model on all data for export
    best_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    best_model.fit(X, y)

    # Feature importance (top 10)
    importances = best_model.feature_importances_
    feature_names = [f"emb_{i}" for i in range(768)] + ["query_len", "counting", "temporal", "asst_ref", "direct_factual", "question_marks"]
    top_idx = np.argsort(importances)[-10:][::-1]
    print(f"\nTop 10 features (RandomForest):")
    for idx in top_idx:
        print(f"  {feature_names[idx]:20s} importance={importances[idx]:.4f}")

    # Save model
    import pickle
    model_path = BENCH / "phase-2/learned_router.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "feature_names": feature_names}, f)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
