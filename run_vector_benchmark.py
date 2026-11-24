import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util


def clean_str(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).lower()
    s = re.sub(r"[^\w\sа-яіїєґ0-9-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def tfidf_pair_scores(df: pd.DataFrame) -> np.ndarray:
    t1 = [clean_str(x) for x in df["title_1"]]
    t2 = [clean_str(x) for x in df["title_2"]]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vec.fit(t1 + t2)
    m1 = vec.transform(t1)
    m2 = vec.transform(t2)
    return np.clip(np.asarray(m1.multiply(m2).sum(axis=1)).ravel(), 0.0, 1.0)


def encode_model_scores(df: pd.DataFrame, model_name: str, is_e5: bool = False) -> np.ndarray:
    model = SentenceTransformer(model_name)
    if is_e5:
        q = ["query: " + str(t) for t in df["title_1"]]
        c = ["passage: " + str(t) for t in df["title_2"]]
    else:
        q = [str(t) for t in df["title_1"]]
        c = [str(t) for t in df["title_2"]]
    emb_q = model.encode(q, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    emb_c = model.encode(c, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return util.pairwise_cos_sim(emb_q, emb_c).cpu().numpy()


def tune_threshold_dev(y_dev: np.ndarray, scores_dev: np.ndarray) -> float:
    grid = np.arange(0.05, 0.96, 0.01)
    best_th = 0.50
    best_f1 = -1.0
    for th in grid:
        pred = (scores_dev >= th).astype(int)
        score = f1_score(y_dev, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return round(best_th, 2)


def eval_on_test(name: str, y_test: np.ndarray, scores_test: np.ndarray, th: float) -> dict:
    pred = (scores_test >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) > 1 else np.nan
    return {
        "Модель": name,
        "theta_dev": th,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "Precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "Specificity": round(float(spec), 4),
        "ROC-AUC": round(float(auc), 4) if not np.isnan(auc) else None,
        "_scores": scores_test,
    }


def save_plots(y_dev: np.ndarray, dev_curves: dict, y_test: np.ndarray, test_rows: list, out_dir: str) -> None:
    grid = np.arange(0.05, 0.96, 0.01)

    plt.figure(figsize=(8, 5))
    for label, s_dev in dev_curves.items():
        f1s = [f1_score(y_dev, (s_dev >= th).astype(int), zero_division=0) for th in grid]
        plt.plot(grid, f1s, label=label, linewidth=2)
    plt.xlabel("Поріг θ")
    plt.ylabel("F1-Score (dev, 105 пар)")
    plt.title("Підбір порогу за F1 на вибірці dev")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bench_f1_dev.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for row in test_rows:
        fpr, tpr, _ = roc_curve(y_test, row["_scores"])
        plt.plot(fpr, tpr, linewidth=2, label=f"{row['Модель']} (AUC={row['ROC-AUC']})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Випадковий вибір")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC-криві моделей на контрольній вибірці test (195 пар)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bench_roc_test.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, row in zip(axes.ravel(), test_rows):
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["UNIQUE (0)", "DUPLICATE (1)"])
        ax.set_yticklabels(["UNIQUE (0)", "DUPLICATE (1)"])
        ax.set_title(f"{row['Модель']} (θ={row['theta_dev']}) | F1={row['F1']}")
        for r in range(2):
            for c in range(2):
                col = "white" if cm[r, c] > cm.max() / 2 else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color=col, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bench_cm_test.png"), dpi=300)
    plt.close()


def main():
    out_dir = "outputs_benchmark"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel("ground_truth_300_balanced.xlsx")
    y = df["is_similar"].astype(int).values

    df_dev, df_test, y_dev, y_test = train_test_split(
        df, y, train_size=105, test_size=195, stratify=y, random_state=42
    )

    models_spec = [
        ("TF-IDF / BM25", "tfidf", False),
        ("MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False),
        ("LaBSE", "sentence-transformers/LaBSE", False),
        ("multilingual-e5-small", "intfloat/multilingual-e5-small", True),
    ]

    dev_curves = {}
    test_rows = []
    for label, src, is_e5 in models_spec:
        if src == "tfidf":
            s_dev = tfidf_pair_scores(df_dev)
            s_test = tfidf_pair_scores(df_test)
        else:
            s_dev = encode_model_scores(df_dev, src, is_e5=is_e5)
            s_test = encode_model_scores(df_test, src, is_e5=is_e5)
        th = tune_threshold_dev(y_dev, s_dev)
        dev_curves[label] = s_dev
        test_rows.append(eval_on_test(label, y_test, s_test, th))

    res_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in test_rows])
    print("\n=== БЕНЧМАРКІНГ НА TEST (195 ПАР; ПОРОГИ З DEV, 105 ПАР) ===")
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(out_dir, "benchmark_test.csv"), index=False, encoding="utf-8-sig")
    save_plots(y_dev, dev_curves, y_test, test_rows, out_dir)


if __name__ == "__main__":
    main()