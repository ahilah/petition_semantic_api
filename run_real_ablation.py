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
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

try:
    from engine import DomainContextEngine
except ImportError:
    DomainContextEngine = None


W_SEM = 0.65
W_DOM = 0.25
W_GEOD = 0.10


def clean_str(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).lower()
    s = re.sub(r"[^\w\sа-яіїєґ0-9-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def tfidf_scores(df: pd.DataFrame) -> np.ndarray:
    t1 = [clean_str(x) for x in df["title_1"]]
    t2 = [clean_str(x) for x in df["title_2"]]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vec.fit(t1 + t2)
    return np.clip(
        np.asarray(vec.transform(t1).multiply(vec.transform(t2)).sum(axis=1)).ravel(),
        0.0,
        1.0,
    )


def e5_scores(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    q = ["query: " + str(t) for t in df["title_1"]]
    c = ["passage: " + str(t) for t in df["title_2"]]
    emb_q = model.encode(q, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    emb_c = model.encode(c, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    return util.pairwise_cos_sim(emb_q, emb_c).cpu().numpy()


def extract_factors(res: dict) -> tuple[float, float]:
    """Безпечно дістає доменний та геопросторовий коефіцієнти зі словника відповіді engine."""
    p_dom = 1.0
    for key in ("domain_score", "domain_factor", "p_dom", "ner_score"):
        if key in res and res[key] is not None:
            p_dom = float(res[key])
            break
    if p_dom == 1.0 and "domain_penalty" in res and res["domain_penalty"] is not None:
        p_dom = max(0.0, 1.0 - float(res["domain_penalty"]))

    p_geod = 1.0
    for key in ("geo_score", "geo_factor", "p_geod", "geodesic_score"):
        if key in res and res[key] is not None:
            p_geod = float(res[key])
            break
    if p_geod == 1.0 and "geo_penalty" in res and res["geo_penalty"] is not None:
        p_geod = max(0.0, 1.0 - float(res["geo_penalty"]))

    return np.clip(p_dom, 0.0, 1.0), np.clip(p_geod, 0.0, 1.0)


def integral_scores(
    sub_df: pd.DataFrame,
    base_sim: np.ndarray,
    include_ner: bool,
    include_geo: bool,
) -> np.ndarray:
    if DomainContextEngine is None or (not include_ner and not include_geo):
        return np.copy(base_sim)

    denom = W_SEM + W_DOM + W_GEOD
    out = np.zeros(len(sub_df), dtype=float)

    for i, (_, row) in enumerate(sub_df.iterrows()):
        s_sem = float(base_sim[i])
        cand = {
            "title": str(row.get("title_2", "")),
            "text": str(row.get("text_2", "")),
            "authority": str(row.get("authority_2", "")),
        }
        res = DomainContextEngine.evaluate_match(
            query_text=str(row.get("title_1", "")),
            candidate_row=cand,
            base_semantic_score=s_sem,
        )
        p_dom, p_geod = extract_factors(res)
        eff_dom = p_dom if include_ner else 1.0
        eff_geod = p_geod if include_geo else 1.0

        s_int = (W_SEM * s_sem + W_DOM * (s_sem * eff_dom) + W_GEOD * (s_sem * eff_geod)) / denom
        out[i] = float(np.clip(s_int, 0.0, 1.0))

    return out


def best_th_dev(y_dev: np.ndarray, s_dev: np.ndarray) -> float:
    best_th, best_f1 = 0.80, -1.0
    for th in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_dev, (s_dev >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return round(best_th, 2)


def summarize(cfg_name: str, y_test: np.ndarray, s_test: np.ndarray, th: float) -> dict:
    pred = (s_test >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "Конфігурація": cfg_name,
        "Поріг (dev)": th,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "Precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }


def plot_ablation(rows: list, out_dir: str) -> None:
    names = [r["Конфігурація"] for r in rows]
    x = np.arange(len(names))
    w = 0.18

    plt.figure(figsize=(9.5, 5))
    for idx, m in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
        plt.bar(x + (idx - 1.5) * w, [r[m] for r in rows], width=w, label=m)
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel("Значення метрики")
    plt.title("Абляційне порівняння конфігурацій на вибірці test (195 пар)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_bars_test.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, r in zip(axes.ravel(), rows):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["UNIQUE (0)", "DUPLICATE (1)"])
        ax.set_yticklabels(["UNIQUE (0)", "DUPLICATE (1)"])
        ax.set_title(f"{r['Конфігурація']} (θ={r['Поріг (dev)']}) | F1={r['F1-Score']}")
        for rr in range(2):
            for cc in range(2):
                txt_col = "white" if cm[rr, cc] > cm.max() / 2 else "black"
                ax.text(cc, rr, str(cm[rr, cc]), ha="center", va="center", color=txt_col, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_cm_test.png"), dpi=300)
    plt.close()


def main():
    out_dir = "outputs_ablation_v2"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel("ground_truth_300_balanced.xlsx")
    y = df["is_similar"].astype(int).values

    df_dev, df_test, y_dev, y_test = train_test_split(
        df, y, train_size=105, test_size=195, stratify=y, random_state=42
    )

    model = SentenceTransformer("intfloat/multilingual-e5-small")

    configs = [
        ("TF-IDF / BM25", tfidf_scores(df_dev), tfidf_scores(df_test)),
    ]
    e5_dev = e5_scores(df_dev, model)
    e5_test = e5_scores(df_test, model)
    configs.append(("Лише E5", e5_dev, e5_test))
    configs.append((
        "E5 + NER",
        integral_scores(df_dev, e5_dev, include_ner=True, include_geo=False),
        integral_scores(df_test, e5_test, include_ner=True, include_geo=False),
    ))
    configs.append((
        "E5 + NER + гео",
        integral_scores(df_dev, e5_dev, include_ner=True, include_geo=True),
        integral_scores(df_test, e5_test, include_ner=True, include_geo=True),
    ))

    summary_rows = []
    for cfg_name, s_dev, s_test in configs:
        th = best_th_dev(y_dev, s_dev)
        summary_rows.append(summarize(cfg_name, y_test, s_test, th))

    out_df = pd.DataFrame(summary_rows)
    print("\n=== ТАБЛИЦЯ АБЛЯЦІЇ НА TEST (195 ПАР; ПОРОГИ ПІДІБРАНО НА DEV) ===")
    print(out_df.to_string(index=False))
    out_df.to_csv(os.path.join(out_dir, "ablation_summary.csv"), index=False, encoding="utf-8-sig")
    plot_ablation(summary_rows, out_dir)


if __name__ == "__main__":
    main()