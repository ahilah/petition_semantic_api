import inspect
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for path_candidate in (BASE_DIR, PROJECT_ROOT):
    if path_candidate not in sys.path:
        sys.path.insert(0, path_candidate)

import engine as eng_mod
from engine import DomainContextEngine

MODEL_NAME = "intfloat/multilingual-e5-small"
MIN_DEV_PRECISION = 0.75


def resolve_data_path(filename: str) -> str:
    """Знаходить файл у data/ або поруч зі скриптом."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(PROJECT_ROOT, "data", filename),
        os.path.join(PROJECT_ROOT, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Не знайдено файл датасету: {filename}")


def ensure_tfidf_scores(df_dev: pd.DataFrame, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Повертає TF-IDF косинусну подібність для DEV і TEST.
    Якщо колонка 'tfidf_score' вже є у файлах — бере її,
    інакше навчає TfidfVectorizer тільки на DEV і трансформує DEV та TEST без витоку.
    """
    if "tfidf_score" in df_dev.columns and "tfidf_score" in df_test.columns:
        return (
            df_dev["tfidf_score"].fillna(0.0).to_numpy(dtype=float),
            df_test["tfidf_score"].fillna(0.0).to_numpy(dtype=float),
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    dev_corpus = (
        df_dev["title_1"].fillna("").astype(str).tolist()
        + df_dev["title_2"].fillna("").astype(str).tolist()
    )
    vectorizer.fit(dev_corpus)

    def pair_cosine(df: pd.DataFrame) -> np.ndarray:
        m1 = vectorizer.transform(df["title_1"].fillna("").astype(str))
        m2 = vectorizer.transform(df["title_2"].fillna("").astype(str))
        scores = np.asarray(m1.multiply(m2).sum(axis=1)).ravel()
        return np.clip(scores, 0.0, 1.0)

    return pair_cosine(df_dev), pair_cosine(df_test)


def compute_e5_scores(df_pairs: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Обчислює базову косинусну подібність через multilingual-e5-small (з префіксами query/passage)."""
    q_texts = [f"query: {str(t1).strip()}" for t1 in df_pairs["title_1"].fillna("")]
    p_texts = [f"passage: {str(t2).strip()}" for t2 in df_pairs["title_2"].fillna("")]

    emb_q = model.encode(
        q_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb_p = model.encode(
        p_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = util.cos_sim(emb_q, emb_p).diagonal().cpu().numpy()
    return np.clip(scores, 0.0, 1.0)


def compute_integral_scores(
    df_pairs: pd.DataFrame,
    base_scores: np.ndarray,
    w_sem: float,
    w_dom: float,
    w_geod: float,
    use_ner: bool,
    use_geo: bool,
) -> np.ndarray:
    """
    Обчислює інтегральну оцінку для кожної пари за конфігурацією:
    - use_ner=False, use_geo=False -> лише E5
    - use_ner=True,  use_geo=False -> E5 + NER (без геоперевірки)
    - use_ner=True,  use_geo=True  -> E5 + NER + гео (Гаверсинус)
    """
    if not use_ner and not use_geo:
        return base_scores.copy()

    supports_use_geo_arg = "use_geo" in inspect.signature(DomainContextEngine.evaluate_match).parameters
    orig_compute_geod = getattr(eng_mod, "compute_geodesic_penalty", None)

    if not use_geo and not supports_use_geo_arg and orig_compute_geod is not None:
        eng_mod.compute_geodesic_penalty = lambda *args, **kwargs: (1.0, None)

    integral_scores = []
    try:
        for idx, row in df_pairs.iterrows():
            s_base = float(base_scores[idx])
            q_text = str(row.get("title_1", ""))
            cand_row = {
                "title": str(row.get("title_2", "")),
                "text": str(row.get("text_2", "")),
                "authority": str(row.get("authority_2", "")),
            }

            kwargs = {
                "query_text": q_text,
                "candidate_row": cand_row,
                "base_semantic_score": s_base,
                "w_sem": w_sem,
                "w_dom": w_dom,
                "w_geod": w_geod,
            }
            if supports_use_geo_arg:
                kwargs["use_geo"] = use_geo

            res = DomainContextEngine.evaluate_match(**kwargs)
            integral_scores.append(float(res["final_score"]))
    finally:
        if not use_geo and not supports_use_geo_arg and orig_compute_geod is not None:
            eng_mod.compute_geodesic_penalty = orig_compute_geod

    return np.array(integral_scores, dtype=float)


def find_best_threshold_on_dev(
    y_dev: np.ndarray,
    scores_dev: np.ndarray,
    th_min: float = 0.20,
    th_max: float = 0.95,
    steps: int = 151,
    min_precision: float = MIN_DEV_PRECISION,
) -> dict:
    """
    Підбирає оптимальний поріг класифікації ВИКЛЮЧНО на DEV.
    Спочатку шукає максимум F1 серед порогів із precision >= min_precision;
    якщо жоден поріг не дотягує до min_precision — обирає максимум F1 без обмеження.
    """
    thresholds = np.linspace(th_min, th_max, steps)
    auc_dev = float(roc_auc_score(y_dev, scores_dev)) if len(np.unique(y_dev)) > 1 else 0.0

    candidates = []
    for th in thresholds:
        preds = (scores_dev >= th).astype(int)
        f1 = float(f1_score(y_dev, preds, zero_division=0))
        prec = float(precision_score(y_dev, preds, zero_division=0))
        rec = float(recall_score(y_dev, preds, zero_division=0))
        candidates.append(
            {
                "threshold": round(float(th), 3),
                "dev_f1": round(f1, 4),
                "dev_precision": round(prec, 4),
                "dev_recall": round(rec, 4),
                "dev_roc_auc": round(auc_dev, 4),
                "_raw_f1": f1,
                "_raw_prec": prec,
            }
        )

    qualified = [c for c in candidates if c["_raw_prec"] >= min_precision]
    pool = qualified if qualified else candidates

    best = max(pool, key=lambda c: (round(c["_raw_f1"], 6), round(c["_raw_prec"], 6)))
    return {
        "threshold": best["threshold"],
        "dev_f1": best["dev_f1"],
        "dev_precision": best["dev_precision"],
        "dev_recall": best["dev_recall"],
        "dev_roc_auc": best["dev_roc_auc"],
    }


def tune_weights_and_thresholds_on_dev(
    df_dev: pd.DataFrame,
    scores_tfidf_dev: np.ndarray,
    scores_e5_dev: np.ndarray,
) -> dict:
    """
    Підбирає оптимальні ваги (w_sem, w_dom, w_geod) та окремі пороги
    для ВСІХ 4 конфігурацій виключно на DEV.
    """
    y_dev = df_dev["is_similar"].astype(int).to_numpy()
    df_dev_reset = df_dev.reset_index(drop=True)

    tfidf_best = find_best_threshold_on_dev(y_dev, scores_tfidf_dev, th_min=0.10, th_max=0.90, steps=161)
    e5_best = find_best_threshold_on_dev(y_dev, scores_e5_dev, th_min=0.50, th_max=0.98, steps=97)

    weight_candidates = [
        (0.65, 0.25, 0.10),
        (0.60, 0.25, 0.15),
        (0.70, 0.20, 0.10),
        (0.55, 0.30, 0.15),
        (0.50, 0.30, 0.20),
        (0.75, 0.15, 0.10),
    ]

    best_full = None
    best_full_key = (-1.0, -1.0, -1.0)
    weight_search_log = []

    for w_sem, w_dom, w_geod in weight_candidates:
        dev_full_scores = compute_integral_scores(
            df_dev_reset,
            scores_e5_dev,
            w_sem=w_sem,
            w_dom=w_dom,
            w_geod=w_geod,
            use_ner=True,
            use_geo=True,
        )
        th_stats = find_best_threshold_on_dev(y_dev, dev_full_scores, th_min=0.45, th_max=0.95, steps=101)
        weight_search_log.append(
            {
                "w_sem": w_sem,
                "w_dom": w_dom,
                "w_geod": w_geod,
                **th_stats,
            }
        )

        key = (th_stats["dev_f1"], th_stats["dev_precision"], th_stats["dev_roc_auc"])
        if key > best_full_key:
            best_full_key = key
            best_full = {
                "w_sem": w_sem,
                "w_dom": w_dom,
                "w_geod": w_geod,
                **th_stats,
            }

    dev_ner_scores = compute_integral_scores(
        df_dev_reset,
        scores_e5_dev,
        w_sem=best_full["w_sem"],
        w_dom=best_full["w_dom"],
        w_geod=best_full["w_geod"],
        use_ner=True,
        use_geo=False,
    )
    ner_best = find_best_threshold_on_dev(y_dev, dev_ner_scores, th_min=0.45, th_max=0.95, steps=101)

    return {
        "tfidf": tfidf_best,
        "e5": e5_best,
        "e5_ner": ner_best,
        "full": best_full,
        "weight_search_log": pd.DataFrame(weight_search_log),
    }


def evaluate_on_test(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    threshold: float,
    name: str,
    elapsed_ms: float,
) -> tuple[dict, np.ndarray]:
    preds = (scores_test >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc_test = float(roc_auc_score(y_test, scores_test)) if len(np.unique(y_test)) > 1 else 0.0

    return {
        "Configuration": name,
        "Threshold (from DEV)": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "Precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "Specificity": round(float(specificity), 4),
        "F1-Score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "ROC-AUC": round(auc_test, 4),
        "Time_ms": round(float(elapsed_ms), 1),
    }, preds


def main():
    dev_path = resolve_data_path("ground_truth_dev.xlsx")
    test_path = resolve_data_path("ground_truth_test.xlsx")

    df_dev = pd.read_excel(dev_path).reset_index(drop=True)
    df_test = pd.read_excel(test_path).reset_index(drop=True)

    print(f"[*] Завантажено DEV: {len(df_dev)} пар | TEST: {len(df_test)} пар")

    print("\n=== ДІАГНОСТИКА КОЛОНОК DEV / TEST ===")
    for split_name, df_split in [("DEV", df_dev), ("TEST", df_test)]:
        has_auth = "authority_2" in df_split.columns
        has_text = "text_2" in df_split.columns
        non_empty_auth = (
            int(df_split["authority_2"].fillna("").astype(str).str.strip().ne("").sum())
            if has_auth
            else 0
        )
        non_empty_text = (
            int(df_split["text_2"].fillna("").astype(str).str.strip().ne("").sum())
            if has_text
            else 0
        )
        print(
            f"{split_name}: колонки={list(df_split.columns)} | "
            f"authority_2={has_auth} (непорожніх: {non_empty_auth}/{len(df_split)}) | "
            f"text_2={has_text} (непорожніх: {non_empty_text}/{len(df_split)})"
        )

    model = SentenceTransformer(MODEL_NAME)

    # 1. Підбір порогів і ваг ТІЛЬКИ на DEV
    scores_tfidf_dev, scores_tfidf_test = ensure_tfidf_scores(df_dev, df_test)
    scores_e5_dev = compute_e5_scores(df_dev, model)

    tuned = tune_weights_and_thresholds_on_dev(df_dev, scores_tfidf_dev, scores_e5_dev)

    print("\n=== ПЕРЕБІР ВАГ НА DEV (E5 + NER + ГЕО) ===")
    print(tuned["weight_search_log"].to_string(index=False))

    print("\n=== ЗАФІКСОВАНІ ПАРАМЕТРИ, ПІДІБРАНІ ЛИШЕ НА DEV ===")
    print(f"TF-IDF поріг:        {tuned['tfidf']}")
    print(f"Лише E5 поріг:       {tuned['e5']}")
    print(f"E5 + NER поріг:      {tuned['e5_ner']}")
    print(f"E5 + NER + гео:      {tuned['full']}")

    w_sem = tuned["full"]["w_sem"]
    w_dom = tuned["full"]["w_dom"]
    w_geod = tuned["full"]["w_geod"]

    # 2. Одноразова оцінка всіх 4 конфігурацій на відкладеній вибірці TEST
    y_test = df_test["is_similar"].astype(int).to_numpy()
    test_rows = []

    t0 = time.perf_counter()
    dt_tfidf = (time.perf_counter() - t0) * 1000
    row_tfidf, _ = evaluate_on_test(
        y_test,
        scores_tfidf_test,
        tuned["tfidf"]["threshold"],
        "1. TF-IDF Baseline",
        dt_tfidf,
    )
    test_rows.append(row_tfidf)

    t0 = time.perf_counter()
    scores_e5_test = compute_e5_scores(df_test, model)
    dt_e5 = (time.perf_counter() - t0) * 1000
    row_e5, _ = evaluate_on_test(
        y_test,
        scores_e5_test,
        tuned["e5"]["threshold"],
        "2. Лише E5 (multilingual-e5-small)",
        dt_e5,
    )
    test_rows.append(row_e5)

    t0 = time.perf_counter()
    scores_ner_test = compute_integral_scores(
        df_test,
        scores_e5_test,
        w_sem=w_sem,
        w_dom=w_dom,
        w_geod=w_geod,
        use_ner=True,
        use_geo=False,
    )
    dt_ner = (time.perf_counter() - t0) * 1000
    row_ner, _ = evaluate_on_test(
        y_test,
        scores_ner_test,
        tuned["e5_ner"]["threshold"],
        "3. E5 + NER",
        dt_ner,
    )
    test_rows.append(row_ner)

    t0 = time.perf_counter()
    scores_full_test = compute_integral_scores(
        df_test,
        scores_e5_test,
        w_sem=w_sem,
        w_dom=w_dom,
        w_geod=w_geod,
        use_ner=True,
        use_geo=True,
    )
    dt_full = (time.perf_counter() - t0) * 1000
    row_full, final_preds = evaluate_on_test(
        y_test,
        scores_full_test,
        tuned["full"]["threshold"],
        "4. E5 + NER + гео (Гаверсинус)",
        dt_full,
    )
    test_rows.append(row_full)

    diff_e5_ner = int(np.sum(~np.isclose(scores_e5_test, scores_ner_test, atol=1e-6)))
    diff_ner_full = int(np.sum(~np.isclose(scores_ner_test, scores_full_test, atol=1e-6)))
    print("\n=== ДІАГНОСТИКА ЗМІНИ БАЛІВ НА TEST ===")
    print(f"Пар зі зміною балу (Лише E5 -> E5 + NER):        {diff_e5_ner}/{len(df_test)}")
    print(f"Пар зі зміною балу (E5 + NER -> E5 + NER + гео): {diff_ner_full}/{len(df_test)}")

    if diff_ner_full == 0:
        print("[!] Різниця між E5 + NER та E5 + NER + гео на TEST становить 0 пар. Приклади рядків:")
        cols_preview = [c for c in ["title_1", "title_2", "authority_2", "is_similar"] if c in df_test.columns]
        print(df_test[cols_preview].head(5).to_string(index=False))

    df_res = pd.DataFrame(test_rows)
    print("\n=== РЕЗУЛЬТАТИ НА ВІДКЛАДЕНІЙ ВИБІРЦІ TEST (БЕЗ ВИТОКУ ПОРОГІВ) ===")
    print(df_res.to_string(index=False))

    out_excel = os.path.join(BASE_DIR, "test_ablation_results.xlsx")
    with pd.ExcelWriter(out_excel) as writer:
        df_res.to_excel(writer, sheet_name="TEST_Results", index=False)
        tuned["weight_search_log"].to_excel(writer, sheet_name="DEV_Weight_Search", index=False)

    cm = confusion_matrix(y_test, final_preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Unique (0)", "Duplicate (1)"],
        yticklabels=["Unique (0)", "Duplicate (1)"],
    )
    plt.title(
        f"Матриця плутанини на TEST (E5 + NER + гео)\n"
        f"w=({w_sem}, {w_dom}, {w_geod}), th={tuned['full']['threshold']}"
    )
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Справжній клас (Ground Truth)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "test_confusion_matrix.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()