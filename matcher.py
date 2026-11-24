import inspect
import os
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from engine import DomainContextEngine

# Визначаємо шлях до поточної папки
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Базові пороги та ваги інтегральної оцінки (зафіксовані на dev або передаються явно)
DEFAULT_THRESHOLD_DUPLICATE = 0.88
DEFAULT_THRESHOLD_REVIEW = 0.76
DEFAULT_W_SEM = 0.65
DEFAULT_W_DOM = 0.25
DEFAULT_W_GEOD = 0.10


def find_file(filename: str) -> str:
    """Шукає файл у корені проєкту або в підпапці data/."""
    possible_paths = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "data", filename),
        filename,
        os.path.join("data", filename),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None


def load_system_core():
    """Завантажує датасет, ML-модель та серіалізовані ембединги з автопошуком шляхів."""
    dataset_path = find_file("petitions_preprocessed.csv")
    if not dataset_path:
        raise FileNotFoundError(
            "Не знайдено файл датасету petitions_preprocessed.csv. "
            "Переконайся, що файл лежить у корені проєкту або у папці data/."
        )

    df = pd.read_csv(dataset_path)

    embeddings_path = find_file("embeddings.pt")
    if not embeddings_path:
        raise FileNotFoundError(
            "Не знайдено файл embeddings.pt. "
            "Переконайся, що файл лежить у корені проєкту або у папці data/."
        )

    embeddings = torch.load(embeddings_path, map_location="cpu")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    return df, model, embeddings


def search_similar_petitions(
    query_text: str,
    df_corpus: pd.DataFrame,
    model: SentenceTransformer,
    corpus_embeddings,
    selected_authority: str,
    strict_mode: bool,
    top_k: int,
    selected_statuses: list = None,
    threshold_duplicate: float = DEFAULT_THRESHOLD_DUPLICATE,
    threshold_review: float = DEFAULT_THRESHOLD_REVIEW,
    w_sem: float = DEFAULT_W_SEM,
    w_dom: float = DEFAULT_W_DOM,
    w_geod: float = DEFAULT_W_GEOD,
    use_geo: bool = True,
) -> pd.DataFrame:
    """
    Двохетапний пошук (Two-Stage Retrieval):
    1. Обчислення косинусної подібності до нормалізованого корпусу.
    2. Фільтрація за юрисдикцією та статусом петиції.
    3. Доменне та геопросторове переранжування пулу кандидатів (DomainContextEngine).
    """
    # 1. Формуємо нормалізований ембединг запиту
    encoded_query = model.encode(
        [f"query: {query_text.strip()}"],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # 2. Обчислення подібності (для нормалізованих векторів це dot_score)
    raw_scores = util.dot_score(encoded_query, corpus_embeddings)[0].cpu().numpy()

    # 3. Прив'язка точних (неокруглених) балів до датафрейму зі збереженням індексів
    df_working = df_corpus.copy()
    df_working["base_score"] = pd.Series(raw_scores.astype(float), index=df_corpus.index)

    # 4. Фільтрація за юрисдикцією ОМС
    if strict_mode and selected_authority != "Всі юрисдикції (Національний рівень)":
        df_working = df_working[df_working["authority"] == selected_authority]

    # 5. Фільтрація за статусом петиції
    if selected_statuses and "status" in df_working.columns:
        df_working = df_working[df_working["status"].isin(selected_statuses)]

    if df_working.empty:
        return pd.DataFrame()

    # 6. Відбираємо пул найкращих кандидатів (top_k * 3, мінімум 20)
    candidate_pool_size = max(top_k * 3, 20)
    top_candidates = df_working.sort_values(by="base_score", ascending=False).head(candidate_pool_size).copy()

    # 7. Застосовуємо зважену інтегральну оцінку DomainContextEngine до точного base_score
    supports_use_geo = "use_geo" in inspect.signature(DomainContextEngine.evaluate_match).parameters

    evaluations = []
    for _, row in top_candidates.iterrows():
        kwargs = {
            "query_text": query_text,
            "candidate_row": row,
            "base_semantic_score": float(row["base_score"]),
            "threshold_duplicate": threshold_duplicate,
            "threshold_review": threshold_review,
            "w_sem": w_sem,
            "w_dom": w_dom,
            "w_geod": w_geod,
        }
        if supports_use_geo:
            kwargs["use_geo"] = use_geo

        evaluations.append(DomainContextEngine.evaluate_match(**kwargs))

    top_candidates["base_score"] = top_candidates["base_score"].round(4)
    top_candidates["final_score"] = [e["final_score"] for e in evaluations]
    top_candidates["verdict"] = [e["verdict"] for e in evaluations]
    top_candidates["status_text"] = [e["status_text"] for e in evaluations]
    top_candidates["badge_class"] = [e["badge_class"] for e in evaluations]
    top_candidates["reasons"] = [e["reasons"] for e in evaluations]

    # 8. Сортуємо фінальний пул
    return top_candidates.sort_values(by=["final_score", "base_score"], ascending=[False, False]).head(top_k)