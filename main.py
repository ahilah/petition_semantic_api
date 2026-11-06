import time
import logging
import pandas as pd
import torch

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Response, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

#  наявних модулів системи
from matcher import load_system_core, search_similar_petitions
from validation import is_valid_input_text
from reports import export_to_csv, export_to_pdf
from engine import DomainContextEngine
from database import init_db, log_petition_check, get_system_analytics, log_expert_feedback

# 1. Налаштування логування для виведення в консоль ТА у файл app.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),                  # Вивід у термінал 
        logging.FileHandler("app.log", encoding="utf-8")  # Запис у файл
    ]
)
logger = logging.getLogger("PetitionAPI")

# 2. Ініціалізація FastAPI додатку 
app = FastAPI(
    title="Municipal Petition Semantic & Deduplication API",
    description=(
        "Високопродуктивний мікросервіс на базі FastAPI для автоматизованого семантичного зіставлення, дедуплікації та геопросторової верифікації електронних петицій. "
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Додаємо CORS для безшовної інтеграції з будь-якими сайтами ОМС
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Глобальні змінні для завантаження ML-ядра в пам'ять при старті
df_corpus: Optional[pd.DataFrame] = None
model: Optional[SentenceTransformer] = None
corpus_embeddings: Optional[torch.Tensor] = None


@app.on_event("startup")
def startup_event():
    """
    Подія при запуск сервера: одноразове завантаження датасету, 
    трансформерної моделі Multilingual-E5 та серіалізованих ембедингів.
    """
    init_db()
    
    global df_corpus, model, corpus_embeddings
    logger.info("⏳ Ініціалізація ML-ядра та завантаження корпусу петицій...")
    try:
        df_corpus, model, corpus_embeddings = load_system_core()
        logger.info(f"✅ Успішно завантажено записів у корпусі: {len(df_corpus):,}".replace(",", " "))
        logger.info(f"✅ Модель SentenceTransformer активовано. Формат ембедингів: {corpus_embeddings.shape}")
    except Exception as e:
        logger.error(f"❌ Критична помилка під час завантаження системного ядра: {e}")
        df_corpus, model, corpus_embeddings = None, None, None


# 4. Pydantic-схеми для валідації вхідних та вихідних даних
class PetitionCheckRequest(BaseModel):
    title: str = Field(..., description="Заголовок нової петиції", min_length=3, example="Ремонт дорожнього покриття на вулиці Шевченка")
    body: str = Field("", description="Повний текст звернення громадян", example="Просимо терміново провести ямковий ремонт...")
    authority: str = Field("Всі юрисдикції (Національний рівень)", description="Цільовий орган місцевого самоврядування (ОМС)")
    strict_mode: bool = Field(True, description="Сувора ізоляція юрисдикції громади")
    top_k: int = Field(5, description="Кількість найбільш схожих кандидатів для видачі", ge=1, le=10)
    threshold_duplicate: float = Field(0.88, description="Поріг класифікації як прямий дублікат", ge=0.5, le=1.0)
    threshold_review: float = Field(0.76, description="Поріг для направлення на ручну модерацію", ge=0.4, le=0.9)
    confidence_floor: float = Field(0.60, description="Поріг відсікання фонового шуму", ge=0.3, le=0.8)


class MatchItem(BaseModel):
    title: str
    authority: str
    author: str
    base_score: float
    final_score: float
    verdict: str
    status_text: str
    reasons: List[str]


class PetitionCheckResponse(BaseModel):
    success: bool
    query_title: str
    top_verdict: str
    status_text: str
    top_score: float
    matches_count: int
    latency_ms: float
    candidates_evaluated: int
    similar_petitions: List[MatchItem]


class BatchPetitionCheckRequest(BaseModel):
    petitions: List[PetitionCheckRequest] = Field(
        ..., 
        description="Масив петицій для пакетної перевірки (до 50 штук за раз)", 
        min_items=1, 
        max_items=50
    )


class BatchItemResult(BaseModel):
    query_title: str
    top_verdict: str
    status_text: str
    top_score: float
    matches_count: int
    latency_ms: float
    similar_petitions: List[MatchItem]


class BatchPetitionCheckResponse(BaseModel):
    success: bool
    total_processed: int
    total_latency_ms: float
    results: List[BatchItemResult]



class ClusterInfo(BaseModel):
    cluster_id: int = Field(..., description="Ідентифікатор кластера")
    petition_count: int = Field(..., description="Кількість петицій у кластері")
    percentage: float = Field(..., description="Частка від загального корпусу (%)")
    sample_titles: List[str] = Field(..., description="Приклади заголовків петицій у цьому кластері")


class ClustersResponse(BaseModel):
    success: bool
    total_clusters: int
    total_petitions: int
    clusters: List[ClusterInfo]


class FeedbackRequest(BaseModel):
    query_title: str = Field(..., description="Заголовок нової (перевірюваної) петиції")
    matched_title: str = Field(..., description="Заголовок знайденої схожої петиції-кандидата")
    is_duplicate: bool = Field(..., description="Чи є ця пара дублікатами за висновком експерта (true/false)")
    comment: Optional[str] = Field("", description="Коментар або примітка модератора ОМС")


class FeedbackResponse(BaseModel):
    success: bool
    message: str


# 5. Ендпоїнти API
@app.get("/", tags=["Системні методи"])
def read_root():
    """Перевірка стану працездатності мікросервісу (Health Check)."""
    return {
        "status": "online",
        "service": "Municipal Petition Semantic & Deduplication API",
        "corpus_loaded": df_corpus is not None,
        "total_records": len(df_corpus) if df_corpus is not None else 0,
        "documentation": "/docs"
    }


@app.get("/api/v1/authorities", tags=["Довідники"])
def get_authorities():
    """Повертає список усіх доступних органів місцевого самоврядування (ОМС) у базі."""
    if df_corpus is None:
        raise HTTPException(status_code=500, detail="Корпус даних не завантажено на сервері.")
    
    authorities = sorted([str(a) for a in df_corpus['authority'].dropna().unique()])
    return {
        "success": True,
        "count": len(authorities),
        "authorities": authorities
    }


@app.post("/api/v1/check-petition", response_model=PetitionCheckResponse, tags=["Аналітичне ядро"])
def check_petition_endpoint(payload: PetitionCheckRequest):
    """
    Головний аналітичний ендпоїнт:
    1. Перевіряє вхідний текст на спам та валідність.
    2. Виконує двохетапний пошук (Two-Stage Retrieval) за допомогою Multilingual-E5.
    3. Застосовує доменно-контекстний рушій (DomainContextEngine) та геопросторовий штраф Гаверсинуса.
    4. Повертає структурований JSON з вердиктом та аргументами.
    """
    if df_corpus is None or model is None or corpus_embeddings is None:
        raise HTTPException(status_code=500, detail="ML-модель або корпус даних не ініціалізовані.")

    start_time = time.perf_counter()

    # 1. Валідація тексту
    is_valid, validation_msg = is_valid_input_text(payload.title)
    if not is_valid:
        logger.warning(f"⚠️ Відхилено невалідний запит: {validation_msg}")
        return {
            "success": False,
            "query_title": payload.title,
            "top_verdict": "INVALID",
            "status_text": validation_msg,
            "top_score": 0.0,
            "matches_count": 0,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 1),
            "candidates_evaluated": 0,
            "similar_petitions": []
        }

    # 2. Формування повного запиту
    full_query = f"{payload.title.strip()}. {payload.body.strip()}".strip()

    # 3. Виконання пошуку та переранжування
    try:
        results_df = search_similar_petitions(
            query_text=full_query,
            df_corpus=df_corpus,
            model=model,
            corpus_embeddings=corpus_embeddings,
            selected_authority=payload.authority,
            strict_mode=payload.strict_mode,
            top_k=payload.top_k,
            threshold_duplicate=payload.threshold_duplicate,
            threshold_review=payload.threshold_review
        )
    except Exception as e:
        logger.exception("❌ Помилка під час векторного інференсу або фільтрації")
        raise HTTPException(status_code=500, detail=f"Помилка обробки запиту: {str(e)}")

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    # Підрахунок кількості реально перевірених кандидатів у межах юрисдикції
    df_scope = df_corpus.copy()
    if payload.strict_mode and payload.authority != "Всі юрисдикції (Національний рівень)":
        df_scope = df_scope[df_scope['authority'] == payload.authority]
    candidates_count = len(df_scope)

    # Безпечна ініціалізація змінних за замовчуванням (усуває попередження лінтера)
    top_verdict = "UNIQUE"
    status_text = "Оригінальна ініціатива"
    top_score = 0.0
    matches_list = []

    if not results_df.empty:
        top_record = results_df.iloc[0]
        top_score = float(top_record.get('final_score', 0.0))
        top_verdict = str(top_record.get('verdict', 'UNIQUE'))
        status_text = str(top_record.get('status_text', 'Оригінальна ініціатива'))

        # Форматування списку знайдених аналогів
        for _, row in results_df.iterrows():
            matches_list.append(MatchItem(
                title=str(row.get('title', 'Без назви')),
                authority=str(row.get('authority', 'Не вказано')),
                author=str(row.get('author', 'Невідомий')),
                base_score=float(row.get('base_score', 0.0)),
                final_score=float(row.get('final_score', 0.0)),
                verdict=str(row.get('verdict', 'UNIQUE')),
                status_text=str(row.get('status_text', '')),
                reasons=row.get('reasons', []) if isinstance(row.get('reasons'), list) else []
            ))

    # Зберігаємо результати у базу даних SQLite
    try:
        log_petition_check(
            query_title=payload.title,
            authority=payload.authority,
            strict_mode=payload.strict_mode,
            top_k=payload.top_k,
            threshold_duplicate=payload.threshold_duplicate,
            threshold_review=payload.threshold_review,
            top_verdict=top_verdict,
            status_text=status_text,
            top_score=top_score,
            candidates_evaluated=candidates_count,
            latency_ms=latency_ms
        )
    except Exception as db_err:
        logger.error(f"⚠️ Не вдалося записати лог у базу даних: {db_err}")

    logger.info(f"Processed query in {latency_ms}ms | Top Score: {top_score} | Verdict: {top_verdict}")

    return {
        "success": True,
        "query_title": payload.title,
        "top_verdict": top_verdict,
        "status_text": status_text,
        "top_score": top_score,
        "matches_count": len(matches_list),
        "latency_ms": latency_ms,
        "candidates_evaluated": candidates_count,
        "similar_petitions": matches_list
    }


@app.post("/api/v1/export/pdf", tags=["Експорт звітів"])
def export_pdf_report(payload: PetitionCheckRequest):
    """
    Генерує та повертає офіційний експертний PDF-протокол перевірки звернення 
    з деталізацією причин коригування та списком знайдених аналогів.
    """
    if df_corpus is None or model is None:
        raise HTTPException(status_code=500, detail="Системні ресурси не ініціалізовано.")

    full_query = f"{payload.title.strip()}. {payload.body.strip()}".strip()
    
    results_df = search_similar_petitions(
        query_text=full_query,
        df_corpus=df_corpus,
        model=model,
        corpus_embeddings=corpus_embeddings,
        selected_authority=payload.authority,
        strict_mode=payload.strict_mode,
        top_k=payload.top_k,
        threshold_duplicate=payload.threshold_duplicate,
        threshold_review=payload.threshold_review
    )

    top_verdict = "Оригінальна ініціатива"
    if not results_df.empty:
        top_verdict = f"{results_df.iloc[0].get('status_text', '')} (Бал: {results_df.iloc[0].get('final_score', 0)})"

    try:
        pdf_bytes = export_to_pdf(
            query_title=payload.title,
            query_body=payload.body,
            authority=payload.authority,
            results_df=results_df,
            top_verdict=top_verdict
        )
    except Exception as e:
        logger.exception("❌ Помилка генерації PDF-протоколу")
        raise HTTPException(status_code=500, detail=f"Не вдалося згенерувати PDF: {str(e)}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=petition_audit_protocol.pdf"}
    )


@app.post("/api/v1/export/csv", tags=["Експорт звітів"])
def export_csv_report(payload: PetitionCheckRequest):
    """Експортує результати знайдених збігів у формат CSV для аналітичної обробки."""
    if df_corpus is None or model is None:
        raise HTTPException(status_code=500, detail="Системні ресурси не ініціалізовано.")

    full_query = f"{payload.title.strip()}. {payload.body.strip()}".strip()
    
    results_df = search_similar_petitions(
        query_text=full_query,
        df_corpus=df_corpus,
        model=model,
        corpus_embeddings=corpus_embeddings,
        selected_authority=payload.authority,
        strict_mode=payload.strict_mode,
        top_k=payload.top_k
    )

    try:
        csv_bytes = export_to_csv(results_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не вдалося згенерувати CSV: {str(e)}")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=petition_audit_results.csv"}
    )


@app.post("/api/v1/batch-check", response_model=BatchPetitionCheckResponse, tags=["Аналітичне ядро"])
def batch_check_petitions(payload: BatchPetitionCheckRequest):
    """
    Пакетний аналітичний ендпоїнт (Batch Processing):
    1. Приймає масив до 50 нових петицій за один запит.
    2. Послідовно виконує двохетапний семантичний пошук та переранжування для кожної петиції.
    3. Автоматично логує кожну перевірку в реляційну базу даних SQLite.
    4. Повертає сукупний аналітичний звіт та час виконання для всього пакета.
    """
    if df_corpus is None or model is None or corpus_embeddings is None:
        raise HTTPException(status_code=500, detail="ML-модель або корпус даних не ініціалізовані.")

    batch_start_time = time.perf_counter()
    results_list = []
    df_scope_global = df_corpus.copy()

    for item in payload.petitions:
        item_start_time = time.perf_counter()

        # 1. Валідація тексту
        is_valid, validation_msg = is_valid_input_text(item.title)
        if not is_valid:
            results_list.append(BatchItemResult(
                query_title=item.title,
                top_verdict="INVALID",
                status_text=validation_msg,
                top_score=0.0,
                matches_count=0,
                latency_ms=round((time.perf_counter() - item_start_time) * 1000, 1),
                similar_petitions=[]
            ))
            continue

        # 2. Формування повного запиту
        full_query = f"{item.title.strip()}. {item.body.strip()}".strip()

        # 3. Пошук та переранжування
        try:
            results_df = search_similar_petitions(
                query_text=full_query,
                df_corpus=df_corpus,
                model=model,
                corpus_embeddings=corpus_embeddings,
                selected_authority=item.authority,
                strict_mode=item.strict_mode,
                top_k=item.top_k,
                threshold_duplicate=item.threshold_duplicate,
                threshold_review=item.threshold_review
            )
        except Exception as e:
            logger.error(f"❌ Помилка під час обробки петиції у пакеті '{item.title}': {e}")
            continue

        latency_ms = round((time.perf_counter() - item_start_time) * 1000, 1)

        # Підрахунок кандидатів у межах юрисдикції
        df_scope = df_scope_global.copy()
        if item.strict_mode and item.authority != "Всі юрисдикції (Національний рівень)":
            df_scope = df_scope[df_scope['authority'] == item.authority]
        candidates_count = len(df_scope)

        # Значення за замовчуванням
        top_verdict = "UNIQUE"
        status_text = "Оригінальна ініціатива"
        top_score = 0.0
        matches_list = []

        if not results_df.empty:
            top_record = results_df.iloc[0]
            top_score = float(top_record.get('final_score', 0.0))
            top_verdict = str(top_record.get('verdict', 'UNIQUE'))
            status_text = str(top_record.get('status_text', 'Оригінальна ініціатива'))

            for _, row in results_df.iterrows():
                matches_list.append(MatchItem(
                    title=str(row.get('title', 'Без назви')),
                    authority=str(row.get('authority', 'Не вказано')),
                    author=str(row.get('author', 'Невідомий')),
                    base_score=float(row.get('base_score', 0.0)),
                    final_score=float(row.get('final_score', 0.0)),
                    verdict=str(row.get('verdict', 'UNIQUE')),
                    status_text=str(row.get('status_text', '')),
                    reasons=row.get('reasons', []) if isinstance(row.get('reasons'), list) else []
                ))

        # 📌 Логування кожного елемента пакету в SQLite для статистики
        try:
            log_petition_check(
                query_title=item.title,
                authority=item.authority,
                strict_mode=item.strict_mode,
                top_k=item.top_k,
                threshold_duplicate=item.threshold_duplicate,
                threshold_review=item.threshold_review,
                top_verdict=top_verdict,
                status_text=status_text,
                top_score=top_score,
                candidates_evaluated=candidates_count,
                latency_ms=latency_ms
            )
        except Exception as db_err:
            logger.error(f"⚠️ Не вдалося записати пакетний лог у базу: {db_err}")

        results_list.append(BatchItemResult(
            query_title=item.title,
            top_verdict=top_verdict,
            status_text=status_text,
            top_score=top_score,
            matches_count=len(matches_list),
            latency_ms=latency_ms,
            similar_petitions=matches_list
        ))

    total_latency_ms = round((time.perf_counter() - batch_start_time) * 1000, 1)
    logger.info(f"Processed batch of {len(results_list)} petitions in {total_latency_ms}ms")

    return BatchPetitionCheckResponse(
        success=True,
        total_processed=len(results_list),
        total_latency_ms=total_latency_ms,
        results=results_list
    )


@app.get("/api/v1/clusters", response_model=ClustersResponse, tags=["Аналітика та моніторинг"])
def get_semantic_clusters(
    n_clusters: int = Query(6, description="Кількість тематичних кластерів для групування", ge=3, le=12)
):
    """
    Аналіз трендів та семантичних кластерів (Unsupervised Clustering):
    1. Використовує матрицю ембедингів усього корпусу петицій.
    2. Застосовує алгоритм кластеризації K-Means для виділення латентних тем.
    3. Агрегує статистику, визначає вагу кожної теми та виводить типові приклади звернень.
    4. Дозволяє міській раді бачити актуальні проблеми мешканців у реальному часі.
    """
    if df_corpus is None or corpus_embeddings is None:
        raise HTTPException(status_code=500, detail="Корпус даних або ембединги не ініціалізовані.")

    try:
        # Конвертуємо тензор PyTorch у масив NumPy
        embeddings_np = corpus_embeddings.cpu().numpy()
        total_records = len(df_corpus)

        if total_records < n_clusters:
            raise HTTPException(status_code=400, detail="Кількість записів у корпусі менша за бажану кількість кластерів.")

        # Запускаємо алгоритм K-Means кластеризації
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)

        # Додаємо мітки кластерів до копії датафрейму
        df_clustered = df_corpus.copy()
        df_clustered['cluster'] = cluster_labels

        clusters_result = []
        for cluster_id in range(n_clusters):
            cluster_subset = df_clustered[df_clustered['cluster'] == cluster_id]
            count = len(cluster_subset)
            percentage = round((count / total_records) * 100, 2)

            # Беремо до 3 характерних заголовків петицій як приклади теми
            sample_titles = cluster_subset['title'].head(3).tolist()

            clusters_result.append(ClusterInfo(
                cluster_id=cluster_id,
                petition_count=count,
                percentage=percentage,
                sample_titles=sample_titles
            ))

        # Сортуємо кластери за кількістю петицій (від найбільшої проблеми до найменшої)
        clusters_result = sorted(clusters_result, key=lambda x: x.petition_count, reverse=True)

        logger.info(f"Successfully clustered {total_records} petitions into {n_clusters} semantic groups.")

        return ClustersResponse(
            success=True,
            total_clusters=n_clusters,
            total_petitions=total_records,
            clusters=clusters_result
        )

    except Exception as e:
        logger.exception("❌ Помилка під час семантичної кластеризації корпусу")
        raise HTTPException(status_code=500, detail=f"Помилка кластеризації: {str(e)}")


@app.post("/api/v1/feedback", response_model=FeedbackResponse, tags=["Active Learning та Модерація"])
def submit_expert_feedback(payload: FeedbackRequest):
    """
    Модуль зворотного зв'язку (Active Learning / Human-in-the-Loop):
    1. Дозволяє модератору ОМС надіслати підтвердження щодо правильності дедуплікації.
    2. Зберігає експертну пару в базу даних для розширення контрольного датасету (Ground Truth).
    3. Забезпечує основу для подальшого донавчання або калібрування порогів моделі.
    """
    try:
        log_expert_feedback(
            query_title=payload.query_title,
            matched_title=payload.matched_title,
            is_duplicate=payload.is_duplicate,
            comment=payload.comment
        )
        logger.info(f"✅ Отримано експертний фідбек: дублікат = {payload.is_duplicate} для пари '{payload.query_title}' / '{payload.matched_title}'")
        
        return FeedbackResponse(
            success=True,
            message="Експертну оцінку успішно збережено в корпус Ground Truth."
        )
    except Exception as e:
        logger.exception("❌ Помилка під час збереження фідбеку модератора")
        raise HTTPException(status_code=500, detail=f"Не вдалося зберегти фідбек: {str(e)}")
     

@app.get("/api/v1/export/geojson", tags=["Геопросторовий аналіз та Експорт"])
def export_geojson_endpoint(
    authority: Optional[str] = Query(None, description="Фільтрація петицій за конкретним органом місцевого самоврядування (ОМС)")
):
    """
    Геопросторовий експорт у форматі GeoJSON:
    1. Фільтрує корпус петицій за заданою юрисдикцією (опціонально).
    2. Збирає просторові координати (або генерує геопросторові маркери для об'єктів з урахуванням виділених урбанонімів).
    3. Формує стандартну структуру GeoJSON FeatureCollection.
    4. Забезпечує миттєву інтеграцію з ГІС-системами та інтерактивними картами (Leaflet / Folium).
    """
    if df_corpus is None:
        raise HTTPException(status_code=500, detail="Корпус даних не ініціалізовано на сервері.")

    try:
        df_subset = df_corpus.copy()
        
        # Фільтрація за ОМС, якщо вказано параметр
        if authority and authority != "Всі юрисдикції (Національний рівень)":
            df_subset = df_subset[df_subset['authority'] == authority]

        features = []
        for _, row in df_subset.iterrows():
            # Перевіряємо наявність координат у датасеті (або використовуємо дефолтні геомітки центру міста, наприклад Львова чи відповідного регіону)
            lat = float(row.get('latitude', 49.8397))  # За замовчуванням центр Львова, якщо координати відсутні
            lon = float(row.get('longitude', 24.0297))

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # У форматі GeoJSON спочатку йде довгота (longitude), потім широта (latitude)
                },
                "properties": {
                    "title": str(row.get('title', 'Без назви')),
                    "authority": str(row.get('authority', 'Не вказано')),
                    "author": str(row.get('author', 'Невідомий')),
                    "category": str(row.get('category', 'Інфраструктура'))
                }
            }
            features.append(feature)

        geojson_collection = {
            "type": "FeatureCollection",
            "features": features
        }

        logger.info(f"Successfully generated GeoJSON with {len(features)} spatial features.")
        return geojson_collection

    except Exception as e:
        logger.exception("❌ Помилка під час генерації GeoJSON-експорту")
        raise HTTPException(status_code=500, detail=f"Помилка експорту геоданих: {str(e)}")


@app.get("/api/v1/analytics/summary", tags=["Аналітика та моніторинг"])
def get_analytics_summary():
    """Повертає загальну телеметрію та статистику роботи семантичного рушія."""
    stats = get_system_analytics()
    return {
        "success": True,
        "analytics": stats
    }