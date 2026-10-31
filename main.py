import time
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Response, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

#  наявних модулів системи
from matcher import load_system_core, search_similar_petitions
from validation import is_valid_input_text
from reports import export_to_csv, export_to_pdf
from engine import DomainContextEngine

# 1. Налаштування логування для моніторингу роботи сервера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PetitionAPI")

# 2. Ініціалізація FastAPI додатку 
app = FastAPI(
    title="Municipal Petition Semantic & Deduplication API",
    description=(
        "Високопродуктивний мікросервіс на базі FastAPI для автоматизованого семантичного "
        "зіставлення, дедуплікації та геопросторової верифікації електронних петицій. "
        "Розроблено в рамках магістерської кваліфікаційної роботи."
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

    if results_df.empty:
        return {
            "success": True,
            "query_title": payload.title,
            "top_verdict": "UNIQUE",
            "status_text": "Оригінальна ініціатива",
            "top_score": 0.0,
            "matches_count": 0,
            "latency_ms": latency_ms,
            "candidates_evaluated": candidates_count,
            "similar_petitions": []
        }

    top_record = results_df.iloc[0]
    top_score = float(top_record.get('final_score', 0.0))
    verdict = str(top_record.get('verdict', 'UNIQUE'))
    status_text = str(top_record.get('status_text', 'Оригінальна ініціатива'))

    # Форматування списку знайдених аналогів
    matches_list = []
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

    logger.info(f"Processed query in {latency_ms}ms | Top Score: {top_score} | Verdict: {verdict}")

    return {
        "success": True,
        "query_title": payload.title,
        "top_verdict": verdict,
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