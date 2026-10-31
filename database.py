import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime

# Визначаємо шлях до бази даних у папці data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "petition_analytics.db")

@contextmanager
def get_db_connection():
    """Контекстний менеджер для безпечного відкриття та закриття з'єднання з БД."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # дступ до колонок за іменами
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Ініціалізація структури бази даних з індексами для швидкого пошуку."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Основна таблиця логування перевірок петицій
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS petition_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_title TEXT NOT NULL,
                authority TEXT NOT NULL,
                strict_mode BOOLEAN NOT NULL,
                top_k INTEGER NOT NULL,
                threshold_duplicate REAL NOT NULL,
                threshold_review REAL NOT NULL,
                top_verdict TEXT NOT NULL,
                status_text TEXT NOT NULL,
                top_score REAL NOT NULL,
                candidates_evaluated INTEGER NOT NULL,
                latency_ms REAL NOT NULL
            )
        ''')
        
        # Створюємо індекси для аналітичних запитів (наприклад, за вердиктом чи датою)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_verdict ON petition_checks(top_verdict);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_authority ON petition_checks(authority);')

        # Таблиця для активного навчання та збереження експертних оцінок (Feedback / Ground Truth)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS petition_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_title TEXT NOT NULL,
                matched_title TEXT NOT NULL,
                is_duplicate BOOLEAN NOT NULL,
                comment TEXT
            )
        ''')
        
        conn.commit()

def log_petition_check(
    query_title: str,
    authority: str,
    strict_mode: bool,
    top_k: int,
    threshold_duplicate: float,
    threshold_review: float,
    top_verdict: str,
    status_text: str,
    top_score: float,
    candidates_evaluated: int,
    latency_ms: float
):
    """Зберігає детальний телеметричний звіт про перевірку петиції."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO petition_checks (
                timestamp, query_title, authority, strict_mode, top_k,
                threshold_duplicate, threshold_review, top_verdict, 
                status_text, top_score, candidates_evaluated, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            query_title,
            authority,
            int(strict_mode),
            top_k,
            threshold_duplicate,
            threshold_review,
            top_verdict,
            status_text,
            float(top_score),
            int(candidates_evaluated),
            float(latency_ms)
        ))
        conn.commit()

def get_system_analytics() -> dict:
    """Формує зведену статистику використання системи для наукової роботи."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM petition_checks")
        total_checks = cursor.fetchone()["total"]
        
        if total_checks == 0:
            return {"total_checks": 0, "avg_latency_ms": 0, "verdicts_distribution": {}}
            
        cursor.execute("SELECT AVG(latency_ms) as avg_lat FROM petition_checks")
        avg_latency = round(cursor.fetchone()["avg_lat"], 2)
        
        cursor.execute("""
            SELECT top_verdict, COUNT(*) as count 
            FROM petition_checks 
            GROUP BY top_verdict
        """)
        verdicts = {row["top_verdict"]: row["count"] for row in cursor.fetchall()}
        
        return {
            "total_checks": total_checks,
            "avg_latency_ms": avg_latency,
            "verdicts_distribution": verdicts
        }


def log_expert_feedback(query_title: str, matched_title: str, is_duplicate: bool, comment: str = ""):
    """Зберігає експертну оцінку модератора ОМС для системи Active Learning."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO petition_feedback (timestamp, query_title, matched_title, is_duplicate, comment)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            query_title,
            matched_title,
            int(is_duplicate),
            comment
        ))
        conn.commit()