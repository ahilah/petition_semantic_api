# Використовуємо офіційний легкий образ Python 3.11
FROM python:3.11-slim

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Встановлюємо системні залежності, якщо необхідні для збірки деяких пакетів
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копіюємо файл залежностей та встановлюємо їх
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь решту коду проєкту в контейнер
COPY . .

# Відкриваємо порт 8000 для FastAPI
EXPOSE 8000

# Команда для запуску сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]