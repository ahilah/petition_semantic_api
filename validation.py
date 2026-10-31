import re

def is_valid_input_text(text: str) -> tuple[bool, str]:
    """Перевіряє вхідний текст на відповідність базовим критеріям валідності."""
    cleaned = text.strip()
    if len(cleaned) < 8:
        return False, "Запит занадто короткий для семантичного аналізу (мінімум 8 символів)."
    if re.search(r'(.)\1{3,}', cleaned.lower()):
        return False, "Виявлено незв'язний набір символів або повторювані літери. Введіть коректну назву ініціативи."
    words = re.findall(r'[а-яіїєґa-z0-9\-]+', cleaned.lower())
    if len(words) < 2:
        return False, "Заголовок має містити хоча б 2 повнозначні слова."
    return True, ""