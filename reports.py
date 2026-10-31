import io
import os
import urllib.request
import pandas as pd
from datetime import datetime
from fpdf import FPDF


# Визначаємо кореневу папку проєкту та шлях до шрифту в папці assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "assets", "DejaVuSans.ttf")
FONT_URL = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/ttf/DejaVuSans.ttf"


def ensure_font_exists():
    """Перевіряє наявність шрифту з кирилицею в папці assets або завантажує його за потреби."""
    if not os.path.exists(FONT_PATH):
        # На випадок, якщо папка assets раптом не створена
        os.makedirs(os.path.dirname(FONT_PATH), exist_ok=True)
        
        win_arial = "C:/Windows/Fonts/arial.ttf"
        if os.path.exists(win_arial):
            return win_arial

        try:
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception:
            return None
    return FONT_PATH


def export_to_csv(results_df: pd.DataFrame) -> bytes:
    """Експортує знайдені збіги у формат CSV (включно з причинами/штрафами)."""
    cols_to_export = [
        col for col in ['title', 'authority', 'author', 'final_score', 'verdict', 'base_score', 'reasons']
        if col in results_df.columns
    ]
    df_export = results_df[cols_to_export].copy()
    
    # Якщо поле reasons є списком, перетворюємо його на звичайний текстовий рядок через крапку з комою
    if 'reasons' in df_export.columns:
        df_export['reasons'] = df_export['reasons'].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else (str(x) if pd.notna(x) else "")
        )

    rename_dict = {
        'title': 'Назва петиції',
        'authority': 'Орган місцевого самоврядування',
        'author': 'Автор ініціативи',
        'final_score': 'Підсумковий бал схожості',
        'verdict': 'Вердикт системи',
        'base_score': 'Базова косинусна подібність',
        'reasons': 'Причини коригування (NER / Домени)'
    }
    df_export = df_export.rename(columns=rename_dict)
    return df_export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def export_to_pdf(query_title: str, query_body: str, authority: str, results_df: pd.DataFrame, top_verdict: str) -> bytes:
    """Генерує офіційний протокол перевірки звернення у форматі PDF."""
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    
    # Робоча ширина сторінки з урахуванням полів
    w = pdf.epw 

    font_file = ensure_font_exists()
    font_name = "CustomFont"
    if font_file:
        pdf.add_font(font_name, "", font_file)
        pdf.set_font(font_name, size=11)
    else:
        pdf.set_font("Helvetica", size=10)

    # 1. Шапка документа
    pdf.set_font(font_name, size=14)
    pdf.set_x(15)
    pdf.multi_cell(w, 7, "ПРОТОКОЛ АВТОМАТИЗОВАНОЇ ПЕРЕВІРКИ ЕЛЕКТРОННОЇ ПЕТИЦІЇ", align="C")
    
    pdf.set_font(font_name, size=9)
    pdf.set_x(15)
    pdf.multi_cell(w, 5, f"Дата формування: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')} | Юрисдикція: {authority}", align="C")
    pdf.ln(3)
    
    # Горизонтальна лінія
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # 2. Інформація про подану петицію
    pdf.set_font(font_name, size=12)
    pdf.set_x(15)
    pdf.multi_cell(w, 6, "1. Параметри вхідного звернення:")
    
    pdf.set_font(font_name, size=9)
    pdf.set_x(15)
    pdf.multi_cell(w, 5, f"Заголовок: {query_title.strip()}")
    
    if query_body and query_body.strip():
        pdf.set_x(15)
        pdf.multi_cell(w, 5, f"Текст (фрагмент): {query_body.strip()[:250]}...")
    pdf.ln(3)

    # 3. Підсумковий вердикт
    pdf.set_font(font_name, size=12)
    pdf.set_x(15)
    pdf.multi_cell(w, 6, "2. Висновок аналітичного модуля:")
    
    pdf.set_font(font_name, size=9)
    pdf.set_x(15)
    pdf.multi_cell(w, 5, f"Статус системи: {top_verdict}")
    pdf.ln(4)

    # 4. Перелік знайдених збігів
    pdf.set_font(font_name, size=12)
    pdf.set_x(15)
    pdf.multi_cell(w, 6, "3. Виявлені семантичні аналоги (топ знайдених):")

    if results_df.empty:
        pdf.set_font(font_name, size=9)
        pdf.set_x(15)
        pdf.multi_cell(w, 5, "Аналогів вище порогового значення не знайдено.")
    else:
        for idx, (_, row) in enumerate(results_df.iterrows(), start=1):
            title = str(row.get('title', 'Без назви')).strip()
            score = row.get('final_score', 0.0)
            status = row.get('status_text', row.get('verdict', ''))
            auth = str(row.get('authority', 'Не вказано')).strip()
            
            pdf.set_x(15)
            pdf.set_font(font_name, size=9)
            pdf.multi_cell(w, 5, f"[{idx}] {status} | Бал схожості: {score} | Рада: {auth}")
            
            pdf.set_x(15)
            pdf.set_font(font_name, size=8)
            pdf.multi_cell(w, 4, f"Назва: {title}")
            
            # Покращене виведення причин (кожної з нового рядка з маркером)
            reasons_data = row.get('reasons', [])
            if reasons_data:
                reasons_list = reasons_data if isinstance(reasons_data, list) else [str(reasons_data)]
                for reason in reasons_list:
                    if str(reason).strip():
                        pdf.set_x(18) # Невеликий відступ праворуч для ієрархії
                        pdf.multi_cell(w - 3, 4, f" • Аргумент системи: {str(reason).strip()}")
                
            pdf.ln(2)

    pdf.ln(4)
    pdf.set_font(font_name, size=8)
    pdf.set_x(15)
    pdf.multi_cell(w, 4, "Документ згенеровано автоматизованою системою муніципального моніторингу електронних петицій.", align="R")

    return bytes(pdf.output())