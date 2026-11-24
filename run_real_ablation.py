import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score
from unittest.mock import patch

try:
    from engine import DomainContextEngine
    print("[*] Модуль engine.py успішно імпортовано.")
except ImportError as e:
    print(f"[!] Помилка імпорту: {e}")
    exit()

def evaluate_stage(df, y_true, base_scores, theta_opt, patch_geography=False):
    final_scores = np.copy(base_scores)
    
    for idx, row in df.iterrows():
        # Перевіряємо тільки ті пари, які E5 визнав як дублікати (щоб зекономити час геокодера)
        if base_scores[idx] >= theta_opt:
            candidate_row = {
                'title': str(row['title_2']),
                'text': '', 
                'authority': str(row['authority_2'])
            }
            
            if patch_geography:
                with patch('engine.DomainContextEngine.has_street_conflict', return_value=False):
                    res = DomainContextEngine.evaluate_match(
                        query_text=str(row['title_1']),
                        candidate_row=candidate_row,
                        base_semantic_score=base_scores[idx]
                    )
            else:
                res = DomainContextEngine.evaluate_match(
                    query_text=str(row['title_1']),
                    candidate_row=candidate_row,
                    base_semantic_score=base_scores[idx]
                )
                
            final_scores[idx] = res['final_score']
            
    preds = (final_scores >= theta_opt).astype(int)
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    return fp, prec, rec, f1

def main():
    print("[*] Завантажуємо Ground Truth...")
    df = pd.read_excel('ground_truth_300_balanced.xlsx')
    y_true = df['is_similar'].values
    theta_opt = 0.91

    print("[*] Завантажуємо модель intfloat/multilingual-e5-small...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")

   
    q_texts = ["query: " + str(t) for t in df['title_1']]
    c_texts = ["passage: " + str(t) for t in df['title_2']]
    
    emb_q = model.encode(q_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    emb_c = model.encode(c_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    
    
    base_scores = util.pairwise_cos_sim(emb_q, emb_c).cpu().numpy()
    
    # ---------------------------------------------------------
    # STAGE 1: Чистий E5-small
    # ---------------------------------------------------------
    preds_1 = (base_scores >= theta_opt).astype(int)
    fp_1 = int(np.sum((preds_1 == 1) & (y_true == 0)))
    prec_1 = precision_score(y_true, preds_1, zero_division=0)
    rec_1 = recall_score(y_true, preds_1, zero_division=0)
    f1_1 = f1_score(y_true, preds_1, zero_division=0)
    
    # ---------------------------------------------------------
    # STAGE 2: NER (Без геопросторових штрафів Гаверсинуса)
    # ---------------------------------------------------------
    print(f"[*] Базових FP після E5: {fp_1}. Запускаємо NER (spaCy)...")
    fp_2, prec_2, rec_2, f1_2 = evaluate_stage(df, y_true, base_scores, theta_opt, patch_geography=True)
    
    # ---------------------------------------------------------
    # STAGE 3: Повний конвеєр (NER + Гаверсинус)
    # ---------------------------------------------------------
    print(f"[*] FP після NER: {fp_2}. Запускаємо повний конвеєр (Гаверсинус)...")
    fp_3, prec_3, rec_3, f1_3 = evaluate_stage(df, y_true, base_scores, theta_opt, patch_geography=False)

    # ---------------------------------------------------------
    # ВИВІД ТАБЛИЦІ
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" ТАБЛИЦЯ 4.4 (АБЛЯЦІЙНЕ ДОСЛІДЖЕННЯ НА ТВОЄМУ КОДІ)")
    print("="*75)
    table_data = [
        {"Stage": "1. Тільки Stage 1 (E5-small)", "FP": fp_1, "Precision": round(prec_1, 4), "Recall": round(rec_1, 4), "F1": round(f1_1, 4)},
        {"Stage": "2. Stage 1 + NER (spaCy)", "FP": fp_2, "Precision": round(prec_2, 4), "Recall": round(rec_2, 4), "F1": round(f1_2, 4)},
        {"Stage": "3. Повний конвеєр (NER + Geo)", "FP": fp_3, "Precision": round(prec_3, 4), "Recall": round(rec_3, 4), "F1": round(f1_3, 4)}
    ]
    print(pd.DataFrame(table_data).to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    main()