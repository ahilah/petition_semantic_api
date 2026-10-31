import re
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from geocoding import compute_geodesic_penalty

def init_spacy_pipeline():
    """
    Ініціалізує конвеєр spaCy з моделлю uk_core_news_lg (або fallback на sm)
    та інтегрує компонент EntityRuler перед основним NER для точного доменного розпізнавання.
    """
    try:
        nlp_model = spacy.load("uk_core_news_lg")
    except OSError:
        try:
            nlp_model = spacy.load("uk_core_news_sm")
        except OSError:
            return None

    # Додаємо EntityRuler перед стандартним "ner", щоб наші патерни мали пріоритет
    ruler = nlp_model.add_pipe("entity_ruler", before="ner")
    
    patterns = [
        # 1. Заклади освіти (SCHOOL)
        {"label": "SCHOOL", "pattern": [{"LOWER": {"IN": ["школа", "школи", "школі", "ліцей", "ліцею", "ліцеї", "гімназія", "гімназії", "днз", "дитсадок", "садочок", "дсюш"]}}, {"IS_PUNCT": True, "OP": "?"}, {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}}]},
        {"label": "SCHOOL", "pattern": [{"LOWER": {"IN": ["школа", "школи", "школі", "ліцей", "ліцею", "ліцеї", "гімназія", "гімназії", "днз", "дитсадок", "садочок", "дсюш"]}}, {"LOWER": {"IN": ["№", "номер"]}}, {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}}]},
        
        # 2. Транспортні маршрути (ROUTE)
        {"label": "ROUTE", "pattern": [{"LOWER": {"IN": ["маршрут", "маршруту", "маршрутка", "маршрутки", "автобус", "автобуса", "тролейбус", "тролейбуса", "трамвай", "трамвая"]}}, {"IS_PUNCT": True, "OP": "?"}, {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}}]},
        {"label": "ROUTE", "pattern": [{"LOWER": {"IN": ["маршрут", "маршруту", "маршрутка", "маршрутки", "автобус", "автобуса", "тролейбус", "тролейбуса", "трамвай", "трамвая"]}}, {"LOWER": {"IN": ["№", "номер"]}}, {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}}]},
        
        # 3. Медичні заклади (HOSPITAL)
        {"label": "HOSPITAL", "pattern": [{"LOWER": {"IN": ["лікарня", "лікарні", "поліклініка", "поліклініки", "амбулаторія", "амбулаторії"]}}, {"IS_PUNCT": True, "OP": "?"}, {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}}]},
        {"label": "HOSPITAL", "pattern": [{"LOWER": {"IN": ["лікарня", "лікарні", "поліклініка", "поліклініки", "амбулаторія", "амбулаторії"]}}, {"LOWER": {"IN": ["№", "номер"]}}, {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}}]},
    ]
    ruler.add_patterns(patterns)
    return nlp_model

nlp = init_spacy_pipeline()


class DomainContextEngine:
    @staticmethod
    def clean_html(raw_html):
        if pd.isna(raw_html):
            return ""
        val = str(raw_html).strip()
        if val.lower() in ["example.docx", "none", "nan", "null", "false"]:
            return ""
        text = re.sub(r'<[^>]+>', ' ', val)
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def detect_domain(text: str) -> str:
        """
        Класифікатор доменів на основі лематизованих токенів spaCy
        з високою лінгвістичною точністю та стійкістю до відмінків.
        """
        if not text:
            return "GENERAL"
        
        clean_text = DomainContextEngine.clean_html(text)
        
        # Використовуємо глобальний nlp об'єкт для точної лематизації
        if nlp is not None:
            doc = nlp(clean_text.lower())
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        else:
            tokens = clean_text.lower().split()

        scores = {
            "ROADS_INFRA": sum(1 for token in tokens if any(token.startswith(kw) for kw in ["дорог", "дорож", "вулиц", "вул", "провул", "проспект", "асфальт", "яма", "тротуар", "світлофор", "перехід"])),
            "EDUCATION": sum(1 for token in tokens if any(token.startswith(kw) for kw in ["школ", "ліцей", "гімназі", "освіт", "днз", "садоц", "садок", "учен", "клас", "педагог"])),
            "TRANSPORT": sum(1 for token in tokens if any(token.startswith(kw) for kw in ["маршрут", "автобус", "тролейбус", "трамвай", "перевез", "графік", "розклад", "зупинк"])),
            "ANIMALS": sum(1 for token in tokens if any(token.startswith(kw) for kw in ["тварин", "собак", "кіт", "кот", "притулок", "стериліз", "безпритуль", "вигул"])),
            "ECOLOGY_PARKS": sum(1 for token in tokens if any(token.startswith(kw) for kw in ["парк", "сквер", "дерев", "озелен", "смітт", "урн", "відход", "повітр", "озер", "річ"])),
            "HOUSING": sum(1 for token in tokens if any(token.startswith(kw) for kw in ["опален", "будинок", "прибудин", "під'їзд", "дах", "підвал", "ліфт", "водопостач", "жкг", "комірн"]))
        }
        
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "GENERAL"

    @classmethod
    def extract_entities(cls, text: str) -> dict:
        """
        Комплексна екстракція сутностей через spaCy Doc (NER + EntityRuler + лематизація).
        """
        if not text:
            return {
                "streets": set(), "schools": set(), "routes": set(), 
                "hospitals": set(), "quoted": set(), "domain": "GENERAL"
            }

        clean_text = cls.clean_html(text)
        domain = cls.detect_domain(clean_text)
        
        streets = set()
        schools = set()
        routes = set()
        hospitals = set()

        if nlp is not None:
            doc = nlp(clean_text)

            for ent in doc.ents:
                # 1. Розпізнавання через EntityRuler
                if ent.label_ == "SCHOOL":
                    num = re.findall(r'\d+[а-яіє]?', ent.text.lower())
                    if num:
                        schools.add(num[-1])
                elif ent.label_ == "ROUTE":
                    num = re.findall(r'\d+[а-яіє]?', ent.text.lower())
                    if num:
                        routes.add(num[-1])
                elif ent.label_ == "HOSPITAL":
                    num = re.findall(r'\d+[а-яіє]?', ent.text.lower())
                    if num:
                        hospitals.add(num[-1])
                
                # 2. Розпізнавання топонімів (LOC / FAC) зі зняттям відмінків
                elif ent.label_ in ["LOC", "FAC"]:
                    tokens_lemma = [
                        token.lemma_.lower() for token in ent
                        if token.pos_ not in ["ADP", "PUNCT", "CCONJ"]
                        and token.lemma_.lower() not in [
                            "вулиця", "вул", "проспект", "просп", "провулок", "пров",
                            "бульвар", "бульв", "майдан", "площа", "місто", "район", "село", "область"
                        ]
                    ]
                    clean_lemma = " ".join(tokens_lemma).strip()
                    if clean_lemma and len(clean_lemma) > 2 and not clean_lemma.isdigit():
                        streets.add(clean_lemma)

        # Додатковий страховий regex для перехресть і специфічних відмінків топонімів
        street_patterns = [
            r'(?:по|на|біля|вздовж|до)?\s*(?:вул\.?|вулиц[яіцьюем]|просп\.?|проспект[а-я]*|пров\.?|провул(?:ок|ка)?|бульв\.?|бульвар[а-я]*|майдан[а-я]*|площ[аі])\s+([А-ЯІЄЇа-яієїґ\w\-\’]+(?:\s+[А-ЯІЄЇа-яієїґ\w\-\’]+)?)',
            r'перехрест[іях]+\s+вулиц[яіцьюем]+\s+([А-ЯІЄЇа-яієїґ\w\-\’]+)\s+(?:та|і|й)\s+([А-ЯІЄЇа-яієїґ\w\-\’]+(?:\s+[А-ЯІЄЇа-яієїґ\w\-\’]+)?)'
        ]
        
        for m in re.findall(street_patterns[0], clean_text, re.IGNORECASE):
            token_clean = m.strip().lower()
            if token_clean not in ["мікрорайону", "району", "міста", "села", "області", "громади", "центру"]:
                if nlp:
                    lemma_val = " ".join([t.lemma_.lower() for t in nlp(token_clean)])
                    streets.add(lemma_val)
                else:
                    streets.add(token_clean)

        intersections = re.findall(street_patterns[1], clean_text, re.IGNORECASE)
        for st1, st2 in intersections:
            for s in [st1, st2]:
                s_clean = s.strip().lower()
                if nlp:
                    streets.add(" ".join([t.lemma_.lower() for t in nlp(s_clean)]))
                else:
                    streets.add(s_clean)

        quoted = set(re.findall(r'["«]([^"»]+)["»]', clean_text))
        clean_quoted = {q.strip().lower() for q in quoted if len(q.strip()) > 2}

        return {
            "streets": streets,
            "schools": schools,
            "routes": routes,
            "hospitals": hospitals,
            "quoted": clean_quoted,
            "domain": domain
        }

    @staticmethod
    def has_street_conflict(streets_q: set, streets_c: set) -> bool:
        """
        Нечітке порівняння вулиць: перевіряє, чи є хоча б одне спільне повнозначне слово.
        Запобігає конфлікту між 'Шевченка' та 'Тараса Шевченка'.
        """
        if not streets_q or not streets_c:
            return False
        for sq in streets_q:
            words_q = set(sq.split())
            for sc in streets_c:
                words_c = set(sc.split())
                if words_q.intersection(words_c):
                    return False
        return True

    @classmethod
    def evaluate_match(cls, query_text: str, candidate_row, base_semantic_score: float,
                       threshold_duplicate: float = 0.88, threshold_review: float = 0.76) -> dict:
        entities_q = cls.extract_entities(query_text)
        candidate_full_text = f"{candidate_row['title']} {cls.clean_html(candidate_row.get('text', ''))}"
        entities_c = cls.extract_entities(candidate_full_text)

        reasons = []
        penalized_score = float(base_semantic_score)

        # 1. ТЕМАТИЧНА ІЗОЛЯЦІЯ ДОМЕНІВ (detect_domain)
        # Блокує хибні дублікати між різними сферами при спільних словах типу 'ремонт'
        if entities_q["domain"] != "GENERAL" and entities_c["domain"] != "GENERAL":
            if entities_q["domain"] != entities_c["domain"]:
                domain_names = {
                    "ROADS_INFRA": "Дороги/Інфраструктура",
                    "EDUCATION": "Освіта",
                    "TRANSPORT": "Громадський транспорт",
                    "ANIMALS": "Захист тварин",
                    "ECOLOGY_PARKS": "Екологія та парки",
                    "HOUSING": "ЖКГ/Благоустрій"
                }
                d_q = domain_names.get(entities_q["domain"], entities_q["domain"])
                d_c = domain_names.get(entities_c["domain"], entities_c["domain"])
                reasons.append(f"Розбіжність сфер: запит стосується '{d_q}', а знайдена ініціатива — '{d_c}'")
                penalized_score = min(penalized_score, 0.50)

        # 2. КОНФЛІКТ ТОПОНІМІВ (Неперервна просторова експонента Гаверсинуса)
        if entities_q["streets"] and entities_c["streets"]:
            if cls.has_street_conflict(entities_q["streets"], entities_c["streets"]):
                # Запитуємо просторову відстань d та коефіцієнт P_geod = exp(-d / sigma)
                auth_hint = str(candidate_row.get('authority', ''))
                p_geod, dist_km = compute_geodesic_penalty(entities_q["streets"], entities_c["streets"], authority_hint=auth_hint, sigma=1.5)

                if dist_km is not None:
                    # Якщо геокоординати знайдено — масштабуємо бал схожості пропорційно відстані
                    # Наприклад: для d=0.5 км (P=0.72) бал зріжеться мінімально,
                    # а для d=5.0 км (P=0.04) бал впаде гарантовано нижче зони рев'ю.
                    geod_score = base_semantic_score * p_geod
                    penalized_score = min(penalized_score, max(0.40, geod_score))
                    reasons.append(
                        f"Геопросторовий аналіз: відстань d ≈ {dist_km} км між об'єктами "
                        f"(P_geod = {p_geod:.2f}, σ = 1.5 км)"
                    )
                else:
                    # Fallback-штраф, якщо в OpenStreetMap немає точних координат локації
                    reasons.append(f"Розбіжність локацій: '{', '.join(entities_q['streets'])}' проти '{', '.join(entities_c['streets'])}'")
                    penalized_score = min(penalized_score, 0.65)

        # 3. КОНФЛІКТ ЗАКЛАДІВ ОСВІТИ (EntityRuler SCHOOL)
        if entities_q["schools"] and entities_c["schools"]:
            if not entities_q["schools"].intersection(entities_c["schools"]):
                reasons.append(f"Розбіжність номерів закладів освіти (№{', '.join(entities_q['schools'])} проти бази)")
                penalized_score = min(penalized_score, 0.60)

        # ДОДАТКОВИЙ ШТРАФ: якщо запит стосується конкретної школи, а знайдена петиція — ні (наприклад, ремонт вулиці)
        elif entities_q["schools"] and not entities_c["schools"]:
            reasons.append("Конфлікт об'єктів: запит містить заклад освіти, а знайдена ініціатива не стосується школи")
            penalized_score = min(penalized_score, 0.55)

        # 4. КОНФЛІКТ МАРШРУТІВ (EntityRuler ROUTE)
        if entities_q["routes"] and entities_c["routes"]:
            if not entities_q["routes"].intersection(entities_c["routes"]):
                reasons.append(f"Розбіжність номерів маршрутів (№{', '.join(entities_q['routes'])} проти бази)")
                penalized_score = min(penalized_score, 0.58)

        # 5. МІЖДОМЕННИЙ КОНФЛІКТ ЧИСЕЛ (Школа проти Маршруту)
        if (entities_q["schools"] and entities_c["routes"]) or (entities_q["routes"] and entities_c["schools"]):
            reasons.append("Конфлікт категорій: зіткнення закладу освіти та міського маршруту")
            penalized_score = min(penalized_score, 0.52)

        # 6. КОНФЛІКТ ВЛАСНИХ НАЗВ У ЛАПКАХ
        if entities_q["quoted"] and entities_c["quoted"]:
            if not entities_q["quoted"].intersection(entities_c["quoted"]):
                reasons.append("Розбіжність назв об'єктів/ЖК у лапках")
                penalized_score = min(penalized_score, 0.67)

        # 7. ШТРАФ ЗА СПЕЦИФІЧНІСТЬ
        q_has_spec = bool(entities_q["streets"] or entities_q["schools"] or entities_q["routes"] or entities_q["hospitals"] or entities_q["quoted"])
        c_has_spec = bool(entities_c["streets"] or entities_c["schools"] or entities_c["routes"] or entities_c["hospitals"] or entities_c["quoted"])

        if q_has_spec and not c_has_spec:
            reasons.append("Штраф за специфічність: запит містить точний об'єкт/локацію, тоді як знайдена петиція має загальний характер")
            penalized_score = min(penalized_score, 0.72)

        final_score = round(penalized_score, 3)

        # Класифікація за динамічними порогами
        if final_score >= threshold_duplicate:
            verdict, status_text, badge_class = "DUPLICATE", "ПОВНИЙ ДУБЛІКАТ", "badge-duplicate"
        elif final_score >= threshold_review:
            verdict, status_text, badge_class = "REVIEW", "ПОТРЕБУЄ РЕВ'Ю", "badge-review"
        else:
            verdict, status_text, badge_class = "UNIQUE", "ОРИГІНАЛЬНА ІНІЦІАТИВА", "badge-unique"

        return {
            "final_score": final_score,
            "verdict": verdict,
            "status_text": status_text,
            "badge_class": badge_class,
            "reasons": reasons
        }