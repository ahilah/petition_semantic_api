import re
import pandas as pd
import spacy
from spacy.pipeline import EntityRuler
from geocoding import compute_geodesic_penalty

DEFAULT_THRESHOLD_DUPLICATE: float = 0.88
DEFAULT_THRESHOLD_REVIEW: float = 0.76
DEFAULT_W_SEM: float = 0.65
DEFAULT_W_DOM: float = 0.25
DEFAULT_W_GEOD: float = 0.10
DEFAULT_GEO_SIGMA_KM: float = 1.5

STREET_TYPE_TOKENS = {
    "вулиця", "вул", "проспект", "просп", "провулок", "пров",
    "бульвар", "бульв", "майдан", "площа", "місто", "район", "село", "область",
    "мікрорайон", "жм", "проїзд", "узвіз", "набережна", "шосе"
}


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

    ruler = nlp_model.add_pipe("entity_ruler", before="ner")

    patterns = [
        # 1. Заклади освіти (SCHOOL)
        {
            "label": "SCHOOL",
            "pattern": [
                {"LOWER": {"IN": ["школа", "школи", "школі", "ліцей", "ліцею", "ліцеї", "гімназія", "гімназії", "днз", "дитсадок", "садочок", "дсюш"]}},
                {"IS_PUNCT": True, "OP": "?"},
                {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}},
            ],
        },
        {
            "label": "SCHOOL",
            "pattern": [
                {"LOWER": {"IN": ["школа", "школи", "школі", "ліцей", "ліцею", "ліцеї", "гімназія", "гімназії", "днз", "дитсадок", "садочок", "дсюш"]}},
                {"LOWER": {"IN": ["№", "номер"]}},
                {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}},
            ],
        },
        # 2. Транспортні маршрути (ROUTE)
        {
            "label": "ROUTE",
            "pattern": [
                {"LOWER": {"IN": ["маршрут", "маршруту", "маршрутка", "маршрутки", "автобус", "автобуса", "тролейбус", "тролейбуса", "трамвай", "трамвая"]}},
                {"IS_PUNCT": True, "OP": "?"},
                {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}},
            ],
        },
        {
            "label": "ROUTE",
            "pattern": [
                {"LOWER": {"IN": ["маршрут", "маршруту", "маршрутка", "маршрутки", "автобус", "автобуса", "тролейбус", "тролейбуса", "трамвай", "трамвая"]}},
                {"LOWER": {"IN": ["№", "номер"]}},
                {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}},
            ],
        },
        # 3. Медичні заклади (HOSPITAL)
        {
            "label": "HOSPITAL",
            "pattern": [
                {"LOWER": {"IN": ["лікарня", "лікарні", "поліклініка", "поліклініки", "амбулаторія", "амбулаторії"]}},
                {"IS_PUNCT": True, "OP": "?"},
                {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}},
            ],
        },
        {
            "label": "HOSPITAL",
            "pattern": [
                {"LOWER": {"IN": ["лікарня", "лікарні", "поліклініка", "поліклініки", "амбулаторія", "амбулаторії"]}},
                {"LOWER": {"IN": ["№", "номер"]}},
                {"SHAPE": {"IN": ["d", "dd", "ddd", "dddd", "ddX", "dX"]}},
            ],
        },
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
        text = re.sub(r"<[^>]+>", " ", val)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def detect_domain(text: str) -> str:
        """
        Класифікатор доменів на основі лематизованих токенів spaCy
        з високою лінгвістичною точністю та стійкістю до відмінків.
        """
        if not text:
            return "GENERAL"

        clean_text = DomainContextEngine.clean_html(text)

        if nlp is not None:
            doc = nlp(clean_text.lower())
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        else:
            tokens = clean_text.lower().split()

        scores = {
            "ROADS_INFRA": sum(
                1
                for token in tokens
                if any(
                    token.startswith(kw)
                    for kw in [
                        "дорог", "дорож", "вулиц", "вул", "провул", "проспект",
                        "асфальт", "яма", "тротуар", "світлофор", "перехід"
                    ]
                )
            ),
            "EDUCATION": sum(
                1
                for token in tokens
                if any(
                    token.startswith(kw)
                    for kw in ["школ", "ліцей", "гімназі", "освіт", "днз", "садоц", "садок", "учен", "клас", "педагог"]
                )
            ),
            "TRANSPORT": sum(
                1
                for token in tokens
                if any(
                    token.startswith(kw)
                    for kw in ["маршрут", "автобус", "тролейбус", "трамвай", "перевез", "графік", "розклад", "зупинк"]
                )
            ),
            "ANIMALS": sum(
                1
                for token in tokens
                if any(
                    token.startswith(kw)
                    for kw in ["тварин", "собак", "кіт", "кот", "притулок", "стериліз", "безпритуль", "вигул"]
                )
            ),
            "ECOLOGY_PARKS": sum(
                1
                for token in tokens
                if any(
                    token.startswith(kw)
                    for kw in ["парк", "сквер", "дерев", "озелен", "смітт", "урн", "відход", "повітр", "озер", "річ"]
                )
            ),
            "HOUSING": sum(
                1
                for token in tokens
                if any(
                    token.startswith(kw)
                    for kw in ["опален", "будинок", "прибудин", "під'їзд", "дах", "підвал", "ліфт", "водопостач", "жкг", "комірн"]
                )
            ),
        }

        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "GENERAL"

    @staticmethod
    def _normalize_street_type(prefix: str) -> str:
        p = (prefix or "").strip().lower().rstrip(".")
        if p.startswith("вул"):
            return "вулиця"
        if p.startswith("просп"):
            return "проспект"
        if p.startswith("пров"):
            return "провулок"
        if p.startswith("бульв"):
            return "бульвар"
        if p.startswith("площ") or p.startswith("майдан"):
            return "площа"
        return p

    @classmethod
    def _format_street_entry(cls, street_name: str, street_type: str = "") -> str:
        clean_name = " ".join(street_name.lower().split()).strip()
        if not clean_name:
            return ""
        norm_type = cls._normalize_street_type(street_type)
        return f"{norm_type} {clean_name}".strip() if norm_type else clean_name

    @classmethod
    def extract_entities(cls, text: str) -> dict:
        """
        Комплексна екстракція сутностей через spaCy Doc (NER + EntityRuler + лематизація).
        """
        if not text:
            return {
                "streets": set(),
                "schools": set(),
                "routes": set(),
                "hospitals": set(),
                "quoted": set(),
                "domain": "GENERAL",
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
                if ent.label_ == "SCHOOL":
                    num = re.findall(r"\d+[а-яіє]?", ent.text.lower())
                    if num:
                        schools.add(num[-1])
                elif ent.label_ == "ROUTE":
                    num = re.findall(r"\d+[а-яіє]?", ent.text.lower())
                    if num:
                        routes.add(num[-1])
                elif ent.label_ == "HOSPITAL":
                    num = re.findall(r"\d+[а-яіє]?", ent.text.lower())
                    if num:
                        hospitals.add(num[-1])
                elif ent.label_ in ["LOC", "FAC"]:
                    ent_tokens = [t for t in ent if t.pos_ not in ["ADP", "PUNCT", "CCONJ"]]
                    detected_type = ""
                    name_tokens = []
                    for tok in ent_tokens:
                        lem = tok.lemma_.lower()
                        if lem in STREET_TYPE_TOKENS:
                            if not detected_type:
                                detected_type = lem
                        else:
                            name_tokens.append(lem)
                    clean_lemma = " ".join(name_tokens).strip()
                    if clean_lemma and len(clean_lemma) > 2 and not clean_lemma.isdigit():
                        formatted = cls._format_street_entry(clean_lemma, detected_type)
                        if formatted:
                            streets.add(formatted)

        street_pattern = (
            r"(?:по|на|біля|вздовж|до)?\s*"
            r"(вул\.?|вулиц[яіцьюем]|просп\.?|проспект[а-я]*|пров\.?|провул(?:ок|ка)?|бульв\.?|бульвар[а-я]*|майдан[а-я]*|площ[аі])"
            r"\s+([А-ЯІЄЇа-яієїґ\w\-\’]+(?:\s+[А-ЯІЄЇа-яієїґ\w\-\’]+)?)"
        )
        intersection_pattern = (
            r"перехрест[іях]+\s+вулиц[яіцьюем]+\s+"
            r"([А-ЯІЄЇа-яієїґ\w\-\’]+)\s+(?:та|і|й)\s+"
            r"([А-ЯІЄЇа-яієїґ\w\-\’]+(?:\s+[А-ЯІЄЇа-яієїґ\w\-\’]+)?)"
        )

        for st_type, st_name in re.findall(street_pattern, clean_text, re.IGNORECASE):
            token_clean = st_name.strip().lower()
            if token_clean not in ["мікрорайону", "району", "міста", "села", "області", "громади", "центру"]:
                if nlp:
                    lemma_val = " ".join([t.lemma_.lower() for t in nlp(token_clean)])
                else:
                    lemma_val = token_clean
                formatted = cls._format_street_entry(lemma_val, st_type)
                if formatted:
                    streets.add(formatted)

        for st1, st2 in re.findall(intersection_pattern, clean_text, re.IGNORECASE):
            for s in [st1, st2]:
                s_clean = s.strip().lower()
                if nlp:
                    lemma_val = " ".join([t.lemma_.lower() for t in nlp(s_clean)])
                else:
                    lemma_val = s_clean
                formatted = cls._format_street_entry(lemma_val, "вулиця")
                if formatted:
                    streets.add(formatted)

        quoted = set(re.findall(r'["«]([^"»]+)["»]', clean_text))
        clean_quoted = {q.strip().lower() for q in quoted if len(q.strip()) > 2}

        return {
            "streets": streets,
            "schools": schools,
            "routes": routes,
            "hospitals": hospitals,
            "quoted": clean_quoted,
            "domain": domain,
        }

    @classmethod
    def _split_street(cls, entry: str) -> tuple[str, str]:
        parts = entry.strip().lower().split()
        if len(parts) >= 2 and parts[0] in STREET_TYPE_TOKENS:
            return cls._normalize_street_type(parts[0]), " ".join(parts[1:])
        return "", " ".join(parts)

    @classmethod
    def has_street_conflict(cls, streets_q: set, streets_c: set) -> bool:
        """
        Перевіряє, чи є просторовий/назвовий конфлікт між множинами локацій запиту та кандидата.
        Повертає False (конфлікту немає), лише якщо знайдено збіг назви локації
        з узгодженим типом об'єкта (наприклад, 'вулиця шевченка' vs 'вулиця шевченка',
        а не 'вулиця шевченка' vs 'провулок шевченка').
        """
        if not streets_q or not streets_c:
            return False

        for eq in streets_q:
            type_q, name_q = cls._split_street(eq)
            words_q = set(name_q.split())
            if not words_q:
                continue

            for ec in streets_c:
                type_c, name_c = cls._split_street(ec)
                words_c = set(name_c.split())
                if not words_c:
                    continue

                # Якщо обидва типи вказані і вони різні (наприклад, вулиця vs провулок) — це конфлікт
                if type_q and type_c and type_q != type_c:
                    continue

                # Повний збіг назви або перетин повнозначних слів назви за сумісного типу
                if name_q == name_c or words_q.intersection(words_c):
                    return False

        return True

    @classmethod
    def evaluate_match(
        cls,
        query_text: str,
        candidate_row,
        base_semantic_score: float,
        threshold_duplicate: float = DEFAULT_THRESHOLD_DUPLICATE,
        threshold_review: float = DEFAULT_THRESHOLD_REVIEW,
        w_sem: float = DEFAULT_W_SEM,
        w_dom: float = DEFAULT_W_DOM,
        w_geod: float = DEFAULT_W_GEOD,
        use_geo: bool = True,
    ) -> dict:
        entities_q = cls.extract_entities(query_text)
        candidate_full_text = f"{candidate_row['title']} {cls.clean_html(candidate_row.get('text', ''))}"
        entities_c = cls.extract_entities(candidate_full_text)

        reasons = []
        s_sem = float(base_semantic_score)

        # 1. Доменна та сутнісна узгодженість (p_dom у діапазоні [0.0; 1.0])
        p_dom = 1.0

        if entities_q["domain"] != "GENERAL" and entities_c["domain"] != "GENERAL":
            if entities_q["domain"] != entities_c["domain"]:
                domain_names = {
                    "ROADS_INFRA": "Дороги/Інфраструктура",
                    "EDUCATION": "Освіта",
                    "TRANSPORT": "Громадський транспорт",
                    "ANIMALS": "Захист тварин",
                    "ECOLOGY_PARKS": "Екологія та парки",
                    "HOUSING": "ЖКГ/Благоустрій",
                }
                d_q = domain_names.get(entities_q["domain"], entities_q["domain"])
                d_c = domain_names.get(entities_c["domain"], entities_c["domain"])
                p_dom *= 0.45
                reasons.append(
                    f"Розбіжність сфер: запит стосується '{d_q}', а знайдена ініціатива — '{d_c}' "
                    f"(коефіцієнт домену p_dom = {p_dom:.2f})"
                )

        if entities_q["schools"] and entities_c["schools"]:
            if not entities_q["schools"].intersection(entities_c["schools"]):
                p_dom *= 0.40
                reasons.append(
                    f"Розбіжність номерів закладів освіти (№{', '.join(sorted(entities_q['schools']))} проти бази)"
                )
        elif entities_q["schools"] and not entities_c["schools"]:
            p_dom *= 0.55
            reasons.append(
                "Конфлікт об'єктів: запит містить заклад освіти, а знайдена ініціатива не стосується школи"
            )

        if entities_q["routes"] and entities_c["routes"]:
            if not entities_q["routes"].intersection(entities_c["routes"]):
                p_dom *= 0.45
                reasons.append(
                    f"Розбіжність номерів маршрутів (№{', '.join(sorted(entities_q['routes']))} проти бази)"
                )

        if entities_q["hospitals"] and entities_c["hospitals"]:
            if not entities_q["hospitals"].intersection(entities_c["hospitals"]):
                p_dom *= 0.45
                reasons.append(
                    f"Розбіжність номерів медичних закладів (№{', '.join(sorted(entities_q['hospitals']))} проти бази)"
                )
        elif entities_q["hospitals"] and not entities_c["hospitals"]:
            p_dom *= 0.60
            reasons.append(
                "Конфлікт об'єктів: запит містить медичний заклад, а знайдена ініціатива не вказує номер закладу"
            )

        if (entities_q["schools"] and entities_c["routes"]) or (
            entities_q["routes"] and entities_c["schools"]
        ):
            p_dom *= 0.40
            reasons.append("Конфлікт категорій: зіткнення закладу освіти та міського маршруту")

        if entities_q["quoted"] and entities_c["quoted"]:
            if not entities_q["quoted"].intersection(entities_c["quoted"]):
                p_dom *= 0.65
                reasons.append("Розбіжність назв об'єктів/ЖК у лапках")

        q_has_spec = bool(
            entities_q["streets"]
            or entities_q["schools"]
            or entities_q["routes"]
            or entities_q["hospitals"]
            or entities_q["quoted"]
        )
        c_has_spec = bool(
            entities_c["streets"]
            or entities_c["schools"]
            or entities_c["routes"]
            or entities_c["hospitals"]
            or entities_c["quoted"]
        )

        # Застосовуємо м'який штраф лише якщо в кандидата є хоча б якийсь текст/опис,
        # щоб не занижувати Recall у парах самих лише заголовків
        cand_has_body = bool(cls.clean_html(candidate_row.get("text", "")))
        if q_has_spec and not c_has_spec and cand_has_body:
            p_dom *= 0.90
            reasons.append(
                "Штраф за специфічність: запит містить точний об'єкт/локацію, тоді як знайдена петиція має загальний характер"
            )

        # 2. Геопросторовий коефіцієнт за формулою Гаверсинуса (p_geod у діапазоні (0.0; 1.0])
        p_geod = 1.0
        if use_geo and entities_q["streets"] and entities_c["streets"]:
            if cls.has_street_conflict(entities_q["streets"], entities_c["streets"]):
                auth_hint = str(candidate_row.get("authority", ""))
                geod_penalty, dist_km = compute_geodesic_penalty(
                    entities_q["streets"],
                    entities_c["streets"],
                    authority_hint=auth_hint,
                    sigma=DEFAULT_GEO_SIGMA_KM,
                )
                if dist_km is not None:
                    p_geod = geod_penalty
                    reasons.append(
                        f"Геопросторовий аналіз: відстань d ≈ {dist_km} км між локаціями "
                        f"(P_geod = {p_geod:.2f}, σ = {DEFAULT_GEO_SIGMA_KM} км)"
                    )
                else:
                    p_geod = 0.65
                    reasons.append(
                        f"Розбіжність локацій: '{', '.join(sorted(entities_q['streets']))}' "
                        f"проти '{', '.join(sorted(entities_c['streets']))}' (резервний P_geod = {p_geod:.2f})"
                    )

        # 3. Прозора зважена інтегральна оцінка без каскадних множників
        sem_component = s_sem
        dom_component = s_sem * p_dom
        geod_component = s_sem * p_geod

        weight_sum = w_sem + w_dom + w_geod
        if weight_sum > 0:
            integral_score = (
                w_sem * sem_component
                + w_dom * dom_component
                + w_geod * geod_component
            ) / weight_sum
        else:
            integral_score = s_sem

        final_score = round(max(0.0, min(1.0, integral_score)), 3)

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
            "reasons": reasons,
        }