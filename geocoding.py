import os
import json
import math
import urllib.parse
import urllib.request
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "data", "coords_cache.json")

def load_coords_cache() -> dict:
    """Завантажує збережений кеш геокоординат."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_coords_cache(cache: dict):
    """Зберігає оновлений кеш у data/coords_cache.json."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_COORDS_CACHE = load_coords_cache()

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Обчислює ортодромічну відстань між двома точками в кілометрах за формулою Гаверсинуса.
    """
    R = 6371.0  # Радіус Землі в км
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def get_location_coordinates(address: str, city_hint: str = "") -> tuple[float, float] | None:
    """
    Отримує широту та довготу (lat, lon) адреси з кешу або через OpenStreetMap Nominatim.
    """
    clean_addr = address.strip().lower()
    clean_city = city_hint.strip().lower() if city_hint else ""
    cache_key = f"{clean_addr}___{clean_city}"

    if cache_key in _COORDS_CACHE:
        coords = _COORDS_CACHE[cache_key]
        return (coords["lat"], coords["lon"]) if coords else None

    # Пошуковий запит до OpenStreetMap Nominatim API
    query_parts = [clean_addr]
    if clean_city and clean_city not in ["всі юрисдикції (національний рівень)", "nan"]:
        # Прибираємо службові слова на кшталт "міська рада", "виконавчий комітет"
        normalized_city = clean_city.replace("міська рада", "").replace("селищна рада", "").strip()
        query_parts.append(normalized_city)
    query_parts.append("Україна")

    search_query = ", ".join(query_parts)
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(search_query)}&format=json&limit=1"
    req = urllib.request.Request(url, headers={'User-Agent': 'MunicipalPetitionAuditApp/1.0 (academic-thesis)'})

    try:
        time.sleep(1.0)  # Дотримання регламенту OpenStreetMap (макс 1 запит/сек)
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode())
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                _COORDS_CACHE[cache_key] = {"lat": lat, "lon": lon}
                save_coords_cache(_COORDS_CACHE)
                return (lat, lon)
    except Exception:
        pass

    # Якщо адресу не знайдено — кешуємо None, щоб не спамити повторно
    _COORDS_CACHE[cache_key] = None
    save_coords_cache(_COORDS_CACHE)
    return None

def compute_geodesic_penalty(streets_q: set, streets_c: set, authority_hint: str = "", sigma: float = 1.5) -> tuple[float, float | None]:
    """
    Обчислює неперервний коефіцієнт просторової близькості P_geod(d) = exp(-d / sigma).
    Повертає:
      - p_factor: число в діапазоні (0; 1]
      - min_distance: мінімальна знайдена відстань у км (або None, якщо геокодування не вдалося)
    """
    if not streets_q or not streets_c:
        return 1.0, None

    min_dist = float('inf')

    for sq in streets_q:
        c1 = get_location_coordinates(sq, authority_hint)
        if not c1:
            continue
        for sc in streets_c:
            c2 = get_location_coordinates(sc, authority_hint)
            if not c2:
                continue
            d = haversine_distance(c1[0], c1[1], c2[0], c2[1])
            if d < min_dist:
                min_dist = d

    if min_dist == float('inf'):
        return 1.0, None

    p_factor = math.exp(-min_dist / sigma)
    return p_factor, round(min_dist, 2)