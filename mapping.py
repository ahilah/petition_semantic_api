import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd

# Базові координати центрів громад
# Порядок важливий: специфічні назви (наприклад, Кременчук) мають іти перед ширшими (Полтава)
CITY_CENTERS = [
    ("кременчу", (49.0630, 33.4042)),
    ("терешк", (49.5292, 34.6083)),
    ("полтав", (49.5883, 34.5514)),
    ("володимир", (50.8482, 24.3222)),
    ("рівнен", (50.6199, 26.2516)),
    ("бровар", (50.5113, 30.7901)),
    ("борщів", (48.8028, 26.0425)),
    ("покровськ", (48.2827, 37.1758)),
    ("бурштин", (49.2611, 24.6300)),
    ("виноградів", (48.1470, 23.0280)),
    ("мостиськ", (49.7917, 23.1517)),
    ("комарн", (49.6303, 23.7022)),
    ("львів", (49.8397, 24.0297)),
    ("київ", (50.4501, 30.5234)),
    ("слобожан", (49.7289, 36.5208))
]

DEFAULT_CENTER = (49.1, 31.2)  # Географічний центр України

def get_status_color(status_text: str) -> str:
    st_low = str(status_text).lower()
    if any(w in st_low for w in ["збір", "активн", "триває"]):
        return "green"
    elif any(w in st_low for w in ["розгляд", "відповід", "підтриман"]):
        return "blue"
    elif any(w in st_low for w in ["не підтриман", "відхилен"]):
        return "red"
    return "gray"

def find_city_coordinates(authority_str: str) -> tuple[float, float] | None:
    auth_low = str(authority_str).lower()
    for key, coords in CITY_CENTERS:
        if key in auth_low:
            return coords
    return None

def build_petition_map(df_corpus: pd.DataFrame, selected_authority: str, show_heatmap: bool = True) -> folium.Map:
    df_scope = df_corpus.copy()
    
    if selected_authority != "Всі юрисдикції (Національний рівень)":
        df_scope = df_scope[df_scope['authority'] == selected_authority]
        matched_coords = find_city_coordinates(selected_authority)
        center = matched_coords if matched_coords else DEFAULT_CENTER
        zoom_start = 12 if matched_coords else 7
    else:
        center = DEFAULT_CENTER
        zoom_start = 6

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True
    )

    heat_data = []
    marker_cluster = MarkerCluster(name="Петиції громади").add_to(m)

    # Збалансована вибірка: беремо до 40 петицій з кожної громади, щоб показати всі міста
    if 'authority' in df_scope.columns:
        sample_df = df_scope.groupby('authority', group_keys=False).apply(
            lambda g: g.head(40)
        ).reset_index(drop=True)
    else:
        sample_df = df_scope.head(400)

    for idx, row in sample_df.iterrows():
        title = str(row.get('title', 'Без назви')).replace("'", "\\'").replace('"', '&quot;')
        status = str(row.get('status', 'Невідомо'))
        authority = str(row.get('authority', 'ОМС'))
        
        lat, lon = row.get('lat', None), row.get('lon', None)
        
        if pd.isna(lat) or pd.isna(lon) or not lat or not lon:
            coords = find_city_coordinates(authority)
            if coords:
                # Акуратний розкид у межах 1–2 км від центру міста
                jitter_lat = (hash(f"{title}_lat_{idx}") % 1000 - 500) / 45000.0
                jitter_lon = (hash(f"{title}_lon_{idx}") % 1000 - 500) / 35000.0
                lat = coords[0] + jitter_lat
                lon = coords[1] + jitter_lon

        if lat and lon:
            heat_data.append([lat, lon, 1.0])
            
            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 200px;">
                <h5 style="margin: 0 0 5px 0; color: #1E293B;">{title}</h5>
                <p style="margin: 2px 0; font-size: 12px; color: #64748B;"><b>ОМС:</b> {authority}</p>
                <span style="font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 4px; background: #F1F5F9;">
                    Статус: {status}
                </span>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                popup=folium.Popup(popup_html, max_width=300),
                color=get_status_color(status),
                fill=True,
                fill_color=get_status_color(status),
                fill_opacity=0.7
            ).add_to(marker_cluster)

    if show_heatmap and heat_data:
        HeatMap(
            heat_data,
            radius=16,
            blur=12,
            max_zoom=13,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m