from __future__ import annotations

import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.pdfgen import canvas

ROOT = Path.home() / 'iCloud Drive (Archive)' / 'Documents' / 'Documents - Matthias’s 16" MacBook Pro (246)' / 'Paper Pythons' / 'Master Thesis'
FIG_DIR = ROOT / 'figures' / 'part-II'
BOUNDARIES_PATH = ROOT / 'historical_swiss_boundaries.json'
GEOM_PATH = ROOT / 'swiss_municipality_station_geometries.json'

sys.path.insert(0, str(ROOT))
from part2_service import Part2DataService  # noqa: E402

YEAR = '2026'
ORIGIN = 'Wallisellen'
OUT_PDF = FIG_DIR / 'part2_wallisellen_reachability_2026.pdf'
OUT_SVG = FIG_DIR / 'part2_wallisellen_reachability_2026.svg'

THRESHOLDS = [30, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 720]
ISOCHRONE_COLORS = [
    '#ff3b30', '#ff6d1f', '#ff9f0a', '#ffd60a', '#9ad200', '#34c759',
    '#16c3a5', '#1fb7d9', '#0a84ff', '#4361ee', '#6f52ed', '#a259ff',
    '#d946ef', '#ff2d55',
]
OVERFLOW_COLOR = '#d7dbe0'

APP_WIDTH = 1120.0
APP_HEIGHT = 650.0
APP_PADDING = 52.0
PANEL_GAP = 18.0
SCALE = 0.43
TITLE_H = 54.0
PANEL_LABEL_H = 28.0
LEGEND_H = 70.0
PAGE_MARGIN_X = 18.0
PAGE_MARGIN_Y = 16.0

PAGE_W = PAGE_MARGIN_X * 2 + SCALE * (APP_WIDTH * 2 + PANEL_GAP)
PAGE_H = PAGE_MARGIN_Y * 2 + TITLE_H + PANEL_LABEL_H + SCALE * APP_HEIGHT + LEGEND_H

# The original app-style raster used a 66 x 46 grid. Doubling both
# dimensions makes each interpolated cell one quarter of the previous area.
RASTER_COLUMNS = 132
RASTER_ROWS = 92
CONTOUR_COLUMNS = 96
CONTOUR_ROWS = 68


@dataclass
class Bounds:
    width: float = APP_WIDTH
    height: float = APP_HEIGHT
    padding: float = APP_PADDING
    projection: str = 'lv95'
    minX: float = 0.0
    maxX: float = 0.0
    minY: float = 0.0
    maxY: float = 0.0


def load_reachability() -> dict:
    service = Part2DataService(ROOT)
    payload = service.reachability(origin=ORIGIN, optimized_version='optimized_step2')
    year_payload = payload['years'][YEAR]
    if not year_payload.get('ok'):
        raise RuntimeError(year_payload.get('error', 'reachability payload failed'))
    return year_payload


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wgs84_to_lv95(lat: float, lon: float) -> tuple[float, float]:
    lat_seconds = float(lat) * 3600.0
    lon_seconds = float(lon) * 3600.0
    lat_aux = (lat_seconds - 169028.66) / 10000.0
    lon_aux = (lon_seconds - 26782.5) / 10000.0
    east = (
        2600072.37
        + 211455.93 * lon_aux
        - 10938.51 * lon_aux * lat_aux
        - 0.36 * lon_aux * lat_aux * lat_aux
        - 44.54 * lon_aux * lon_aux * lon_aux
    )
    north = (
        1200147.07
        + 308807.95 * lat_aux
        + 3745.25 * lon_aux * lon_aux
        + 76.63 * lat_aux * lat_aux
        - 194.56 * lon_aux * lon_aux * lat_aux
        + 119.79 * lat_aux * lat_aux * lat_aux
    )
    return east, north


def lv95_to_approx_wgs84(east: float, north: float) -> tuple[float, float]:
    lat_scale = 111320.0
    lon_scale = 74800.0
    return (
        46.9511 + (float(north) - 1200147.07) / lat_scale,
        7.4386 + (float(east) - 2600072.37) / lon_scale,
    )


def projection_bounds(boundaries: dict) -> Bounds:
    raw = boundaries['bounds']
    return Bounds(
        minX=float(raw['minX']) - 9000.0,
        maxX=float(raw['maxX']) + 9000.0,
        minY=float(raw['minY']) - 9000.0,
        maxY=float(raw['maxY']) + 9000.0,
    )


def project_planar(bounds: Bounds, east: float, north: float) -> tuple[float, float]:
    usable_w = bounds.width - bounds.padding * 2.0
    usable_h = bounds.height - bounds.padding * 2.0
    range_x = max(bounds.maxX - bounds.minX, 0.0001)
    range_y = max(bounds.maxY - bounds.minY, 0.0001)
    scale = min(usable_w / range_x, usable_h / range_y)
    offset_x = (bounds.width - range_x * scale) / 2.0
    offset_y = (bounds.height - range_y * scale) / 2.0
    x = offset_x + (float(east) - bounds.minX) * scale
    y = offset_y + (bounds.maxY - float(north)) * scale
    return x, y


def project_station(bounds: Bounds, station: dict) -> tuple[float, float]:
    east, north = wgs84_to_lv95(station['lat'], station['lon'])
    return project_planar(bounds, east, north)


def unproject_point(bounds: Bounds, x: float, y: float) -> tuple[float, float]:
    usable_w = bounds.width - bounds.padding * 2.0
    usable_h = bounds.height - bounds.padding * 2.0
    range_x = max(bounds.maxX - bounds.minX, 0.0001)
    range_y = max(bounds.maxY - bounds.minY, 0.0001)
    scale = min(usable_w / range_x, usable_h / range_y)
    offset_x = (bounds.width - range_x * scale) / 2.0
    offset_y = (bounds.height - range_y * scale) / 2.0
    east = bounds.minX + (float(x) - offset_x) / scale
    north = bounds.maxY - (float(y) - offset_y) / scale
    return east, north


def haversine_km(a: dict | tuple[float, float], b: dict | tuple[float, float]) -> float:
    if isinstance(a, dict):
        lat1, lon1 = float(a['lat']), float(a['lon'])
    else:
        lat1, lon1 = float(a[0]), float(a[1])
    if isinstance(b, dict):
        lat2, lon2 = float(b['lat']), float(b['lon'])
    else:
        lat2, lon2 = float(b[0]), float(b[1])
    earth_radius_km = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return earth_radius_km * 2 * math.atan2(math.sqrt(h), math.sqrt(max(0.0, 1.0 - h)))


def interpolate_value(point_latlon: tuple[float, float], stations: list[dict]) -> float | None:
    sigma_km = 12.0
    radius_km = 25.0
    nearest_km = math.inf
    weighted_sum = 0.0
    weight_total = 0.0
    for station in stations:
        km = haversine_km(point_latlon, station)
        nearest_km = min(nearest_km, km)
        if km > radius_km:
            continue
        weight = math.exp(-(km * km) / (2.0 * sigma_km * sigma_km))
        weighted_sum += float(station['fastestTimeMin']) * weight
        weight_total += weight
    if nearest_km > radius_km or weight_total <= 0:
        return None
    return weighted_sum / weight_total


def is_within_station_coverage(point_latlon: tuple[float, float], stations: list[dict], radius_km: float = 25.0) -> bool:
    return any(haversine_km(point_latlon, station) <= radius_km for station in stations)


def bucket_index(minutes: float | None) -> int | None:
    try:
        numeric = float(minutes)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    for idx, threshold in enumerate(THRESHOLDS):
        if numeric <= threshold:
            return idx
    return len(THRESHOLDS)


def bucket_color(idx: int | None) -> str:
    if idx is None:
        return OVERFLOW_COLOR
    if idx >= len(THRESHOLDS):
        return OVERFLOW_COLOR
    return ISOCHRONE_COLORS[idx]


def format_threshold_label(minutes: int) -> str:
    if minutes < 60:
        return f'{minutes} min'
    hours = minutes // 60
    mins = minutes % 60
    return f'{hours} h {mins} min' if mins else f'{hours} h'


def legend_label(idx: int) -> str:
    if idx < len(THRESHOLDS):
        threshold = THRESHOLDS[idx]
        if idx == 0:
            return '<=30 min'
        previous = THRESHOLDS[idx - 1]
        if threshold <= 180:
            return f'{previous + 1}-{threshold} min'
        return f'{format_threshold_label(previous + 1)}-{format_threshold_label(threshold)}'
    return '>12 h'


def finite_minutes(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def panel_stations(reachability: dict, geom: dict, scenario: str) -> list[dict]:
    station_geo = geom.get('stationMunicipalities', {})
    stations: list[dict] = []
    for row in reachability.get('stations', []):
        name = row.get('station')
        meta = station_geo.get(name)
        if not meta:
            continue
        baseline_min = finite_minutes((row.get('baseline') or {}).get('minutes'))
        optimized_min = finite_minutes((row.get('optimized') or {}).get('minutes'))
        fastest = baseline_min if scenario == 'baseline' else optimized_min
        if fastest is None:
            continue
        idx = bucket_index(fastest)
        stations.append({
            'station': name,
            'lat': float(meta['lat']),
            'lon': float(meta['lon']),
            'fastestTimeMin': fastest,
            'baselineMin': baseline_min,
            'optimizedMin': optimized_min,
            'optimizedFaster': bool(row.get('optimizedFaster')),
            'bucketIndex': idx,
            'color': bucket_color(idx),
        })
    return stations


def country_polygons(boundaries: dict) -> list[list[tuple[float, float]]]:
    return [[(float(x), float(y)) for x, y in poly] for poly in boundaries.get('country', {}).get('polygons', [])]


def canton_polygons(boundaries: dict) -> list[list[tuple[float, float]]]:
    polys = []
    for canton in boundaries.get('cantons', []):
        for poly in canton.get('polygons', []) or []:
            if len(poly) >= 3:
                polys.append([(float(x), float(y)) for x, y in poly])
    return polys


def projected_polygons(bounds: Bounds, polygons: Iterable[Iterable[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    return [[project_planar(bounds, east, north) for east, north in poly] for poly in polygons]


def svg_polygon_path(polygons: list[list[tuple[float, float]]]) -> str:
    parts: list[str] = []
    for poly in polygons:
        if not poly:
            continue
        first, *rest = poly
        line = ' '.join(f'L {x:.2f} {y:.2f}' for x, y in rest)
        parts.append(f'M {first[0]:.2f} {first[1]:.2f} {line} Z')
    return ' '.join(parts)


def build_path(c: canvas.Canvas, polygons: list[list[tuple[float, float]]]):
    path = c.beginPath()
    for poly in polygons:
        if not poly:
            continue
        first, *rest = poly
        path.moveTo(first[0], first[1])
        for x, y in rest:
            path.lineTo(x, y)
        path.close()
    return path


def hex_color(hex_value: str):
    return colors.HexColor(hex_value)


def set_rgba(c: canvas.Canvas, *, fill: str | None = None, stroke: str | None = None, fill_alpha: float = 1.0, stroke_alpha: float = 1.0):
    if fill is not None:
        c.setFillColor(hex_color(fill))
        c.setFillAlpha(fill_alpha)
    if stroke is not None:
        c.setStrokeColor(hex_color(stroke))
        c.setStrokeAlpha(stroke_alpha)


def band_rects(bounds: Bounds, stations: list[dict]) -> list[tuple[float, float, float, float, str]]:
    rects: list[tuple[float, float, float, float, str]] = []
    if len(stations) < 3:
        return rects
    columns = RASTER_COLUMNS
    rows = RASTER_ROWS
    x0, x1 = bounds.padding, bounds.width - bounds.padding
    y0, y1 = bounds.padding, bounds.height - bounds.padding
    cw = (x1 - x0) / columns
    ch = (y1 - y0) / rows
    for row in range(rows):
        for col in range(columns):
            cx = x0 + (col + 0.5) * cw
            cy = y0 + (row + 0.5) * ch
            east, north = unproject_point(bounds, cx, cy)
            point = lv95_to_approx_wgs84(east, north)
            value = interpolate_value(point, stations)
            idx = bucket_index(value)
            if idx is None:
                continue
            rects.append((x0 + col * cw, y0 + row * ch, cw + 0.35, ch + 0.35, bucket_color(idx)))
    return rects


def coverage_mask_rects(bounds: Bounds, stations: list[dict]) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    columns = RASTER_COLUMNS
    rows = RASTER_ROWS
    x0, x1 = bounds.padding, bounds.width - bounds.padding
    y0, y1 = bounds.padding, bounds.height - bounds.padding
    cw = (x1 - x0) / columns
    ch = (y1 - y0) / rows
    for row in range(rows):
        for col in range(columns):
            cx = x0 + (col + 0.5) * cw
            cy = y0 + (row + 0.5) * ch
            east, north = unproject_point(bounds, cx, cy)
            point = lv95_to_approx_wgs84(east, north)
            if is_within_station_coverage(point, stations, 25.0):
                continue
            rects.append((x0 + col * cw, y0 + row * ch, cw + 0.35, ch + 0.35))
    return rects


def contour_segments(bounds: Bounds, stations: list[dict]) -> list[tuple[float, float, float, float]]:
    if len(stations) < 3:
        return []
    columns = CONTOUR_COLUMNS
    rows = CONTOUR_ROWS
    x0, x1 = bounds.padding, bounds.width - bounds.padding
    y0, y1 = bounds.padding, bounds.height - bounds.padding
    dx = (x1 - x0) / (columns - 1)
    dy = (y1 - y0) / (rows - 1)
    grid: list[list[dict]] = []
    for row in range(rows):
        values = []
        for col in range(columns):
            x = x0 + col * dx
            y = y0 + row * dy
            east, north = unproject_point(bounds, x, y)
            point = lv95_to_approx_wgs84(east, north)
            values.append({'x': x, 'y': y, 'value': interpolate_value(point, stations)})
        grid.append(values)

    def edge_point(a: dict, b: dict, threshold: float) -> tuple[float, float]:
        span = float(b['value']) - float(a['value'])
        ratio = 0.5 if abs(span) < 1e-9 else (threshold - float(a['value'])) / span
        ratio = clamp(ratio, 0.0, 1.0)
        return (a['x'] + (b['x'] - a['x']) * ratio, a['y'] + (b['y'] - a['y']) * ratio)

    edge_pairs_by_case = {
        1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
        5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)],
        9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(1, 2)], 12: [(1, 3)],
        13: [(0, 1)], 14: [(3, 0)],
    }
    segments: list[tuple[float, float, float, float]] = []
    for threshold in THRESHOLDS:
        if threshold % 30 != 0:
            continue
        for row in range(rows - 1):
            for col in range(columns - 1):
                corners = [grid[row][col], grid[row][col + 1], grid[row + 1][col + 1], grid[row + 1][col]]
                if any(corner['value'] is None or not math.isfinite(float(corner['value'])) for corner in corners):
                    continue
                mask = 0
                for idx, corner in enumerate(corners):
                    if float(corner['value']) <= threshold:
                        mask |= 1 << idx
                pairs = edge_pairs_by_case.get(mask)
                if not pairs:
                    continue
                edge_points = [
                    edge_point(corners[0], corners[1], threshold),
                    edge_point(corners[1], corners[2], threshold),
                    edge_point(corners[2], corners[3], threshold),
                    edge_point(corners[3], corners[0], threshold),
                ]
                for first_edge, second_edge in pairs:
                    first = edge_points[first_edge]
                    second = edge_points[second_edge]
                    segments.append((first[0], first[1], second[0], second[1]))
    return segments


def draw_poly_path(c: canvas.Canvas, projected: list[list[tuple[float, float]]], fill: str | None, stroke: str | None, sw: float, fill_alpha: float, stroke_alpha: float):
    c.saveState()
    set_rgba(c, fill=fill, stroke=stroke, fill_alpha=fill_alpha, stroke_alpha=stroke_alpha)
    if sw:
        c.setLineWidth(sw)
    path = build_path(c, projected)
    c.drawPath(path, stroke=1 if stroke else 0, fill=1 if fill else 0)
    c.restoreState()


def draw_panel_pdf(c: canvas.Canvas, bounds: Bounds, country_proj, canton_proj, stations: list[dict], *, show_faster: bool):
    c.saveState()
    draw_poly_path(c, country_proj, fill='#ffffff', stroke='#111315', sw=2.1, fill_alpha=0.64, stroke_alpha=0.24)
    draw_poly_path(c, canton_proj, fill=None, stroke='#111315', sw=1.0, fill_alpha=0.0, stroke_alpha=0.08)

    clip_path = build_path(c, country_proj)
    c.saveState()
    c.clipPath(clip_path, stroke=0, fill=0)

    c.saveState()
    c.setFillAlpha(0.14)
    for x, y, w, h, fill in band_rects(bounds, stations):
        c.setFillColor(hex_color(fill))
        c.rect(x, y, w, h, stroke=0, fill=1)
    c.restoreState()

    c.saveState()
    c.setFillAlpha(1.0)
    c.setFillColor(hex_color('#d9dde3'))
    for x, y, w, h in coverage_mask_rects(bounds, stations):
        c.rect(x, y, w, h, stroke=0, fill=1)
    c.restoreState()

    c.saveState()
    c.setStrokeColor(colors.Color(120/255, 31/255, 31/255, alpha=0.28))
    c.setLineWidth(1.05)
    c.setDash(4, 7)
    for x1, y1, x2, y2 in contour_segments(bounds, stations):
        c.line(x1, y1, x2, y2)
    c.restoreState()
    c.restoreState()

    draw_poly_path(c, country_proj, fill=None, stroke='#111315', sw=1.25, fill_alpha=0.0, stroke_alpha=0.20)
    draw_poly_path(c, canton_proj, fill=None, stroke='#111315', sw=0.80, fill_alpha=0.0, stroke_alpha=0.10)

    for station in stations:
        x, y = project_station(bounds, station)
        c.saveState()
        c.setFillColor(hex_color(station['color']))
        c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.18))
        c.setLineWidth(1.1)
        c.circle(x, y, 5.2, stroke=1, fill=1)
        if show_faster and station.get('optimizedFaster'):
            c.setFillColor(colors.white)
            c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.82))
            c.setLineWidth(0.7)
            c.circle(x, y, 2.55, stroke=1, fill=1)
        c.restoreState()
    c.restoreState()


def draw_text(c: canvas.Canvas, text: str, x: float, y: float, font: str = 'Helvetica', size: float = 10, color=colors.black):
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, y, text)
    c.restoreState()


def draw_legend_pdf(c: canvas.Canvas, x: float, y: float):
    c.saveState()
    c.setFont('Helvetica', 7.0)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.68))
    c.drawString(x, y + 42, 'Fastest travel time from Wallisellen')
    x0 = x
    yy = y + 24
    item_gap = 49
    for idx in range(len(THRESHOLDS) + 1):
        if idx == 7:
            yy -= 21
            x0 = x
        c.setFillColor(hex_color(bucket_color(idx)))
        c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.12))
        c.setLineWidth(0.5)
        c.roundRect(x0, yy, 9, 9, 4.5, stroke=1, fill=1)
        c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.70))
        c.drawString(x0 + 13, yy + 1.2, legend_label(idx).replace(' min', 'm').replace(' h', 'h'))
        x0 += item_gap
    x_extra = x + 7 * item_gap + 10
    y_extra = y + 3
    c.setFillColor(hex_color('#d9dde3'))
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.12))
    c.roundRect(x_extra, y_extra, 9, 9, 4.5, stroke=1, fill=1)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.70))
    c.drawString(x_extra + 13, y_extra + 1.2, 'not interpolated')
    x_extra += 112
    c.setFillColor(hex_color('#0a84ff'))
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.18))
    c.circle(x_extra + 4.5, y_extra + 4.5, 4.5, stroke=1, fill=1)
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.82))
    c.circle(x_extra + 4.5, y_extra + 4.5, 2.15, stroke=1, fill=1)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.70))
    c.drawString(x_extra + 13, y_extra + 1.2, 'faster after optimization')
    c.restoreState()


def create_pdf(reachability: dict, boundaries: dict, geom: dict):
    bounds = projection_bounds(boundaries)
    country_proj = projected_polygons(bounds, country_polygons(boundaries))
    canton_proj = projected_polygons(bounds, canton_polygons(boundaries))
    baseline = panel_stations(reachability, geom, 'baseline')
    optimized = panel_stations(reachability, geom, 'optimized')

    c = canvas.Canvas(str(OUT_PDF), pagesize=(PAGE_W, PAGE_H))
    c.setTitle('Wallisellen reachability before and after optimization')
    c.setFillColor(hex_color('#fbf8f1'))
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    draw_text(c, 'Reachability from Wallisellen: baseline and post-optimization', PAGE_MARGIN_X, PAGE_H - PAGE_MARGIN_Y - 17, 'Helvetica-Bold', 15.0, colors.Color(17/255, 19/255, 21/255))

    label_y = PAGE_H - PAGE_MARGIN_Y - TITLE_H
    map_bottom = PAGE_MARGIN_Y + LEGEND_H
    panel1_x = PAGE_MARGIN_X
    panel2_x = PAGE_MARGIN_X + SCALE * (APP_WIDTH + PANEL_GAP)
    map_w = SCALE * APP_WIDTH
    draw_text(c, 'Baseline', panel1_x, label_y + 4, 'Helvetica-Bold', 10.8, colors.Color(17/255, 19/255, 21/255))
    draw_text(c, 'Post-optimization', panel2_x, label_y + 4, 'Helvetica-Bold', 10.8, colors.Color(17/255, 19/255, 21/255))

    for x, stations, faster in [(panel1_x, baseline, False), (panel2_x, optimized, True)]:
        c.saveState()
        c.translate(x, map_bottom + SCALE * APP_HEIGHT)
        c.scale(SCALE, -SCALE)
        draw_panel_pdf(c, bounds, country_proj, canton_proj, stations, show_faster=faster)
        c.restoreState()
        c.saveState()
        c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.08))
        c.setLineWidth(0.75)
        c.roundRect(x, map_bottom, map_w, SCALE * APP_HEIGHT, 8, stroke=1, fill=0)
        c.restoreState()

    draw_legend_pdf(c, PAGE_MARGIN_X, PAGE_MARGIN_Y + 6)
    c.showPage()
    c.save()


def svg_panel(bounds: Bounds, country_path: str, canton_paths: list[str], stations: list[dict], panel_id: str, show_faster: bool) -> str:
    bands = []
    for x, y, w, h, fill in band_rects(bounds, stations):
        bands.append(f'<rect class="part2-reachability-band" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"/>')
    masks = [f'<rect class="part2-uncovered-cell" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>' for x, y, w, h in coverage_mask_rects(bounds, stations)]
    contour_parts = [f'M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}' for x1, y1, x2, y2 in contour_segments(bounds, stations)]
    contour_el = f'<path class="part2-reachability-contour" d="{" ".join(contour_parts)}"/>' if contour_parts else ''
    dots = []
    for station in stations:
        x, y = project_station(bounds, station)
        inner = '<circle class="part2-reachability-faster-dot" r="2.55" fill="#ffffff"/>' if show_faster and station.get('optimizedFaster') else ''
        dots.append(f'<g transform="translate({x:.2f},{y:.2f})"><circle class="part2-reachability-dot" r="5.2" fill="{station["color"]}"/>{inner}</g>')
    return f'''
      <defs><clipPath id="clip-{panel_id}"><path d="{country_path}"/></clipPath></defs>
      <rect x="0" y="0" width="{bounds.width}" height="{bounds.height}" fill="rgba(255,255,255,0.18)"/>
      <path class="part2-country-fill" d="{country_path}"/>
      {''.join(f'<path class="part2-canton-line" d="{p}"/>' for p in canton_paths)}
      <g clip-path="url(#clip-{panel_id})">
        {''.join(bands)}
        {''.join(masks)}
        {contour_el}
      </g>
      {''.join(dots)}
    '''


def create_svg(reachability: dict, boundaries: dict, geom: dict):
    bounds = projection_bounds(boundaries)
    country_proj = projected_polygons(bounds, country_polygons(boundaries))
    canton_proj = projected_polygons(bounds, canton_polygons(boundaries))
    country_path = svg_polygon_path(country_proj)
    canton_paths = [svg_polygon_path([poly]) for poly in canton_proj]
    baseline = panel_stations(reachability, geom, 'baseline')
    optimized = panel_stations(reachability, geom, 'optimized')

    view_w = APP_WIDTH * 2 + PANEL_GAP
    view_h = APP_HEIGHT + 128
    panel2_x = APP_WIDTH + PANEL_GAP
    swatches = []
    x = 0
    y = APP_HEIGHT + 80
    for idx in range(len(THRESHOLDS) + 1):
        if idx == 7:
            y += 22
            x = 0
        swatches.append(f'<g transform="translate({x},{y})"><circle r="5" fill="{bucket_color(idx)}" stroke="rgba(17,19,21,0.12)"/><text x="11" y="4">{html.escape(legend_label(idx))}</text></g>')
        x += 96
    swatches.append(f'<g transform="translate(770,{APP_HEIGHT + 102})"><circle r="5.2" fill="#0a84ff" stroke="rgba(17,19,21,0.18)"/><circle r="2.55" fill="#fff" stroke="rgba(17,19,21,0.82)"/><text x="12" y="4">faster after optimization</text></g>')
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.0f} {view_h:.0f}" width="{view_w:.0f}" height="{view_h:.0f}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Avenir Next", Helvetica, Arial, sans-serif; fill: #2a2d33; font-size: 15px; font-weight: 650; }}
    .title {{ font-size: 28px; font-weight: 800; }}
    .panel-title {{ font-size: 21px; font-weight: 800; }}
    .part2-country-fill {{ fill: rgba(255,255,255,0.64); stroke: rgba(17,19,21,0.24); stroke-width: 2.1; stroke-linejoin: round; vector-effect: non-scaling-stroke; }}
    .part2-canton-line {{ fill: none; stroke: rgba(17,19,21,0.08); stroke-width: 1; stroke-linejoin: round; vector-effect: non-scaling-stroke; }}
    .part2-reachability-band {{ stroke: none; opacity: 0.14; }}
    .part2-reachability-contour {{ fill: none; stroke: rgba(120,31,31,0.28); stroke-width: 1.05; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 4 7; vector-effect: non-scaling-stroke; }}
    .part2-uncovered-cell {{ fill: #d9dde3; stroke: none; }}
    .part2-reachability-dot {{ stroke: rgba(17,19,21,0.18); stroke-width: 1.1; vector-effect: non-scaling-stroke; }}
    .part2-reachability-faster-dot {{ stroke: rgba(17,19,21,0.82); stroke-width: 0.7; vector-effect: non-scaling-stroke; }}
  </style>
  <rect width="100%" height="100%" fill="#fbf8f1"/>
  <text class="title" x="0" y="30">Reachability from Wallisellen: baseline and post-optimization</text>
  <text class="panel-title" x="0" y="62">Baseline</text>
  <text class="panel-title" x="{panel2_x}" y="62">Post-optimization</text>
  <g transform="translate(0,72)">{svg_panel(bounds, country_path, canton_paths, baseline, 'baseline', False)}</g>
  <g transform="translate({panel2_x},72)">{svg_panel(bounds, country_path, canton_paths, optimized, 'post', True)}</g>
  <g transform="translate(0,0)">{''.join(swatches)}</g>
</svg>
'''
    OUT_SVG.write_text(svg, encoding='utf-8')


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reachability = load_reachability()
    boundaries = load_json(BOUNDARIES_PATH)
    geom = load_json(GEOM_PATH)
    create_pdf(reachability, boundaries, geom)
    create_svg(reachability, boundaries, geom)
    print(f'Wrote {OUT_PDF}')
    print(f'Wrote {OUT_SVG}')
    print(f"Stations: {reachability.get('stationCount')} total, {reachability.get('reachableStations')} reachable, {reachability.get('improvedStations')} faster after optimization")


if __name__ == '__main__':
    main()
