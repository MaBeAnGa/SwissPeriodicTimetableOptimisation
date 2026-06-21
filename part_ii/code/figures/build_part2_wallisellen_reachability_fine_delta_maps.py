from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path.home() / 'iCloud Drive (Archive)' / 'Documents' / 'Documents - Matthias’s 16" MacBook Pro (246)' / 'Paper Pythons' / 'Master Thesis'
FIG_DIR = ROOT / 'figures' / 'part-II'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FIG_DIR))

import build_part2_wallisellen_reachability_maps as base  # noqa: E402
from part2_service import Part2DataService  # noqa: E402

YEAR = '2026'
ORIGIN = 'Wallisellen'
OUT_FINE_PDF = FIG_DIR / 'part2_wallisellen_reachability_2026.pdf'
OUT_DELTA_PDF = FIG_DIR / 'part2_wallisellen_reachability_delta_2026.pdf'

# Original app-style raster: 66 x 46. Multiplying rows and columns by 8
# makes each raster cell 1/64 of the original area.
RASTER_COLUMNS = 66 * 8
RASTER_ROWS = 46 * 8
CONTOUR_COLUMNS = 48 * 4
CONTOUR_ROWS = 34 * 4

SIGMA_KM = 12.0
RADIUS_KM = 25.0
EARTH_RADIUS_KM = 6371.0088
CHUNK_SIZE = 2048
TIME_ALPHA = int(round(255 * 0.14))
NOT_INTERPOLATED_RGBA = (217, 221, 227, 255)
WHITE_RGBA = (255, 255, 255, 255)
DELTA_BLUE = np.array([8, 81, 156], dtype=np.float32)
DELTA_MIN_BLUE = np.array([237, 246, 255], dtype=np.float32)


def load_reachability() -> dict:
    service = Part2DataService(ROOT)
    payload = service.reachability(origin=ORIGIN, optimized_version='optimized_step2')
    year_payload = payload['years'][YEAR]
    if not year_payload.get('ok'):
        raise RuntimeError(year_payload.get('error', 'reachability payload failed'))
    return year_payload


def hex_to_rgb(hex_value: str) -> tuple[int, int, int]:
    h = hex_value.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def grid_latlon(bounds: base.Bounds) -> tuple[np.ndarray, np.ndarray]:
    x0, x1 = bounds.padding, bounds.width - bounds.padding
    y0, y1 = bounds.padding, bounds.height - bounds.padding
    cw = (x1 - x0) / RASTER_COLUMNS
    ch = (y1 - y0) / RASTER_ROWS
    xs = x0 + (np.arange(RASTER_COLUMNS, dtype=np.float64) + 0.5) * cw
    ys = y0 + (np.arange(RASTER_ROWS, dtype=np.float64) + 0.5) * ch
    xx, yy = np.meshgrid(xs, ys)

    usable_w = bounds.width - bounds.padding * 2.0
    usable_h = bounds.height - bounds.padding * 2.0
    range_x = max(bounds.maxX - bounds.minX, 0.0001)
    range_y = max(bounds.maxY - bounds.minY, 0.0001)
    scale = min(usable_w / range_x, usable_h / range_y)
    offset_x = (bounds.width - range_x * scale) / 2.0
    offset_y = (bounds.height - range_y * scale) / 2.0

    east = bounds.minX + (xx - offset_x) / scale
    north = bounds.maxY - (yy - offset_y) / scale
    lat = 46.9511 + (north - 1200147.07) / 111320.0
    lon = 7.4386 + (east - 2600072.37) / 74800.0
    return lat.reshape(-1), lon.reshape(-1)


def station_arrays(stations: list[dict], value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats = []
    lons = []
    vals = []
    for station in stations:
        value = station.get(value_key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        lats.append(float(station['lat']))
        lons.append(float(station['lon']))
        vals.append(numeric)
    return (
        np.deg2rad(np.array(lats, dtype=np.float64)),
        np.deg2rad(np.array(lons, dtype=np.float64)),
        np.array(vals, dtype=np.float64),
    )


def interpolate_grid(lat_deg: np.ndarray, lon_deg: np.ndarray, stations: list[dict], value_key: str) -> tuple[np.ndarray, np.ndarray]:
    station_lat, station_lon, station_values = station_arrays(stations, value_key)
    point_lat = np.deg2rad(lat_deg.astype(np.float64, copy=False))
    point_lon = np.deg2rad(lon_deg.astype(np.float64, copy=False))
    out = np.full(point_lat.shape[0], np.nan, dtype=np.float64)
    covered = np.zeros(point_lat.shape[0], dtype=bool)

    for start in range(0, point_lat.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, point_lat.shape[0])
        lat_chunk = point_lat[start:end, None]
        lon_chunk = point_lon[start:end, None]
        dlat = station_lat[None, :] - lat_chunk
        dlon = station_lon[None, :] - lon_chunk
        h = np.sin(dlat / 2.0) ** 2 + np.cos(lat_chunk) * np.cos(station_lat[None, :]) * np.sin(dlon / 2.0) ** 2
        km = EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(h), np.sqrt(np.maximum(0.0, 1.0 - h)))
        within = km <= RADIUS_KM
        nearest = np.min(km, axis=1)
        weights = np.exp(-(km * km) / (2.0 * SIGMA_KM * SIGMA_KM)) * within
        totals = np.sum(weights, axis=1)
        valid = (nearest <= RADIUS_KM) & (totals > 0)
        covered[start:end] = valid
        if np.any(valid):
            out[start:end][valid] = (weights[valid] @ station_values) / totals[valid]
    return out.reshape(RASTER_ROWS, RASTER_COLUMNS), covered.reshape(RASTER_ROWS, RASTER_COLUMNS)


def time_raster(values: np.ndarray, covered: np.ndarray) -> Image.Image:
    rgba = np.zeros((RASTER_ROWS, RASTER_COLUMNS, 4), dtype=np.uint8)
    rgba[:, :, :] = (0, 0, 0, 0)
    rgba[~covered] = NOT_INTERPOLATED_RGBA
    finite = np.isfinite(values) & covered
    for idx in range(len(base.THRESHOLDS) + 1):
        if idx < len(base.THRESHOLDS):
            lower = -np.inf if idx == 0 else base.THRESHOLDS[idx - 1]
            upper = base.THRESHOLDS[idx]
            mask = finite & (values > lower) & (values <= upper)
        else:
            mask = finite & (values > base.THRESHOLDS[-1])
        if not np.any(mask):
            continue
        r, g, b = hex_to_rgb(base.bucket_color(idx))
        rgba[mask] = (r, g, b, TIME_ALPHA)
    return Image.fromarray(rgba, mode='RGBA')


def delta_raster(delta_values: np.ndarray, covered: np.ndarray, delta_max: float) -> Image.Image:
    rgba = np.zeros((RASTER_ROWS, RASTER_COLUMNS, 4), dtype=np.uint8)
    rgba[:, :, :] = WHITE_RGBA
    finite = np.isfinite(delta_values) & covered
    positive = finite & (delta_values > 0)
    if np.any(positive):
        t = np.clip(delta_values[positive] / max(delta_max, 1e-9), 0.0, 1.0).astype(np.float32)
        # Keep very small positive deltas visible while still letting white mean no gain.
        t = 0.18 + 0.82 * t
        rgb = (DELTA_MIN_BLUE[None, :] * (1.0 - t[:, None]) + DELTA_BLUE[None, :] * t[:, None]).clip(0, 255).astype(np.uint8)
        rgba[positive, :3] = rgb
        rgba[positive, 3] = 255
    return Image.fromarray(rgba, mode='RGBA')


def page_polygons(polygons: list[list[tuple[float, float]]], panel_x: float, map_bottom: float) -> list[list[tuple[float, float]]]:
    return [[(panel_x + base.SCALE * x, map_bottom + base.SCALE * (base.APP_HEIGHT - y)) for x, y in poly] for poly in polygons]


def page_point(x: float, y: float, panel_x: float, map_bottom: float) -> tuple[float, float]:
    return panel_x + base.SCALE * x, map_bottom + base.SCALE * (base.APP_HEIGHT - y)


def draw_raster_image(c: canvas.Canvas, img: Image.Image, bounds: base.Bounds, panel_x: float, map_bottom: float):
    x0, x1 = bounds.padding, bounds.width - bounds.padding
    y0, y1 = bounds.padding, bounds.height - bounds.padding
    px = panel_x + base.SCALE * x0
    py = map_bottom + base.SCALE * (base.APP_HEIGHT - y1)
    pw = base.SCALE * (x1 - x0)
    ph = base.SCALE * (y1 - y0)
    c.drawImage(ImageReader(img), px, py, width=pw, height=ph, mask='auto')


def draw_contours(c: canvas.Canvas, bounds: base.Bounds, stations: list[dict], panel_x: float, map_bottom: float):
    old_cols, old_rows = base.CONTOUR_COLUMNS, base.CONTOUR_ROWS
    base.CONTOUR_COLUMNS, base.CONTOUR_ROWS = CONTOUR_COLUMNS, CONTOUR_ROWS
    try:
        segments = base.contour_segments(bounds, stations)
    finally:
        base.CONTOUR_COLUMNS, base.CONTOUR_ROWS = old_cols, old_rows
    c.saveState()
    c.setStrokeColor(colors.Color(120/255, 31/255, 31/255, alpha=0.25))
    c.setLineWidth(0.45)
    c.setDash(2.0, 4.0)
    for x1, y1, x2, y2 in segments:
        px1, py1 = page_point(x1, y1, panel_x, map_bottom)
        px2, py2 = page_point(x2, y2, panel_x, map_bottom)
        c.line(px1, py1, px2, py2)
    c.restoreState()


def draw_panel(c: canvas.Canvas, bounds: base.Bounds, country_proj, canton_proj, stations: list[dict], raster: Image.Image, panel_x: float, map_bottom: float, *, show_faster: bool, show_contours: bool, draw_stations: bool = True):
    country_page = page_polygons(country_proj, panel_x, map_bottom)
    canton_page = page_polygons(canton_proj, panel_x, map_bottom)
    base.draw_poly_path(c, country_page, fill='#ffffff', stroke='#111315', sw=0.85, fill_alpha=1.0, stroke_alpha=0.24)
    base.draw_poly_path(c, canton_page, fill=None, stroke='#111315', sw=0.35, fill_alpha=0.0, stroke_alpha=0.10)

    clip_path = base.build_path(c, country_page)
    c.saveState()
    c.clipPath(clip_path, stroke=0, fill=0)
    draw_raster_image(c, raster, bounds, panel_x, map_bottom)
    if show_contours:
        draw_contours(c, bounds, stations, panel_x, map_bottom)
    c.restoreState()

    base.draw_poly_path(c, country_page, fill=None, stroke='#111315', sw=0.60, fill_alpha=0.0, stroke_alpha=0.24)
    base.draw_poly_path(c, canton_page, fill=None, stroke='#111315', sw=0.28, fill_alpha=0.0, stroke_alpha=0.14)

    if draw_stations:
        for station in stations:
            x, y = base.project_station(bounds, station)
            px, py = page_point(x, y, panel_x, map_bottom)
            c.saveState()
            c.setFillColor(base.hex_color(station['color']))
            c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.18))
            c.setLineWidth(0.45)
            c.circle(px, py, base.SCALE * 5.2, stroke=1, fill=1)
            if show_faster and station.get('optimizedFaster'):
                c.setFillColor(colors.white)
                c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.82))
                c.setLineWidth(0.35)
                c.circle(px, py, base.SCALE * 2.55, stroke=1, fill=1)
            c.restoreState()

    c.saveState()
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.08))
    c.setLineWidth(0.75)
    c.roundRect(panel_x, map_bottom, base.SCALE * base.APP_WIDTH, base.SCALE * base.APP_HEIGHT, 8, stroke=1, fill=0)
    c.restoreState()


def draw_time_legend(c: canvas.Canvas, x: float, y: float, title: str):
    c.saveState()
    c.setFont('Helvetica', 6.6)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.72))
    c.drawString(x, y + 42, title)
    x0 = x
    yy = y + 24
    item_gap = 45
    for idx in range(len(base.THRESHOLDS) + 1):
        if idx == 7:
            yy -= 21
            x0 = x
        c.setFillColor(base.hex_color(base.bucket_color(idx)))
        c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.12))
        c.roundRect(x0, yy, 8.5, 8.5, 4.2, stroke=1, fill=1)
        c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.70))
        c.drawString(x0 + 12, yy + 1.1, base.legend_label(idx).replace(' min', 'm').replace(' h', 'h'))
        x0 += item_gap
    c.restoreState()


def draw_faster_dot_legend(c: canvas.Canvas, x: float, y: float):
    c.saveState()
    c.setFont('Helvetica', 6.6)
    c.setFillColor(base.hex_color('#0a84ff'))
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.18))
    c.circle(x + 4.5, y + 4.5, 4.5, stroke=1, fill=1)
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.82))
    c.circle(x + 4.5, y + 4.5, 2.15, stroke=1, fill=1)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.70))
    c.drawString(x + 13, y + 1.2, 'faster after optimization')
    c.restoreState()


def draw_delta_legend(c: canvas.Canvas, x: float, y: float, width: float, delta_max: float):
    c.saveState()
    c.setFont('Helvetica', 6.8)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.72))
    c.drawString(x, y + 42, 'Minutes faster than baseline')
    steps = 80
    bar_h = 10
    for i in range(steps):
        t = i / max(steps - 1, 1)
        if i == 0:
            rgb = np.array([255, 255, 255], dtype=np.uint8)
        else:
            tt = 0.18 + 0.82 * t
            rgb = (DELTA_MIN_BLUE * (1.0 - tt) + DELTA_BLUE * tt).clip(0, 255).astype(np.uint8)
        c.setFillColor(colors.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255))
        c.rect(x + width * i / steps, y + 25, width / steps + 0.4, bar_h, stroke=0, fill=1)
    c.setStrokeColor(colors.Color(17/255, 19/255, 21/255, alpha=0.16))
    c.roundRect(x, y + 25, width, bar_h, 3, stroke=1, fill=0)
    c.setFillColor(colors.Color(17/255, 19/255, 21/255, alpha=0.70))
    labels = [0, 0.25, 0.50, 0.75, 1.00] if delta_max <= 1.01 else [0, delta_max/4, delta_max/2, 3*delta_max/4, delta_max]
    for val in labels:
        px = x + width * (val / max(delta_max, 1e-9))
        c.line(px, y + 22, px, y + 24)
        label = f'{val:.2f}' if delta_max <= 1.01 and val not in (0, 1.0) else (f'{val:.0f}' if abs(val - round(val)) < 1e-6 else f'{val:.1f}')
        if val == labels[-1]:
            label += ' min'
        c.drawCentredString(px, y + 12, label)
    c.restoreState()


def create_existing_pdf(bounds, country_proj, canton_proj, baseline, optimized, baseline_img, optimized_img):
    c = canvas.Canvas(str(OUT_FINE_PDF), pagesize=(base.PAGE_W, base.PAGE_H))
    c.setTitle('Wallisellen reachability before and after optimization')
    c.setFillColor(colors.white)
    c.rect(0, 0, base.PAGE_W, base.PAGE_H, fill=1, stroke=0)
    base.draw_text(c, 'Reachability from Wallisellen: baseline and post-optimization', base.PAGE_MARGIN_X, base.PAGE_H - base.PAGE_MARGIN_Y - 17, 'Helvetica-Bold', 15.0, colors.Color(17/255, 19/255, 21/255))
    label_y = base.PAGE_H - base.PAGE_MARGIN_Y - base.TITLE_H
    map_bottom = base.PAGE_MARGIN_Y + base.LEGEND_H
    panel1_x = base.PAGE_MARGIN_X
    panel2_x = base.PAGE_MARGIN_X + base.SCALE * (base.APP_WIDTH + base.PANEL_GAP)
    base.draw_text(c, 'Baseline', panel1_x, label_y + 4, 'Helvetica-Bold', 10.8, colors.Color(17/255, 19/255, 21/255))
    base.draw_text(c, 'Post-optimization', panel2_x, label_y + 4, 'Helvetica-Bold', 10.8, colors.Color(17/255, 19/255, 21/255))
    draw_panel(c, bounds, country_proj, canton_proj, baseline, baseline_img, panel1_x, map_bottom, show_faster=False, show_contours=True)
    draw_panel(c, bounds, country_proj, canton_proj, optimized, optimized_img, panel2_x, map_bottom, show_faster=True, show_contours=True)
    draw_time_legend(c, base.PAGE_MARGIN_X, base.PAGE_MARGIN_Y + 6, 'Fastest travel time from Wallisellen')
    draw_faster_dot_legend(c, base.PAGE_MARGIN_X + 7 * 45 + 122, base.PAGE_MARGIN_Y + 9)
    c.showPage()
    c.save()


def create_delta_pdf(bounds, country_proj, canton_proj, optimized, optimized_img, delta_img, delta_max):
    c = canvas.Canvas(str(OUT_DELTA_PDF), pagesize=(base.PAGE_W, base.PAGE_H))
    c.setTitle('Wallisellen post-optimization reachability and faster-than-baseline delta')
    c.setFillColor(colors.white)
    c.rect(0, 0, base.PAGE_W, base.PAGE_H, fill=1, stroke=0)
    base.draw_text(c, 'Reachability from Wallisellen: post-optimization and faster-than-baseline field', base.PAGE_MARGIN_X, base.PAGE_H - base.PAGE_MARGIN_Y - 17, 'Helvetica-Bold', 14.3, colors.Color(17/255, 19/255, 21/255))
    label_y = base.PAGE_H - base.PAGE_MARGIN_Y - base.TITLE_H
    map_bottom = base.PAGE_MARGIN_Y + base.LEGEND_H
    panel1_x = base.PAGE_MARGIN_X
    panel2_x = base.PAGE_MARGIN_X + base.SCALE * (base.APP_WIDTH + base.PANEL_GAP)
    base.draw_text(c, 'Post-optimization', panel1_x, label_y + 4, 'Helvetica-Bold', 10.8, colors.Color(17/255, 19/255, 21/255))
    base.draw_text(c, 'Minutes faster than baseline', panel2_x, label_y + 4, 'Helvetica-Bold', 10.8, colors.Color(17/255, 19/255, 21/255))
    draw_panel(c, bounds, country_proj, canton_proj, optimized, optimized_img, panel1_x, map_bottom, show_faster=True, show_contours=True)
    draw_panel(c, bounds, country_proj, canton_proj, optimized, delta_img, panel2_x, map_bottom, show_faster=False, show_contours=False, draw_stations=False)
    draw_time_legend(c, base.PAGE_MARGIN_X, base.PAGE_MARGIN_Y + 6, 'Post-optimization fastest travel time')
    draw_faster_dot_legend(c, base.PAGE_MARGIN_X + 7 * 45 + 122, base.PAGE_MARGIN_Y + 9)
    draw_delta_legend(c, panel2_x + 12, base.PAGE_MARGIN_Y + 6, base.SCALE * base.APP_WIDTH - 24, delta_max)
    c.showPage()
    c.save()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reachability = load_reachability()
    boundaries = base.load_json(base.BOUNDARIES_PATH)
    geom = base.load_json(base.GEOM_PATH)
    bounds = base.projection_bounds(boundaries)
    country_proj = base.projected_polygons(bounds, base.country_polygons(boundaries))
    canton_proj = base.projected_polygons(bounds, base.canton_polygons(boundaries))
    baseline = base.panel_stations(reachability, geom, 'baseline')
    optimized = base.panel_stations(reachability, geom, 'optimized')

    print(f'Fine raster: {RASTER_COLUMNS} x {RASTER_ROWS} cells ({RASTER_COLUMNS * RASTER_ROWS:,})')
    grid_lat, grid_lon = grid_latlon(bounds)
    print('Interpolating baseline field...')
    baseline_values, baseline_covered = interpolate_grid(grid_lat, grid_lon, baseline, 'fastestTimeMin')
    print('Interpolating post-optimization field...')
    optimized_values, optimized_covered = interpolate_grid(grid_lat, grid_lon, optimized, 'fastestTimeMin')
    combined_covered = baseline_covered & optimized_covered
    delta_values = baseline_values - optimized_values
    positive_delta = delta_values[np.isfinite(delta_values) & combined_covered & (delta_values > 0)]
    delta_max = max(1.0, float(np.nanmax(positive_delta)) if positive_delta.size else 1.0)

    baseline_img = time_raster(baseline_values, baseline_covered)
    optimized_img = time_raster(optimized_values, optimized_covered)
    delta_img = delta_raster(delta_values, combined_covered, delta_max)

    create_existing_pdf(bounds, country_proj, canton_proj, baseline, optimized, baseline_img, optimized_img)
    create_delta_pdf(bounds, country_proj, canton_proj, optimized, optimized_img, delta_img, delta_max)
    print(f'Wrote {OUT_FINE_PDF}')
    print(f'Wrote {OUT_DELTA_PDF}')
    print(f'Max raster delta: {delta_max:.3f} min')
    print(f"Stations: {reachability.get('stationCount')} total, {reachability.get('reachableStations')} reachable, {reachability.get('improvedStations')} faster after optimization")


if __name__ == '__main__':
    main()
