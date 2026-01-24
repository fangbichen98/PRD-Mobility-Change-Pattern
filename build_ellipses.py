#!/usr/bin/env python3
"""
Compute per-grid, per-year confidence ellipses from OD CSVs and grid centers.

Inputs (CSV):
  - grid centers: CSV with columns: grid_id, lon, lat
  - per-year OD CSVs with columns: date_dt, time, o_grid_500, d_grid_500, num_total

Definition: For each grid g and year y, collect its counterpart grid centers
  from OD edges, weighted by num_total, under a chosen direction policy:
    - both: outgoing and incoming combined (default; preserves current behavior)
    - out:  outgoing only (og -> dg)
    - in:   incoming only (dg <- og)
  Compute weighted mean and covariance in local meters (x east, y north)
  relative to g's center, then map covariance to a 2D Gaussian confidence ellipse
  using chi-square threshold for confidence p (default 0.95).

Output (JSON): vis/appdata/ellipses.json
  {
    "years": {
      "2018": [ {"grid_id": 165168, "center": {"lon":..,"lat":..},
                   "axes": {"a": meters, "b": meters},
                   "angle_deg": deg,  # major axis angle, degrees, relative to +x (east), CCW
                   "mean_offset_m": {"dx": meters, "dy": meters},  # mean vector from grid center, meters
                   "mean_angle_deg": deg  # angle of mean vector, degrees, relative to +x (east), CCW
                 }, ...],
      ...
    }
  }

Usage example:
  python vis/tools/build_ellipses.py \
    --grid-centers vis/data/grid_metadata/PRD_grid_metadata.csv \
    --year-file 2018:vis/data/2018.csv \
    --year-file 2021:vis/data/2021.csv \
    --year-file 2024:vis/data/2024.csv \
    --confidence 0.95 \
    --out vis/appdata/ellipses.json
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid-centers', required=True, help='CSV with grid_id,lon,lat')
    ap.add_argument('--year-file', action='append', default=[],
                    help='Mapping YEAR:PATH to OD CSV (date_dt,time,o_grid_500,d_grid_500,num_total). Can repeat.')
    ap.add_argument('--confidence', type=float, default=0.95, help='Confidence level in (0,1), default 0.95')
    ap.add_argument('--out', required=True, help='Output JSON path')
    ap.add_argument('--direction', choices=['both', 'out', 'in'], default='both',
                    help='Which flows to include when building counterparts: both|out|in (default: both)')
    return ap.parse_args()

def chi2_quantile_2d(p: float) -> float:
    """Return chi-square quantile for df=2 at confidence p.
    For common p we hardcode to avoid SciPy dependency.
    """
    tbl = {
        0.50: 1.3863,
        0.68: 2.2789,
        0.80: 3.2189,
        0.90: 4.6052,
        0.95: 5.9915,
        0.975: 7.3778,
        0.99: 9.2103,
        0.999: 13.8155,
    }
    # nearest lookup
    if p in tbl:
        return tbl[p]
    # simple linear interpolation among sorted keys
    keys = sorted(tbl.keys())
    if p <= keys[0]:
        return tbl[keys[0]]
    if p >= keys[-1]:
        return tbl[keys[-1]]
    for i in range(1, len(keys)):
        if p <= keys[i]:
            p0, p1 = keys[i-1], keys[i]
            v0, v1 = tbl[p0], tbl[p1]
            t = (p - p0) / (p1 - p0)
            return v0 + t * (v1 - v0)
    return tbl[0.95]

def read_grid_centers(path):
    centers = {}
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        # Try to detect column names (grid_id, lon, lat) with some flexibility
        cols = {k.lower(): k for k in r.fieldnames or []}
        gid_col = cols.get('grid_id') or cols.get('grid') or cols.get('id')
        lon_col = cols.get('lon') or cols.get('lng') or cols.get('longitude')
        lat_col = cols.get('lat') or cols.get('latitude')
        if not (gid_col and lon_col and lat_col):
            raise ValueError('grid centers CSV must have columns: grid_id, lon, lat')
        for row in r:
            try:
                gid = int(row[gid_col])
                lon = float(row[lon_col])
                lat = float(row[lat_col])
            except Exception:
                continue
            centers[gid] = (lon, lat)
    return centers

def deg2m_factors(lat_deg: float):
    # Approx meters per degree at latitude
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_deg))
    return m_per_deg_lon, m_per_deg_lat

def accumulate_counterparts(year_csv_path, centers, direction: str):
    # For each grid, collect weighted counterpart points (lon, lat, weight)
    out = defaultdict(list)
    with open(year_csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        cols = {k: k for k in (r.fieldnames or [])}
        required = ['o_grid_500', 'd_grid_500', 'num_total']
        for k in required:
            if k not in cols:
                raise ValueError(f'Missing column {k} in {year_csv_path}')
        for row in r:
            try:
                og = int(row['o_grid_500'])
                dg = int(row['d_grid_500'])
                w = float(row['num_total'])
            except Exception:
                continue
            if w <= 0:
                continue
            if og in centers and dg in centers:
                if direction in ('both', 'out'):
                    # outgoing: counterpart for og is destination center
                    out[og].append((*centers[dg], w))
                if direction in ('both', 'in'):
                    # incoming: counterpart for dg is origin center
                    out[dg].append((*centers[og], w))
    return out

def weighted_mean(points):
    # points: list of (x, y, w)
    sw = sum(w for _,_,w in points)
    if sw <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(x*w for x,y,w in points) / sw
    my = sum(y*w for x,y,w in points) / sw
    return mx, my, sw

def weighted_cov(points, mx, my):
    sw = sum(w for _,_,w in points)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0
    # Use weighted covariance (population-like, divide by sum of weights)
    sxx = sum((x-mx)*(x-mx)*w for x,y,w in points) / sw
    syy = sum((y-my)*(y-my)*w for x,y,w in points) / sw
    sxy = sum((x-mx)*(y-my)*w for x,y,w in points) / sw
    return sxx, sxy, syy, sw

def eig2x2(a, b, c):
    # Matrix [[a, b], [b, c]] -> eigenvalues/eigenvectors
    tr = a + c
    det = a*c - b*b
    disc = max(0.0, tr*tr/4.0 - det)
    s = math.sqrt(disc)
    l1 = tr/2.0 + s
    l2 = tr/2.0 - s
    # eigenvector for l1
    if abs(b) > 1e-12:
        v1x = l1 - c
        v1y = b
    else:
        # diagonal matrix or close to
        if a >= c:
            v1x, v1y = 1.0, 0.0
        else:
            v1x, v1y = 0.0, 1.0
    n = math.hypot(v1x, v1y)
    if n == 0:
        v1x, v1y = 1.0, 0.0
    else:
        v1x, v1y = v1x/n, v1y/n
    return l1, l2, v1x, v1y

def build_for_year(year, year_csv_path, centers, chi2, direction: str):
    counterparts = accumulate_counterparts(year_csv_path, centers, direction)
    results = []
    for gid, (glon, glat) in centers.items():
        pts = counterparts.get(gid)
        if not pts:
            continue
        # convert counterpart lon/lat to local meters relative to grid center
        m_per_deg_lon, m_per_deg_lat = deg2m_factors(glat)
        local_pts = []
        for lon, lat, w in pts:
            dx = (lon - glon) * m_per_deg_lon
            dy = (lat - glat) * m_per_deg_lat
            local_pts.append((dx, dy, w))
        mx, my, sw = weighted_mean(local_pts)
        sxx, sxy, syy, _ = weighted_cov(local_pts, mx, my)
        # guard against degenerate covariances
        sxx = max(sxx, 1e-6)
        syy = max(syy, 1e-6)
        l1, l2, v1x, v1y = eig2x2(sxx, sxy, syy)
        l1 = max(l1, 1e-6)
        l2 = max(l2, 1e-6)
        a = math.sqrt(chi2 * l1)
        b = math.sqrt(chi2 * l2)
        # angle of major axis relative to +x (east), in degrees
        angle = math.degrees(math.atan2(v1y, v1x))
        # mean vector angle relative to +x (east), in degrees
        mean_angle = math.degrees(math.atan2(my, mx)) if (mx != 0.0 or my != 0.0) else 0.0
        results.append({
            'grid_id': gid,
            'center': {'lon': glon, 'lat': glat},
            'axes': {'a': a, 'b': b},
            'angle_deg': angle,  # major axis angle, east-CCW
            'mean_offset_m': {'dx': mx, 'dy': my},
            'mean_angle_deg': mean_angle,
            'weight_sum': sw,
        })
    return results

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    centers = read_grid_centers(args.grid_centers)
    # Parse year-file mappings
    year_paths = {}
    for yf in args.year_file:
        if ':' not in yf:
            raise SystemExit('Each --year-file must be YEAR:PATH')
        y, p = yf.split(':', 1)
        y = y.strip()
        year_paths[y] = p.strip()
    chi2 = chi2_quantile_2d(max(1e-6, min(0.999, args.confidence)))
    out = {'years': {}}
    for y, p in year_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        res = build_for_year(y, p, centers, chi2, args.direction)
        out['years'][y] = res
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'Wrote {args.out} with years: {sorted(out["years"].keys())}')

if __name__ == '__main__':
    main()
