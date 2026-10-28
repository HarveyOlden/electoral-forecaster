"""Builds data/house_hex_layout.json: one hexagon per House district, grouped by state, roughly where the state is.
Seats per state are the 2020 apportionment (unchanged for 2026). District numbers inside a state are placed
west->east, north->south. Swap this file for a hand-made layout at any time: the site only reads the JSON."""
import json, math, os

STATES = {  # postal: (fips, name, seats, lat, lon) - lat/lon = rough visual centre
    "AL": ("01", "Alabama", 7, 32.8, -86.8), "AK": ("02", "Alaska", 1, 0, 0), "AZ": ("04", "Arizona", 9, 34.3, -111.7),
    "AR": ("05", "Arkansas", 4, 34.9, -92.4), "CA": ("06", "California", 52, 37.2, -119.5), "CO": ("08", "Colorado", 8, 39.0, -105.5),
    "CT": ("09", "Connecticut", 5, 41.6, -72.7), "DE": ("10", "Delaware", 1, 39.0, -75.5), "FL": ("12", "Florida", 28, 28.6, -82.4),
    "GA": ("13", "Georgia", 14, 32.7, -83.4), "HI": ("15", "Hawaii", 2, 0, 0), "ID": ("16", "Idaho", 2, 44.4, -114.6),
    "IL": ("17", "Illinois", 17, 40.0, -89.2), "IN": ("18", "Indiana", 9, 39.9, -86.3), "IA": ("19", "Iowa", 4, 42.1, -93.5),
    "KS": ("20", "Kansas", 4, 38.5, -98.4), "KY": ("21", "Kentucky", 6, 37.5, -85.3), "LA": ("22", "Louisiana", 6, 31.0, -92.0),
    "ME": ("23", "Maine", 2, 45.4, -69.2), "MD": ("24", "Maryland", 8, 39.0, -76.8), "MA": ("25", "Massachusetts", 9, 42.3, -71.8),
    "MI": ("26", "Michigan", 13, 43.6, -84.7), "MN": ("27", "Minnesota", 8, 46.3, -94.3), "MS": ("28", "Mississippi", 4, 32.7, -89.7),
    "MO": ("29", "Missouri", 8, 38.4, -92.5), "MT": ("30", "Montana", 2, 47.0, -109.6), "NE": ("31", "Nebraska", 3, 41.5, -99.8),
    "NV": ("32", "Nevada", 4, 39.3, -116.6), "NH": ("33", "New Hampshire", 2, 43.7, -71.6), "NJ": ("34", "New Jersey", 12, 40.2, -74.6),
    "NM": ("35", "New Mexico", 3, 34.4, -106.1), "NY": ("36", "New York", 26, 42.9, -75.5), "NC": ("37", "North Carolina", 14, 35.5, -79.4),
    "ND": ("38", "North Dakota", 1, 47.5, -100.5), "OH": ("39", "Ohio", 15, 40.3, -82.8), "OK": ("40", "Oklahoma", 5, 35.6, -97.5),
    "OR": ("41", "Oregon", 6, 43.9, -120.6), "PA": ("42", "Pennsylvania", 17, 40.9, -77.8), "RI": ("44", "Rhode Island", 2, 41.7, -71.5),
    "SC": ("45", "South Carolina", 7, 33.9, -80.9), "SD": ("46", "South Dakota", 1, 44.4, -100.2), "TN": ("47", "Tennessee", 9, 35.9, -86.4),
    "TX": ("48", "Texas", 38, 31.5, -99.3), "UT": ("49", "Utah", 4, 39.3, -111.7), "VT": ("50", "Vermont", 1, 44.1, -72.7),
    "VA": ("51", "Virginia", 11, 37.5, -78.8), "WA": ("53", "Washington", 10, 47.4, -120.5), "WV": ("54", "West Virginia", 2, 38.6, -80.6),
    "WI": ("55", "Wisconsin", 8, 44.6, -89.9), "WY": ("56", "Wyoming", 1, 43.0, -107.5),
}


def project(lat, lon):            # simple conic-ish projection -> hex grid units
    x = (lon + 96) * math.cos(math.radians(lat)) * 0.62
    y = (lat - 38) * -0.78
    return x, y


def hex_xy(q, r):                 # pointy-top axial -> cartesian
    return math.sqrt(3) * (q + r / 2), 1.5 * r


NEI = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def relax(anchor, n_of, scale=2.2, iters=800):
    """Spread the state centres so each state's circle (area ~ its seats) fits without overlapping its neighbours,
    while staying close to where the state really is."""
    pos = {s: [x * scale, y * scale] for s, (x, y) in anchor.items()}
    home = {s: tuple(v) for s, v in pos.items()}
    rad = {s: 0.88 * math.sqrt(n_of[s]) + 0.45 for s in pos}
    keys = list(pos)
    for _ in range(iters):
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                dx, dy = pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]
                d = math.hypot(dx, dy) or 1e-6
                need = rad[a] + rad[b]
                if d < need:
                    push = (need - d) / 2
                    ux, uy = dx / d, dy / d
                    wa, wb = rad[b] / (rad[a] + rad[b]), rad[a] / (rad[a] + rad[b])
                    pos[a][0] -= ux * push * wa; pos[a][1] -= uy * push * wa
                    pos[b][0] += ux * push * wb; pos[b][1] += uy * push * wb
        for s in keys:                                   # gentle pull home
            pos[s][0] += 0.02 * (home[s][0] - pos[s][0]); pos[s][1] += 0.02 * (home[s][1] - pos[s][1])
    return {s: tuple(v) for s, v in pos.items()}


def main(out="data/house_hex_layout.json"):
    taken, blobs = {}, {s: [] for s in STATES}
    anchor = {s: project(lat, lon) for s, (_, _, n, lat, lon) in STATES.items() if s not in ("AK", "HI")}
    anchor = relax(anchor, {s: STATES[s][2] for s in anchor})
    order = sorted(anchor, key=lambda s: -STATES[s][2])
    remaining = {s: STATES[s][2] for s in anchor}
    dist = lambda c, s: (hex_xy(*c)[0] - anchor[s][0]) ** 2 + (hex_xy(*c)[1] - anchor[s][1]) ** 2
    allcells = {(q, r) for q in range(-70, 70) for r in range(-30, 30)}
    while any(remaining.values()):
        for s in order:
            if not remaining[s]:
                continue
            cand = {(q + dq, r + dr) for q, r in blobs[s] for dq, dr in NEI} - set(taken) if blobs[s] else set()
            if not cand:                               # first hex (or boxed in): nearest free cell to the centre
                cand = allcells - set(taken)
            best = min(cand, key=lambda c: dist(c, s))
            taken[best] = s
            blobs[s].append(best)
            remaining[s] -= 1
    # Alaska and Hawaii: bottom-left, below the mainland
    ys = [hex_xy(*c)[1] for c in taken]
    xs = [hex_xy(*c)[0] for c in taken]
    base_r = round(max(ys) / 1.5) + 2
    base_q = round(min(xs) / math.sqrt(3)) - base_r // 2
    blobs["AK"] = [(base_q, base_r)]
    blobs["HI"] = [(base_q + 3, base_r), (base_q + 4, base_r)]
    hexes = []
    for s, cells in blobs.items():
        cells = sorted(cells, key=lambda c: (round(hex_xy(*c)[1]), hex_xy(*c)[0]))   # north->south, west->east
        n = STATES[s][2]
        for k, (q, r) in enumerate(cells):
            hexes.append(dict(id=f"{s}-AL" if n == 1 else f"{s}-{k + 1:02d}", state=s, district=0 if n == 1 else k + 1, q=q, r=r))
    states = [dict(postal=s, fips=v[0], name=v[1], seats=v[2]) for s, v in STATES.items()]
    assert len(hexes) == 435, len(hexes)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(note="Hex tile layout: one hexagon per district (pointy-top axial q, r). Regenerate with tools/make_hex_layout.py "
                        "or replace with a hand-made layout.", hexes=hexes, states=states), open(out, "w"))
    print(f"{len(hexes)} districts written to {out}")


if __name__ == "__main__":
    main()
