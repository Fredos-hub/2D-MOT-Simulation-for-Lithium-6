"""Convert a human-readable Experiment_Settings.xlsx measurement sheet into a
folder of simulation parameter.json files (with sweep expansion).

Unit conventions are RE-DERIVED from the live code on every run (laser_tab.py
factor map + a parameters.py sanity check) and the script ABORTS/WARNS if they
drift — never trusts a hardcoded table. JSON structure follows the base config.

Usage:
    python excel_to_runs.py --xlsx <file> --sheet <name> --base <parameter.json> --out <dir>
"""
import argparse, json, re, copy, sys, math
from pathlib import Path
import numpy as np
import openpyxl

# ---------------------------------------------------------------- unit check
EXPECTED_FACTORS = {"beam_power": 1000.0, "waist": 1000.0,
                    "beam_frequency": 1.0, "detuning": 1.0}  # display = JSON * factor

def derive_factors(repo: Path):
    src = (repo / "GUI/widgets/tabs/laser_tab.py").read_text()
    f = {m.group(1): float(m.group(2))
         for m in re.finditer(r"'key':\s*'(\w+)',\s*'factor':\s*([\d.eE+]+)", src)}
    warns = []
    for k, exp in EXPECTED_FACTORS.items():
        if k not in f:
            warns.append(f"laser_tab.py: factor for '{k}' not found — using {exp}")
        elif f[k] != exp:
            warns.append(f"laser_tab.py: factor for '{k}' is {f[k]}, expected {exp} "
                         f"— UNIT CONVENTION CHANGED, verify before trusting output!")
    # JSON times are ms (parameters.py multiplies by 1e-3); confirm
    psrc = (repo / "src/parameters.py").read_text()
    if 'simulated_time"] * 1e-3' not in psrc:
        warns.append("parameters.py: simulated_time no longer parsed as ms — verify time units!")
    return {**EXPECTED_FACTORS, **f}, warns

# ---------------------------------------------------------------- xlsx parsing
SECTIONS = ["ÜBERSICHT", "MOT-STRAHLEN", "PUSHBEAM", "SPEC-BEAM",
            "UMGEBUNG", "WEITERE PARAMETER", "PARAMETER-SCAN"]

def parse_sheet(ws):
    rows = [[c.value for c in r] for r in ws.iter_rows()]
    out = {"mot": [], "push": {}, "spec": {}, "scan": []}
    sec = None
    for r in rows:
        a = (r[0] or "")
        head = next((s for s in SECTIONS if isinstance(a, str) and a.startswith(s)), None)
        if head:
            sec = head; continue
        if sec == "MOT-STRAHLEN" and isinstance(a, str) and (a.startswith("Trap") or a.startswith("Repump")):
            out["mot"].append(r)                              # [Strahl,Leistung,BasisF,Verst,GesamtF,Hand,Waist]
        elif sec == "PUSHBEAM" and a:
            out["push"][a] = (r[1], r[2])                     # label -> (value, unit)
        elif sec == "SPEC-BEAM" and a:
            out["spec"][a] = (r[1], r[2])
        elif sec == "PARAMETER-SCAN" and a and not a.startswith("Gescannter"):
            out["scan"].append((a, r[1], r[2], r[3]))         # name, werte, einheit, notiz
    return out

def num(x):
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("−", "-").strip()
    try: return float(s)
    except ValueError: return None

def handedness_num(cell):
    """Authoritative = the numeric ±1/0 in the cell; fall back to label."""
    if cell is None: return None
    s = str(cell).replace("−", "-")
    m = re.search(r"[+-]?\d", s)
    if m: return int(m.group())
    u = s.upper()
    return 1 if "LH" in u else -1 if "RH" in u else 0 if "LIN" in u else None

# ---------------------------------------------------------------- value-spec parser
def parse_values(spec):
    if spec is None: return None
    s = str(spec).replace("−", "-").strip()
    if "vs" in s.lower() or "invert" in s.lower():
        return ["normal", "invertiert"]
    if "," in s:
        return [float(p) for p in s.split(",") if num(p) is not None]
    m = re.search(r"(-?[\d.]+)\s*(?:…|\.\.\.)\s*(\+?-?[\d.]+).*?Schritt\s*([\d.]+)", s)
    if m:
        a, b, st = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return list(np.round(np.arange(a, b + st / 2, st), 6))
    return None  # unparseable (e.g. range without step)

# ---------------------------------------------------------------- config helpers
def is_push(l): return l["direction"][2] == 1
def mot(cfg, t): return [l for l in cfg["Lasers"] if not is_push(l) and l.get("type") == t]
def pushes(cfg): return [l for l in cfg["Lasers"] if is_push(l)]

def apply_beam_row(cfg, t, row, F, bottom_only=False):
    """Override power/frequency/detuning/waist (uniform). Handedness is handled
    separately (can't collapse the per-beam MOT pattern into one value)."""
    pw, bf, det, gf, w = row[1], row[2], row[3], row[4], row[6]
    for l in mot(cfg, t):
        if bottom_only and l["direction"][1] <= 0:
            continue
        if num(pw) is not None:  l["beam_power"] = num(pw) / F["beam_power"]
        if num(bf) is not None:  l["beam_frequency"] = num(bf) * 1e6 / F["beam_frequency"]
        elif num(gf) is not None: l["beam_frequency"] = num(gf) / F["beam_frequency"]
        if num(det) is not None: l["detuning"] = num(det) / F["detuning"]
        if num(w) is not None:   l["waist"] = num(w) / F["waist"]

def apply_static(cfg, P, F, warns):
    def get_row(prefix, bottom):
        for r in P["mot"]:
            lbl = str(r[0])
            if lbl.startswith(prefix) and (("unten" in lbl) == bottom):
                return r
        return None
    for t, prefix in [("trap", "Trap"), ("repump", "Repump")]:
        top, bot = get_row(prefix, False), get_row(prefix, True)
        if top: apply_beam_row(cfg, t, top, F)
        if bot and any(num(bot[i]) is not None for i in (1, 2, 3, 4)):
            apply_beam_row(cfg, t, bot, F, bottom_only=True)        # bottom power/freq/det/waist
        th = handedness_num(top[5]) if top else None
        bh = handedness_num(bot[5]) if bot else None
        if th is not None and bh is not None and bh != th:          # flip = invert bottom beams
            for l in mot(cfg, t):
                if l["direction"][1] > 0:
                    l["handedness"] = -l["handedness"]
            warns.append(f"{prefix}: 'unten'-Händigkeit weicht ab → untere (+y) Strahlen "
                         f"invertiert (Flip); Basis-Muster sonst beibehalten.")
    # push
    pu = P["push"]
    def pv(k): return pu.get(k, (None, None))[0]
    pl = pushes(cfg)
    if pl:
        tot0 = sum(l["beam_power"] for l in pl) or 1.0
        if num(pv("Leistung")) is not None:
            tot = num(pv("Leistung")) / F["beam_power"]
            for l in pl: l["beam_power"] = tot * (l["beam_power"] / tot0)
        for l in pl:
            if num(pv("Basisfrequenz")) is not None:
                l["beam_frequency"] = num(pv("Basisfrequenz")) * 1e6 / F["beam_frequency"]
            elif num(pv("Gesamtfrequenz")) is not None:
                l["beam_frequency"] = num(pv("Gesamtfrequenz")) / F["beam_frequency"]
            if num(pv("Verstimmung")) is not None: l["detuning"] = num(pv("Verstimmung")) / F["detuning"]
            if handedness_num(pv("Händigkeit")) is not None: l["handedness"] = handedness_num(pv("Händigkeit"))
            if num(pv("Waist")) is not None: l["waist"] = num(pv("Waist")) / F["waist"]
        t_on = num(pv("Verzögerung MOT-Start → Push (t_on)"))
        if t_on is not None:
            for l in pl: l["t_on"] = t_on
        pulse = num(pv("Pulsdauer (t_off − t_on)"))
        if pulse is not None:
            for l in pl: l["t_off"] = l["t_on"] + pulse
            cfg["Simulation"]["simulated_time"] = pl[0]["t_on"] + pulse
    if any(num(v[0]) is not None for v in P["spec"].values()):
        warns.append("SPEC-BEAM ist ausgefüllt, aber in der Basis-Konfig nicht abgebildet — übersprungen.")

# scan appliers (value -> mutate cfg)
def flip_bottom(cfg):
    for l in cfg["Lasers"]:
        if not is_push(l) and l["direction"][1] > 0:
            l["handedness"] = -l["handedness"]

def scan_apply(cfg, name, v, F):
    n = name.lower()
    if "händigkeit" in n:
        if str(v).startswith("invert"): flip_bottom(cfg)
    elif "verstimmung" in n:
        for l in pushes(cfg): l["detuning"] = v / F["detuning"]
    elif "leistung" in n:
        pl = pushes(cfg); tot0 = sum(l["beam_power"] for l in pl) or 1.0; tot = v / F["beam_power"]
        for l in pl: l["beam_power"] = tot * (l["beam_power"] / tot0)
    elif "pulsdauer" in n:
        for l in pushes(cfg): l["t_off"] = l["t_on"] + v
        cfg["Simulation"]["simulated_time"] = pushes(cfg)[0]["t_on"] + v
    elif "t_on" in n or ("verzögerung" in n and "mot" in n):
        for l in pushes(cfg):
            dur = l["t_off"] - l["t_on"]; l["t_on"] = v; l["t_off"] = v + dur
        cfg["Simulation"]["simulated_time"] = pushes(cfg)[0]["t_off"]
    elif "waist" in n:
        for l in pushes(cfg): l["waist"] = v / F["waist"]
    else:
        return False
    return True

REQUIRED = ["Atoms", "Lasers", "Magnetic_Fields", "Boundaries", "Simulation"]

def slug(s):
    return re.sub(r"[^a-zA-Z0-9.+-]+", "_", str(s)).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True); ap.add_argument("--sheet", required=True)
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()
    repo = Path(a.repo)
    F, warns = derive_factors(repo)
    for w in warns: print("WARN:", w)
    base = json.load(open(a.base))
    for k in REQUIRED:
        if k not in base: sys.exit(f"Basis-Konfig fehlt Schlüssel '{k}'")
    P = parse_sheet(openpyxl.load_workbook(a.xlsx, data_only=True)[a.sheet])
    nominal = copy.deepcopy(base); apply_static(nominal, P, F, warns)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    written = []
    scans = [(name, parse_values(werte)) for name, werte, _, _ in P["scan"]]
    scans = [(n, vs) for n, vs in scans if vs]
    if not scans:
        p = out / "run.json"; json.dump(nominal, open(p, "w"), indent=2); written.append(p)
    for name, values in scans:
        sub = out / slug(name); sub.mkdir(exist_ok=True)
        for v in values:
            cfg = copy.deepcopy(nominal)
            if not scan_apply(cfg, name, v, F):
                print(f"WARN: Scan-Parameter '{name}' nicht zugeordnet — übersprungen."); break
            p = sub / f"{slug(v)}.json"; json.dump(cfg, open(p, "w"), indent=2); written.append(p)
    # report unparseable scans
    for name, werte, _, _ in P["scan"]:
        if parse_values(werte) is None:
            print(f"WARN: Scan-Werte '{werte}' für '{name}' nicht interpretierbar "
                  f"(z. B. Bereich ohne Schritt) — übersprungen.")
    print(f"\n{len(written)} parameter.json geschrieben nach {out}")
    for p in written[:12]: print("  ", p.relative_to(out))
    if len(written) > 12: print(f"   … (+{len(written)-12})")

if __name__ == "__main__":
    main()
