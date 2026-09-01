"""
SkaiBOT — fabrique les VRAIES cartes Australie pour l'entraînement.

OpenFront amont publie la carte `australia` en 2000x1500 avec un pyramidage déjà
calculé (map4x = 1000x750, map16x = 500x375). Ce script les rapatrie et produit des
paquets au format attendu par `base-game` (map.bin + mini_map.bin + manifest.json),
à plusieurs résolutions, pour que l'IA s'entraîne sur la VRAIE géographie :
côte découpée, Tasmanie, désert central, et surtout DE L'OCÉAN (le moteur refuse de
faire passer des troupes sur l'eau : c'est ce qui rend le speedrun d'Australie
intéressant, et ce qui manque totalement aux cartes `*_nt` actuelles).

Usage (depuis la racine du repo) :
    python skai/make_maps.py                      # récupère + génère 3 résolutions
    python skai/make_maps.py --repair             # régénère les 2 cartes cassées
    python skai/make_maps.py --from /chemin/vers/australia   # sans réseau
    python skai/make_maps.py --list               # état des paquets produits

Le format binaire (cf. base-game/src/core/game/GameMap.ts) : un octet par tuile —
  bit 7 = IS_LAND, bit 6 = SHORELINE, bit 5 = OCEAN, bits 0-4 = magnitude
  (magnitude <10 plaines, <20 collines, >=20 montagnes ; la magnitude module le
   coût de conquête via terrainType dans AttackExecution.addNeighbors).
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MAPS = os.path.join(ROOT, "base-game", "resources", "maps")
CACHE = os.path.join(MAPS, "_upstream_australia")

IS_LAND, IS_SHORE, IS_OCEAN, MAG_MASK = 1 << 7, 1 << 6, 1 << 5, 0x1F
API = "https://api.github.com/repos/openfrontio/OpenFrontIO/git/blobs/{sha}"
# sha court des fichiers de resources/maps/australia sur main (vérifié le 2026-09-01)
UPSTREAM = {
    "manifest.json": "6622f9e30a1cfc67f4929de45aa83cd7f6dbcde2",
    "map.bin": "900ba5c0ce8765d610b920b268516c6df07643b4",
    "map4x.bin": "fbbae1825926c7a079ee81e9747d945a11f0f5a9",
    "map16x.bin": "c90e63fab4268ae937115dc5f97e2d28275539b6",
}


# ----------------------------------------------------------------------------- récupération
def fetch_upstream(timeout=300):
    os.makedirs(CACHE, exist_ok=True)
    missing = {k: v for k, v in UPSTREAM.items() if not os.path.exists(os.path.join(CACHE, k))}
    if not missing:
        print(f"  cache déjà complet : {CACHE}")
        return
    print(f"  téléchargement de {len(missing)} fichier(s) depuis OpenFrontIO/main ...")
    for name, sha in missing.items():
        url = API.format(sha=sha)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = json.load(r)
            raw = base64.b64decode(data["content"])
            if len(raw) != data["size"]:
                raise ValueError(f"taille incohérente ({len(raw)} != {data['size']})")
        except Exception as e:  # réseau coupé, proxy, etc.
            raise SystemExit(
                f"\n❌ impossible de récupérer {name} ({e}).\n"
                f"   Récupère à la main le dossier resources/maps/australia du dépôt\n"
                f"   https://github.com/openfrontio/OpenFrontIO puis relance avec\n"
                f"   python skai/make_maps.py --from /chemin/vers/australia\n"
            )
        with open(os.path.join(CACHE, name), "wb") as f:
            f.write(raw)
        print(f"    {name:16s} {len(raw):>9,d} octets")


def load_raw(path):
    return np.frombuffer(open(path, "rb").read(), dtype=np.uint8)


# --------------------------------------------------------------------------------- encodeur
def decode(arr, w, h):
    a = arr.reshape(h, w)
    return {
        "land": ((a >> 7) & 1).astype(bool),
        "ocean": ((a >> 5) & 1).astype(bool),
        "mag": (a & MAG_MASK).astype(np.int32),
    }


def downsample(arr, w, h, factor):
    """Réduction par blocs, en reproduisant la règle du générateur officiel
    (majorité de tuiles terrestres) — validée à 99.7 % contre map4x amont."""
    nh, nw = h // factor, w // factor
    b = arr[: nh * factor, : nw * factor].reshape(nh, factor, nw, factor)
    land = (((b >> 7) & 1) == 1).astype(np.float32).mean(axis=(1, 3))
    is_land = land >= 0.5
    out = np.zeros((nh, nw), dtype=np.uint8)
    out[is_land] |= IS_LAND
    # magnitude = moyenne des magnitudes des tuiles terrestres du bloc
    mag = (b & MAG_MASK).astype(np.float32)
    lm = (((b >> 7) & 1) == 1)
    tot = np.maximum(lm.sum(axis=(1, 3)), 1)
    m = np.round((mag * lm).sum(axis=(1, 3)) / tot).astype(np.int32)
    out[is_land] |= np.clip(m[is_land], 0, MAG_MASK).astype(np.uint8)
    # océan vs lac : de l'eau devient "océan" si la majorité de l'eau du bloc l'était
    water = ~is_land
    oc = (((b >> 5) & 1) == 1).astype(np.float32)
    oc = np.where(water, oc.mean(axis=(1, 3)) >= 0.5, False)
    out[water & oc] |= IS_OCEAN
    return add_shoreline(out, is_land)


def add_shoreline(out, is_land):
    """bit 6 = tuile terrestre touchant de l'eau (le moteur s'en sert pour les ports)."""
    water = ~is_land
    sh = np.zeros_like(is_land)
    sh[1:, :] |= is_land[1:, :] & water[:-1, :]
    sh[:-1, :] |= is_land[:-1, :] & water[1:, :]
    sh[:, 1:] |= is_land[:, 1:] & water[:, :-1]
    sh[:, :-1] |= is_land[:, :-1] & water[:, 1:]
    out[sh] |= IS_SHORE
    return out


def crop(arr, w, h, x, y, size):
    x = min(max(int(x), 0), w - size)
    y = min(max(int(y), 0), h - size)
    return arr[y: y + size, x: x + size].copy()


def write_pack(name, terrain, mini, nations_src=None, src_w=2000, src_h=1500, off_x=0, off_y=0):
    """Écrit base-game/resources/maps/<name>/{map.bin,mini_map.bin,manifest.json}."""
    d = os.path.join(MAPS, name)
    os.makedirs(d, exist_ok=True)
    h, w = terrain.shape
    mh, mw = mini.shape
    land = (((terrain >> 7) & 1) == 1)
    open(os.path.join(d, "map.bin"), "wb").write(np.ascontiguousarray(terrain).tobytes())
    open(os.path.join(d, "mini_map.bin"), "wb").write(np.ascontiguousarray(mini).tobytes())
    # repère du recadrage : on translate l'origine puis on applique l'échelle de réduction,
    # pour que les coordonnées des nations restent justes sur le paquet produit.
    scale_x = w / max(src_w - off_x, 1) if src_w > 0 else 1
    scale_y = h / max(src_h - off_y, 1) if src_h > 0 else 1
    nations = []
    for n in (nations_src or []):
        cx, cy = n["coordinates"]
        nations.append({
            "coordinates": [int((cx - off_x) * scale_x), int((cy - off_y) * scale_y)],
            "flag": n.get("flag", ""),
            "name": n.get("name", ""),
            "strength": n.get("strength", 1),
        })
    nations = [n for n in nations if 0 <= n["coordinates"][0] < w and 0 <= n["coordinates"][1] < h]
    manifest = {
        "name": name,
        "map": {"width": int(w), "height": int(h), "num_land_tiles": int(land.sum())},
        "mini_map": {
            "width": int(mw), "height": int(mh),
            # le mini_map sert au pathfinding des bateaux : mêmes règles, mêmes terres
            "num_land_tiles": int((((mini >> 7) & 1) == 1).sum()),
        },
        "nations": nations,
        "source": {"original_map": "australia (OpenFrontIO/main)", "generated_by": "skai/make_maps.py"},
        "crop": {"offset_x": int(off_x), "offset_y": int(off_y), "source_width": int(src_w), "source_height": int(src_h)},
    }
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    pct = land.mean() * 100
    print(f"  ✅ {name:28s} {w}x{h}  terres {int(land.sum()):>7,d} ({pct:4.1f}%)  "
          f"océan {int((((terrain >> 5) & 1) == 1).sum()):>7,d}  nations {len(nations)}")
    return manifest


# -------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None, help="dossier local contenant map.bin/map4x.bin/map16x.bin")
    ap.add_argument("--repair", action="store_true", help="régénère australia_100x100 et australia_500x500 (actuellement 0 tuile terrestre)")
    ap.add_argument("--sizes", default="500,1000,2000", help="largeurs des paquets réels à produire")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'paquet':30s} {'grille':>10s} {'terres':>9s} {'%':>6s}   ok?")
        for n in sorted(os.listdir(MAPS)):
            p = os.path.join(MAPS, n)
            if not os.path.isdir(p) or n.startswith("_"):
                continue
            try:
                man = json.load(open(os.path.join(p, "manifest.json")))
                t = load_raw(os.path.join(p, "map.bin"))
                w, h = man["map"]["width"], man["map"]["height"]
                land = (((t[: w * h].reshape(h, w) >> 7) & 1) == 1)
                ok = "✅" if land.sum() > 0 else "❌ 0 tuile terrestre"
                print(f"{n:30s} {w:>4d}x{h:<5d} {int(land.sum()):>9,d} {land.mean()*100:5.1f}%   {ok}")
            except Exception as e:
                print(f"{n:30s} {'?':>10s} {'?':>9s} {'?':>6s}   ⚠ {e}")
        return

    if args.src:
        src = args.src
        files = {f: os.path.join(src, f) for f in ("map.bin", "map4x.bin", "map16x.bin", "manifest.json")}
        for f, p in files.items():
            if not os.path.exists(p):
                raise SystemExit(f"❌ il manque {f} dans {src}")
    else:
        fetch_upstream()
        files = {f: os.path.join(CACHE, f) for f in ("map.bin", "map4x.bin", "map16x.bin", "manifest.json")}

    man = json.load(open(files["manifest.json"]))
    nations = man.get("nations", [])
    src_w, src_h = man["map"]["width"], man["map"]["height"]
    full = decode(load_raw(files["map.bin"]), src_w, src_h)
    full_arr = load_raw(files["map.bin"]).reshape(src_h, src_w)
    print(f"carte amont : {src_w}x{src_h}, {int(full['land'].sum()):,d} tuiles terrestres "
          f"({full['land'].mean()*100:.1f} %), {len(nations)} nations")

    # --- paquets "vraie Australie", du plus petit (entraînement) au plus grand (évaluation)
    wanted = [int(x) for x in args.sizes.split(",") if x.strip()]
    for width in wanted:
        if width == src_w:
            terrain = full_arr
            mini = downsample(full_arr, src_w, src_h, 2)
        elif abs(width - src_w / 2) <= 1:
            terrain = load_raw(files["map4x.bin"]).reshape(src_h // 2, src_w // 2)
            mini = load_raw(files["map16x.bin"]).reshape(src_h // 4, src_w // 4)
        elif abs(width - src_w / 4) <= 1:
            terrain = load_raw(files["map16x.bin"]).reshape(src_h // 4, src_w // 4)
            mini = downsample(terrain, src_w // 4, src_h // 4, 2)
        else:
            factor = max(1, round(src_w / width))
            terrain = downsample(full_arr, src_w, src_h, factor)
            mini = downsample(terrain, terrain.shape[1], terrain.shape[0], 2)
        h, w = terrain.shape
        write_pack(f"australia_real_{w}x{h}", terrain, mini, nations, src_w, src_h)

    # --- réparation des deux paquets cassés du dépôt (0 tuile terrestre : 100 % d'eau)
    if args.repair:
        for size in (500, 100):
            # Un paquet size x size doit contenir un VRAI carré de carte : on choisit le
            # facteur de réduction qui laisse au moins 2*size tuiles dans les DEUX axes
            # (sinon `crop` est tronqué et le paquet sort déformé), on recadre autour du
            # centre de la masse terrestre, puis on divise par 2.
            factor = max(1, min(src_w // (2 * size), src_h // (2 * size)))
            base = full_arr if factor == 1 else downsample(full_arr, src_w, src_h, factor)
            bw, bh = base.shape[1], base.shape[0]
            land_mask = (((base >> 7) & 1) == 1)
            ys, xs = np.nonzero(land_mask)
            cxl = int(np.median(xs)) if len(xs) else bw // 2
            cyl = int(np.median(ys)) if len(ys) else bh // 2
            sq = crop(base, bw, bh, cxl - size, cyl - size, size * 2)
            hh, ww = sq.shape
            if (ww, hh) != (size * 2, size * 2):
                print(f"  ⚠ {size}: recadrage {ww}x{hh} < {size*2}x{size*2} requis -> facteur {factor} trop fort")
            ox, oy = max(0, min(cxl - size, bw - ww)), max(0, min(cyl - size, bh - hh))
            small = downsample(sq, ww, hh, 2)
            terrain = add_shoreline(small.copy(), (((small >> 7) & 1) == 1))
            mini = downsample(terrain, terrain.shape[1], terrain.shape[0], 2)
            write_pack(f"australia_{size}x{size}", terrain, mini, nations,
                       bw * factor, bh * factor, ox * factor, oy * factor)
            print(f"     ↑ facteur {factor}, coin ({ox*factor},{oy*factor}) de la carte 2000x1500, "
                  f"centre masse terrestre -> {terrain.shape[1]}x{terrain.shape[0]}")

    print("\nVérifie l'environnement sur la nouvelle carte :")
    print("  python skai/diagnose.py --map australia_real_500x375 --fast")


if __name__ == "__main__":
    main()
