import argparse
import os
import subprocess
import urllib.request
from pathlib import Path

try:
    import cairosvg
except ImportError:
    cairosvg = None


PIECE_CODES = ["wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]


def _svg_to_png(svg_path: Path, dest_png: Path, size: int = 128) -> None:
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    if cairosvg is not None:
        try:
            cairosvg.svg2png(
                url=svg_path.resolve().as_uri(),
                write_to=str(dest_png),
                output_width=size,
                output_height=size,
            )
            return
        except Exception:
            pass

    subprocess.run(
        [
            "rsvg-convert",
            "-w",
            str(size),
            "-h",
            str(size),
            str(svg_path),
            "-o",
            str(dest_png),
        ],
        check=True,
    )

def setup_lichess_assets():
    if cairosvg is None:
        raise RuntimeError("setup_lichess_assets requires cairosvg. Use --local-only to convert local SVG assets.")
    # 1. PIECE SETTINGS
    piece_base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/"
    #piece_base_url = "https://raw.githubusercontent.com/pychess/pychess/master/pieces/"
    sets = [
        'alpha', 'cburnett', 'merida', 'california', 'cardinal', 'gioco', 'dubrovny', 'chessnut', 'fantasy', 'tatiana',
        'caliente', 'celtic', 'companion', 'cooke', 'dubrovny', 'governor' , 'maestro', 'staunty', 'fresca', 'kosal', 'mpchess',
        'chess7', 'firi', 'icpieces', 'pirouetti', 'rhosgfx', 'riohacha', 'spatial', 'xkcd',
        'chessicons', 'chessmonk', 'libra', 'magnetic', 'regular'
    ]
    pieces = PIECE_CODES
    #pieces = ['wp','wn','wb','wr','wq','wk','bp','bn','bb','br','bq','bk']
    
    # 2. BOARD SETTINGS (Exact filenames from Lichess Repo)
    board_base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/images/board/"
    boards = [
        'blue.png', 'blue2.jpg', 'blue3.jpg', 'canvas2.jpg', 'green.png', 'grey.jpg', 'green-plastic.png', 'brown.png',
        'leather.jpg', 'marble.jpg', 'metal.jpg', 'olive.jpg', 'purple.png', 'leather.jpg', 'horsey.jpg', 'ic.png',
        'wood.jpg', 'wood2.jpg', 'wood3.jpg', 'wood4.jpg', 'maple.jpg', 'maple2.jpg', 'pink-pyramid.png'
    ]

    os.makedirs("piece_sets", exist_ok=True)
    os.makedirs("board_themes", exist_ok=True)
    
    # Download Pieces (SVG -> PNG 128x128)
    for s in sets:
        os.makedirs(f"piece_sets/{s}", exist_ok=True)
        print(f"🚀 Fetching Lichess '{s}' Pieces...")
        for p in pieces:
            url = f"{piece_base_url}{s}/{p}.svg"
            dest_png = f"piece_sets/{s}/{p}.png"
            try:
                cairosvg.svg2png(url=url, write_to=dest_png, output_width=128, output_height=128)
            except:
                pass

    # Download Boards
    print("\n🎨 Fetching Official Lichess Board Textures...")
    for b in boards:
        url = f"{board_base_url}{b}"
        dest = f"board_themes/{b}"
        try:
            # Use a User-Agent to prevent GitHub blocks
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest, 'wb') as f:
                f.write(response.read())
            print(f"  ✅ {b} installed.")
        except:
            print(f"  ❌ Failed Board: {b}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Build piece_sets from local ./pieces SVG directories instead of downloading assets.",
    )
    parser.add_argument(
        "--sets",
        default="",
        help="Comma-separated local set names to convert from ./pieces into ./piece_sets.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Output PNG size for local set conversion.",
    )
    args = parser.parse_args()

    if args.local_only:
        root = Path(__file__).resolve().parents[1]
        source_root = root / "pieces"
        dest_root = root / "piece_sets"
        requested_sets = [s.strip() for s in args.sets.split(",") if s.strip()]
        converted = []
        for set_dir in sorted(source_root.iterdir()):
            if not set_dir.is_dir():
                continue
            if requested_sets and set_dir.name not in requested_sets:
                continue
            if not all((set_dir / f"{piece}.svg").exists() for piece in PIECE_CODES):
                continue
            for piece in PIECE_CODES:
                _svg_to_png(set_dir / f"{piece}.svg", dest_root / set_dir.name / f"{piece}.png", size=args.size)
            converted.append(set_dir.name)
        print("Converted local piece sets:")
        for name in converted:
            print(f"  piece_sets/{name}/")
    else:
        setup_lichess_assets()
