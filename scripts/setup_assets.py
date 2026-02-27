import os, urllib.request, cairosvg

def setup_lichess_assets():
    # 1. PIECE SETTINGS
    piece_base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/"
    sets = [
        'alpha', 'cburnett', 'merida', 'california', 'cardinal', 'gioco', 'dubrovny', 'chessnut', 'fantasy', 'tatiana',
        'caliente', 'celtic', 'companion', 'cooke', 'dubrovny', 'governor' , 'maestro', 'staunty', 'fresca', 'kosal', 'mpchess',
        'chess7', 'firi', 'icpieces', 'pirouetti', 'rhosgfx', 'riohacha', 'spatial', 'xkcd'
    ]
    pieces = ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']
    
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
    setup_lichess_assets()

