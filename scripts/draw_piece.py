import os, requests, zipfile, io
from PIL import Image

def verify_and_fix_generator():
    # 1. THE FOLDER CHECK
    path = "piece_sets/cburnett/wP.png"
    if not os.path.exists(path):
        print(f"❌ PIECES MISSING: {path} not found.")
        print("📥 Attempting to download pieces for you...")
        os.makedirs("piece_sets/cburnett", exist_ok=True)
        # Using a reliable direct link to piece images
        r = requests.get("https://github.com/shubham-sharma/chess-images/raw/master/pieces/cburnett.zip")
        if r.status_code != 200:
            print("❌ Download failed. You must manually put piece images in piece_sets/cburnett/")
            return
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("piece_sets/cburnett")
        print("✅ Pieces downloaded.")

    # 2. THE RENDERING TEST
    try:
        test_piece = Image.open("piece_sets/cburnett/wP.png")
        canvas = Image.new("RGB", (64, 64), (128, 128, 128)) # Gray background
        canvas.paste(test_piece.resize((64,64)), (0,0), test_piece.convert("RGBA"))
        canvas.save("TEST_PIECE_REPLY.png")
        print("✅ SUCCESS: 'TEST_PIECE_REPLY.png' created. Open it.")
        print("If you see a White Pawn on a gray square, your generator is fixed.")
    except Exception as e:
        print(f"❌ RENDERING FAILED: {e}")

verify_and_fix_generator()

