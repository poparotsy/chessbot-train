#!/usr/bin/env python3
"""Generate 5 sample boards to visually verify augmentation"""
import os, io, random, chess, torch, numpy as np
from PIL import Image, ImageDraw
import math

# Import from generate script
exec(open('generate_hybrid_v3.py').read())

print("Generating 5 sample boards with augmentation...\n")

for i in range(10):
    # Generate random position
    b = chess.Board()
    for _ in range(random.randint(10, 30)):
        if not b.is_game_over(): 
            b.push(random.choice(list(b.legal_moves)))
    
    fen = b.fen().split()[0]
    
    # Render board (this applies all augmentation)
    board_file = random.choice(os.listdir("board_themes"))
    background = Image.open(f"board_themes/{board_file}").convert("RGB").resize((512, 512))
    p_set = random.choice(os.listdir("piece_sets"))
    ts = 64
    
    grid = [[None]*8 for _ in range(8)]
    for r, row_str in enumerate(fen.split('/')):
        c = 0
        for char in row_str:
            if char.isdigit(): c += int(char)
            else: grid[r][c] = char; c += 1

    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char:
                p_name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
                p_img = Image.open(f"piece_sets/{p_set}/{p_name}").convert("RGBA").resize((ts, ts))
                background.paste(p_img, (c*ts, r*ts), p_img)

    # Add coordinate labels - randomize to test variety
    label_choice = random.random()
    if label_choice > 0.4:  # 60% chance of having labels
        draw = ImageDraw.Draw(background)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = None
        
        # File labels (a-h) at bottom
        for j, letter in enumerate('abcdefgh'):
            x = j * ts + ts - 12
            y = 512 - 16
            draw.text((x, y), letter, fill=(128, 128, 128, 180), font=font)
        
        # Rank labels on left or right
        if label_choice > 0.7:  # 30% on left
            x = 4
        else:  # 30% on right
            x = 512 - 14
        
        for j in range(8):
            y = j * ts + 4
            draw.text((x, y), str(8-j), fill=(128, 128, 128, 180), font=font)
    # 40% have no labels

    # Apply vandalization (arrows, highlights)
    background = vandalize(background)
    
    # Apply augmentation
    background = augment_image(background)
    
    # JPEG compression
    if random.random() > 0.3:
        buf = io.BytesIO()
        background.save(buf, "JPEG", quality=random.randint(30, 90))
        background = Image.open(buf).copy()
    
    background.save(f"sample_board_{i+1}.png")
    print(f"✅ Saved sample_board_{i+1}.png")

print("\n📊 Check sample_board_*.png files to verify:")
print("   - Arrows with arrowheads")
print("   - Square highlights")
print("   - Coordinate labels (a-h, 1-8)")
print("   - Mix of color and grayscale")
print("   - Noise/compression artifacts")
