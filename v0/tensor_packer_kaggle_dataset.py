#!/usr/bin/env python3
"""
📦 TENSOR PACKER BEAST (SAFE SPLIT VERSION)
- Groups tiles by BOARD to prevent data leakage.
- Shuffles boards, not individual tiles.
- Saves separate Train and Validation chunks.
"""
import os
import glob
import time
import torch
import numpy as np
from PIL import Image
import concurrent.futures
from datetime import timedelta
import argparse

# ============ ARGUMENT PARSING ============
parser = argparse.ArgumentParser(description='Tensor Packer Beast (Safe Split)')
parser.add_argument('--input-dir', type=str, help='Path to tiles_real directory')
parser.add_argument('--train-chunks', type=int, default=8, help='Number of training chunks')
parser.add_argument('--val-chunks', type=int, default=2, help='Number of validation chunks')
args = parser.parse_args()

# ============ CONFIG ============
FEN_CHARS = "1RNBQKPrnbqkp"
USE_GRAYSCALE = True
NUM_CPU_WORKERS = os.cpu_count() or 4

# ============ PATHS ============
base_dir = os.path.dirname(os.path.abspath(__file__))
tiles_dir = args.input_dir if args.input_dir else os.path.join(base_dir, "images", "tiles_real")
if not os.path.exists(tiles_dir) and not args.input_dir:
    tiles_dir = os.path.abspath(os.path.join(base_dir, "..", "images", "tiles_real"))

output_dir = os.path.join(base_dir, "tensor_dataset_safe")
os.makedirs(output_dir, exist_ok=True)

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def process_single_image(path):
    piece_type = path[-5]
    if piece_type not in FEN_CHARS:
        return None
    try:
        img = Image.open(path).convert("L" if USE_GRAYSCALE else "RGB")
        img = img.resize((32, 32))
        return np.array(img, dtype=np.uint8), FEN_CHARS.index(piece_type)
    except:
        return None

def pack_and_save(chunk_name, paths):
    print(f"📦 Packing {chunk_name} ({len(paths):,} images)...")
    start = time.time()
    
    imgs, lbls = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
        results = list(executor.map(process_single_image, paths))
        
    for res in results:
        if res is not None:
            imgs.append(res[0])
            lbls.append(res[1])
            
    if not imgs: return

    x = torch.from_numpy(np.stack(imgs))
    if USE_GRAYSCALE: x = x.unsqueeze(1)
    else: x = x.permute(0, 3, 1, 2)
    y = torch.tensor(lbls, dtype=torch.uint8)
    
    chunk_path = os.path.join(output_dir, f"{chunk_name}.pt")
    torch.save({'x': x, 'y': y}, chunk_path)
    print(f"✅ Saved {chunk_path} | {time.time()-start:.1f}s")

def main():
    print("🚀 TENSOR PACKER BEAST (SAFE SPLIT) - INITIALIZING")
    print("="*60)
    
    board_dirs = sorted([d for d in glob.glob(os.path.join(tiles_dir, "*")) if os.path.isdir(d)])
    if not board_dirs:
        print(f"❌ No boards found in {tiles_dir}")
        return
        
    print(f"📊 Found {len(board_dirs):,} board directories.")
    
    # Shuffle BOARDS
    np.random.seed(42)
    np.random.shuffle(board_dirs)
    
    # Split BOARDS
    split_idx = int(0.9 * len(board_dirs))
    train_boards = board_dirs[:split_idx]
    val_boards = board_dirs[split_idx:]
    
    print(f"📈 Split: {len(train_boards):,} Train boards, {len(val_boards):,} Val boards.")

    def get_all_tiles(dirs):
        paths = []
        for d in dirs: paths.extend(glob.glob(os.path.join(d, "*.png")))
        return paths

    # Process Validation
    print("\n🔍 Preparing Validation Set...")
    val_paths = get_all_tiles(val_boards)
    v_size = (len(val_paths) + args.val_chunks - 1) // args.val_chunks
    for i in range(args.val_chunks):
        p = val_paths[i*v_size : (i+1)*v_size]
        if p: pack_and_save(f"val_chunk_{i:02d}", p)

    # Process Training
    print("\n🔍 Preparing Training Set...")
    train_paths = get_all_tiles(train_boards)
    t_size = (len(train_paths) + args.train_chunks - 1) // args.train_chunks
    for i in range(args.train_chunks):
        p = train_paths[i*t_size : (i+1)*t_size]
        if p: pack_and_save(f"train_chunk_{i:02d}", p)
        
    print("="*60)
    print(f"🎉 SAFE PACKING COMPLETE! Output: {output_dir}")

if __name__ == "__main__":
    main()
