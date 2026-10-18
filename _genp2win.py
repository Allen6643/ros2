"""Generate PR2 sample sonar images on Windows (no VM needed)."""
import numpy as np
from PIL import Image
from pathlib import Path

def generate(rock_int, pipe_int, wreck_int, out):
    size = 64
    img = np.zeros((size, size), dtype=np.float32)
    def to_px(x, y):
        return int((x+5)/10*size), int((y+5)/10*size)
    def add_blobs(pos_list, intensity, sigma):
        for (px, py) in pos_list:
            cx, cy = to_px(px, py)
            for dy in range(-sigma*2, sigma*2+1):
                for dx in range(-sigma*2, sigma*2+1):
                    x, y = cx+dx, cy+dy
                    if 0 <= x < size and 0 <= y < size:
                        img[y,x] += intensity * np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))
    add_blobs([[1.5,1.5],[-2.0,1.0]], rock_int, 4)
    add_blobs([[2.5,-1.5]], pipe_int, 2)
    add_blobs([[0.5,-2.5]], wreck_int, 3)
    img += np.random.randn(size,size)*0.05
    img = np.clip(img,0,1)
    Image.fromarray((img*255).astype(np.uint8)).save(out)
    print(f"saved {out}")

outdir = Path(r"D:\git_work\ros2\docs\water\images")
outdir.mkdir(exist_ok=True)
generate(0.8, 0.15, 0.12, str(outdir/"pr2_rock.png"))
generate(0.15, 0.9, 0.12, str(outdir/"pr2_pipe.png"))
generate(0.15, 0.15, 0.6, str(outdir/"pr2_wreck.png"))
