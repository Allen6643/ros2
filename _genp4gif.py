"""Generate PR4 GIF: ROV 逼近对接站动画。"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

outdir = Path(r"D:\git_work\ros2\docs\water\images")
outdir.mkdir(exist_ok=True)

dock = (3.0, 0.0)
obs = [(1.5, 1.5), (1.5, -1.5)]
frames = []
size = 400
for i in range(0, 100, 2):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    def tx(x): return int((x+3)/6*360 + 20)
    def ty(y): return int(size/2 - y/6*360)
    for g in range(-3, 4):
        d.line([tx(g),20,tx(g),380], fill=(230,230,230))
        d.line([20,ty(g),380,ty(g)], fill=(230,230,230))
    # 障碍物
    for ox, oy in obs:
        d.rectangle([tx(ox)-8,ty(oy)-8,tx(ox)+8,ty(oy)+8], fill=(100,100,100))
    # 对接站
    d.ellipse([tx(dock[0])-10,ty(dock[1])-10,tx(dock[0])+10,ty(dock[1])+10], fill=(255,200,0), outline=(200,150,0))
    d.text((tx(dock[0])-10, ty(dock[1])-25), "DOCK", fill=(200,150,0))
    # ROV 从 (0,0) 走到对接站
    t = i/100
    rov_x = 3.0 * (1 - np.exp(-3*t))
    rov_y = 0.0
    d.ellipse([tx(rov_x)-6,ty(rov_y)-6,tx(rov_x)+6,ty(rov_y)+6], fill=(0,100,200))
    d.text((10,5), f"ROV docking approach  step {i}", fill=(0,0,0))
    frames.append(img)

frames[0].save(outdir/"pr4_docking.gif", save_all=True, append_images=frames[1:],
               duration=100, loop=0, optimize=True)
print(f"pr4 GIF: {outdir/'pr4_docking.gif'}")
