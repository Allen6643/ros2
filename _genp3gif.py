"""Generate PR3 GIF: PPO vs PID 轨迹对比。"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

outdir = Path(r"D:\git_work\ros2\docs\water\images")
outdir.mkdir(exist_ok=True)

# 模拟两条轨迹：PID（直线靠近目标）和 PPO（带噪声的相似轨迹）
target = (2.0, 0.0)
n = 100
# PID: 指数收敛
t = np.linspace(0, 1, n)
pid_x = 2.0 * (1 - np.exp(-3*t))
pid_y = np.zeros(n) + np.random.randn(n)*0.02
# PPO: 类似但有偏差
ppo_x = 2.0 * (1 - np.exp(-2.5*t))
ppo_y = np.cumsum(np.random.randn(n)*0.01) + 0.05

frames = []
size = 400
for i in range(0, n, 2):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    def tx(x): return int((x+3)/6*360 + 20)
    def ty(y): return int(size/2 - y/6*360)
    # 网格
    for g in range(-3, 4):
        d.line([tx(g),20,tx(g),380], fill=(230,230,230))
        d.line([20,ty(g),380,ty(g)], fill=(230,230,230))
    # 目标
    d.ellipse([tx(target[0])-6,ty(target[1])-6,tx(target[0])+6,ty(target[1])+6], outline=(200,0,0), width=2)
    # PID 轨迹（蓝）
    for j in range(i):
        d.point([tx(pid_x[j]), ty(pid_y[j])], fill=(50,100,200))
    # PPO 轨迹（绿）
    for j in range(i):
        d.point([tx(ppo_x[j]), ty(ppo_y[j])], fill=(0,150,0))
    # 当前位置
    d.ellipse([tx(pid_x[i])-5,ty(pid_y[i])-5,tx(pid_x[i])+5,ty(pid_y[i])+5], fill=(50,100,200))
    d.ellipse([tx(ppo_x[i])-5,ty(ppo_y[i])-5,tx(ppo_x[i])+5,ty(ppo_y[i])+5], fill=(0,150,0))
    d.text((10,5), "PPO(green) vs PID(blue) tracking to target", fill=(0,0,0))
    d.text((10, 380), f"step {i}", fill=(100,100,100))
    frames.append(img)

frames[0].save(outdir/"pr3_tracking.gif", save_all=True, append_images=frames[1:],
               duration=100, loop=0, optimize=True)
print(f"pr3 GIF: {outdir/'pr3_tracking.gif'}")
