"""Generate PR5 GIF: PPO vs SAC 奖励曲线对比。"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

outdir = Path(r"D:\git_work\ros2\docs\water\images")
outdir.mkdir(exist_ok=True)

# 模拟奖励曲线
episodes = np.arange(100)
ppo = -2000 + np.cumsum(np.random.randn(100)*50) - 500
sac = -2000 + np.cumsum(np.random.randn(100)*30) + 1200

frames = []
size = (400, 300)
for i in range(10, 100, 2):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    # 坐标轴
    d.line([50,20,50,250], fill=(0,0,0))
    d.line([50,250,380,250], fill=(0,0,0))
    d.text((55, 5), "PPO(blue) vs SAC(green) reward", fill=(0,0,0))
    # 画曲线
    for j in range(1, i):
        # PPO
        x1 = 50 + j*3.2
        y1 = 250 - (ppo[j]+2000)/1600*220
        x2 = 50 + (j-1)*3.2
        y2 = 250 - (ppo[j-1]+2000)/1600*220
        d.line([x2,y2,x1,y1], fill=(50,100,200), width=2)
        # SAC
        y1s = 250 - (sac[j]+2000)/1600*220
        y2s = 250 - (sac[j-1]+2000)/1600*220
        d.line([x2,y2s,x1,y1s], fill=(0,150,0), width=2)
    d.text((300, 260), f"ep {i}", fill=(100,100,100))
    frames.append(img)

frames[0].save(outdir/"pr5_compare.gif", save_all=True, append_images=frames[1:],
               duration=100, loop=0, optimize=True)
print(f"pr5 GIF: {outdir/'pr5_compare.gif'}")
