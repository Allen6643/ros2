"""Generate GIF demos for PR1 and PR2."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

outdir = Path(r"D:\git_work\ros2\docs\water\images")
outdir.mkdir(exist_ok=True)

# ============ PR1 GIF: ROV 沿轨迹巡航动画 ============
# 模拟 ROV 从 (0,0) 出发绕航点
waypoints = np.array([[2, 0], [0, 2], [-2, 0], [0, -2], [2, 0]])
# 生成平滑轨迹
t = np.linspace(0, 1, 200)
traj_x = np.interp(t, np.linspace(0,1,len(waypoints)), waypoints[:,0])
traj_y = np.interp(t, np.linspace(0,1,len(waypoints)), waypoints[:,1])

frames = []
size = 400
for i in range(0, len(traj_x), 4):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    # 坐标变换: [-3,3] -> [20, 380]
    def tx(x): return int((x+3)/6*360 + 20)
    def ty(y): return int(size/2 - y/6*360)
    # 网格
    for gx in range(-3, 4):
        d.line([tx(gx),20,tx(gx),380], fill=(230,230,230))
    for gy in range(-3, 4):
        d.line([20,ty(gy),380,ty(gy)], fill=(230,230,230))
    # 已走过的轨迹
    for j in range(i):
        d.point([tx(traj_x[j]), ty(traj_y[j])], fill=(50,100,200))
    # 航点
    for wx, wy in waypoints:
        d.ellipse([tx(wx)-6,ty(wy)-6,tx(wx)+6,ty(wy)+6], outline=(200,0,0), width=2)
    # 当前 ROV 位置
    d.ellipse([tx(traj_x[i])-8,ty(traj_y[i])-8,tx(traj_x[i])+8,ty(traj_y[i])+8],
              fill=(0,150,0), outline=(0,100,0))
    # 标题
    d.text((10,5), f"ROV waypoint navigation  step {i}", fill=(0,0,0))
    frames.append(img)

frames[0].save(outdir/"pr1_trajectory.gif", save_all=True, append_images=frames[1:],
               duration=80, loop=0, optimize=True)
print(f"pr1 GIF: {outdir/'pr1_trajectory.gif'}")

# ============ PR2 GIF: CNN 声呐分类动画 ============
# 三类声呐图，循环显示并标注分类结果
def make_sonar_img(rock_int, pipe_int, wreck_int):
    s = 64
    img = np.zeros((s,s), dtype=np.float32)
    def tpx(x): return int((x+5)/10*s)
    def tpy(y): return int((y+5)/10*s)
    def blobs(pos, inten, sig):
        for px,py in pos:
            cx,cy = tpx(px),tpy(py)
            for dy in range(-sig*2,sig*2+1):
                for dx in range(-sig*2,sig*2+1):
                    x,y = cx+dx,cy+dy
                    if 0<=x<s and 0<=y<s:
                        img[y,x] += inten*np.exp(-(dx*dx+dy*dy)/(2*sig*sig))
    blobs([[1.5,1.5],[-2.0,1.0]], rock_int, 4)
    blobs([[2.5,-1.5]], pipe_int, 2)
    blobs([[0.5,-2.5]], wreck_int, 3)
    img += np.random.randn(s,s)*0.05
    return (np.clip(img,0,1)*255).astype(np.uint8)

samples = [
    (0.8, 0.15, 0.12, "rock (99.9%)"),
    (0.15, 0.9, 0.12, "pipe (100%)"),
    (0.15, 0.15, 0.6, "wreck (100%)"),
]
frames2 = []
for rock_i, pipe_i, wreck_i, label in samples:
    sonar = make_sonar_img(rock_i, pipe_i, wreck_i)
    # 放大到 256x256
    pil = Image.fromarray(sonar).resize((256,256), Image.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (320, 300), "white")
    canvas.paste(pil, (32, 20))
    d = ImageDraw.Draw(canvas)
    d.text((32, 270), f"CNN -> {label}", fill=(0,0,200))
    d.text((32, 2), "side-scan sonar", fill=(100,100,100))
    for _ in range(15):  # 每类停留
        frames2.append(canvas)

frames2[0].save(outdir/"pr2_classify.gif", save_all=True, append_images=frames2[1:],
                duration=300, loop=0, optimize=True)
print(f"pr2 GIF: {outdir/'pr2_classify.gif'}")
