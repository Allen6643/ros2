#!/usr/bin/env python3
"""生成 PR2 演示图：声呐样例图 + CNN 分类结果"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ros2_ws/src/water/mjc_rov_sonar_cnn"))
os.environ["MUJOCO_GL"]="osmesa"; os.environ["PYOPENGL_PLATFORM"]="osmesa"
import numpy as np
import cv2
import torch
from mjc_rov_sonar_cnn import common
from mjc_rov_sonar_cnn.cnn_model import SonarCNN
CLASS_NAMES = ["rock", "pipe", "wreck"]

for label, name in enumerate(["礁石为主", "管道为主", "残骸为主"]):
    rock_int = 0.8 if label == 0 else 0.15
    pipe_int = 0.9 if label == 1 else 0.15
    wreck_int = 0.6 if label == 2 else 0.12
    img = common.generate_sonar_image(
        np.array([0.0, 0.0]),
        rock_pos=[[1.5,1.5],[-2.0,1.0]],
        pipe_pos=[[2.5,-1.5]],
        wreck_pos=[[0.5,-2.5]],
        size=64, rock_intensity=rock_int, pipe_intensity=pipe_int, wreck_intensity=wreck_int,
    )
    cv2.imwrite(f"/tmp/pr2_sample_{label}.png", img)

net = SonarCNN()
wpath = os.path.expanduser("~/ros2_ws/src/water/mjc_rov_sonar_cnn/weights/sonar_cnn.pt")
net.load_state_dict(torch.load(wpath, map_location="cpu"))
net.eval()
for label in range(3):
    img = cv2.imread(f"/tmp/pr2_sample_{label}.png", cv2.IMREAD_GRAYSCALE)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = net(x)
        pred = out.argmax(dim=1).item()
        probs = torch.softmax(out, dim=1).squeeze().tolist()
    print(f"true={label} -> pred={pred} ({CLASS_NAMES[pred]}) probs={[round(p,3) for p in probs]}")
print("done")
