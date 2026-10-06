import torch
import torch.nn as nn

class SubmarineEnd2EndNet(nn.Module):
    """
    水下机器人端到端感知控制网络
    输入：水下环境RGB图像(3,120,160)
    输出：4个推进器的控制信号[-1,1]，分别控制前后、上下、左右姿态
    """
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*13*18,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,4),
            nn.Tanh()   #Tanh输出区间(-1,1)对应推进器正负
        )

    def forward(self,img_tensor):
        feat = self.cnn(img_tensor)
        out = self.fc(feat)
        return out


if __name__=="__main__":
    net = SubmarineEnd2EndNet()
    test_img = torch.randn((1,3,120,160))
    res = net(test_img)
    print("推进器控制输出：",res)
