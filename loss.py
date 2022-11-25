import torch
import torch.nn as nn

def generate_loss(device) :
    # BCELoss 함수의 인스턴스를 생성합니다
    criterion = nn.BCELoss()

    # 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
    fixed_noise = torch.randn(512, 100, 1, 1, device=device)

    # 학습에 사용되는 참/거짓의 라벨을 정합니다
    real_label = 1.
    fake_label = 0.

    # G와 D에서 사용할 Adam옵티마이저를 생성합니다
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))