import torch
from torchvision.utils import make_grid

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

def show_interact_image(imgs, transpose=True):
    # 이미지 출력 함수
    def show_image(index):
        
        img = imgs[index].cpu()
        if transpose:
            img = (img + 1) / 2  # [-1, 1] → [0, 1] 정규화
            img = np.transpose(img, (1,2,0))  # (C, H, W) → (H, W, C)

        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Generated Image #{index}")
        plt.show()

    # 슬라이더 위젯 만들기
    slider = widgets.IntSlider(value=0, min=0, max=len(imgs)-1, step=1, description='index:')
    widgets.interact(show_image, index=slider)

def get_grid_image(imgs):
    imgs = imgs.detach().cpu()
    imgs = np.transpose(make_grid(imgs,nrow=10,normalize=True),(1,2,0))
    return imgs

def print_image(imgs):
    with torch.no_grad():
        imgs = get_grid_image(imgs)
        plt.axis('off')
        plt.imshow(imgs)
        plt.show()
