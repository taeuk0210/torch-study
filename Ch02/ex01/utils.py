import torch
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

from celluloid import Camera

def make_video(img_load):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    for i in range(img_load.shape[0]):
        ax.axis('off')
        ax.imshow(img_load[i])
        camera.snap()
    # interval=500 (0.5초), repeat : 반복여부
    # interval을 3/200 = 0.015 초로 
    animation = camera.animate(interval=30, repeat=False)
    animation.save('./sample.mp4')

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
