import torch
from torchvision.transforms import transforms as T

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

def print_image_from_tensor(x):
    if len(x.shape) == 3:
        a, b, c = x.shape
        x = x.reshape(1, a, b, c)

    x = np.transpose(x, (0, 2, 3, 1))
    x_concat = np.hstack([x[i] for i in range(x.shape[0])])
    plt.imshow(x_concat, cmap="gray")
    plt.axis("off")
    plt.show()

def calc_accuracy(y_pred, y_target):
    y_pred = y_pred.detach().cpu()
    y_target = y_target.detach().cpu()
    return (y_pred.argmax(dim=-1) == y_target).float().mean().item()

def print_image_from_path(path):
    img = Image.open(path)
    img_array = np.array(img)
    plt.imshow(img_array)
    plt.axis("off")
    plt.show()

def image_to_tensor(path):
    img = Image.open(path)
    img_tensor = 1 - T.Compose([
        T.Resize((20, 20)),
        T.Grayscale(1),
        T.ToTensor()
    ])(img)
    return torch.where(img_tensor <= 0.1, torch.tensor(0.0), img_tensor)
