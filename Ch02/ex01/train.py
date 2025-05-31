import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm

from utils import get_grid_image, print_image

def train(
        G,
        D, 
        opt_G,
        opt_D,
        z_size,
        fixed,
        dataloader,
        device,
        num_epochs):
    
    img_list = []
    for epoch in tqdm(range(num_epochs)):
        
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            real_label = torch.FloatTensor(real_imgs.size(0),1).fill_(1.0).to(device)
            fake_label = torch.FloatTensor(real_imgs.size(0),1).fill_(0.0).to(device)
            
            opt_G.zero_grad()
            z = torch.normal(0,1,size=(real_imgs.size(0), z_size, 1, 1)).to(device)
            fake_imgs = G(z)
            g_loss = F.binary_cross_entropy(D(fake_imgs), real_label)
            g_loss.backward()
            opt_G.step()

            opt_D.zero_grad()
            real_loss = F.binary_cross_entropy(D(real_imgs), real_label)
            fake_loss = F.binary_cross_entropy(D(fake_imgs.detach()), fake_label)
            d_loss = (real_loss+fake_loss) / 2
            d_loss.backward()
            opt_D.step()

        # print loss each epoch
        print(f'[Epoch {epoch+1:3d}/{num_epochs:3d}] [G_loss {g_loss.item():2.4f}] [D_loss {d_loss.item():2.4f}]')
        # save fixed image
        img_list.append(get_grid_image(G(fixed)))
        # print sample images
        if epoch % 20 == 0:
            print_image(G(fixed))

    # save result

    torch.save({
        'G':G.state_dict(),
        'D':D.state_dict(),
        'fixed':fixed
        }, f'./DCGAN-model-flower102-E{num_epochs}.pt')

    np.savez_compressed(f'./DCGAN-images-flower102-E{num_epochs}.npz', np.array(img_list))