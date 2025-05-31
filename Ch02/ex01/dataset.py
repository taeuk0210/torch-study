from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from torchvision.transforms import transforms as T

class MyDataset(Dataset):
    def __init__(self, img_size=64):
        super(MyDataset, self).__init__()
        transform = T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train = Flowers102("./train", split="train", transform=transform, download=True)
        self.valid = Flowers102("./valid", split="val", transform=transform, download=True)
        self.test = Flowers102("./test", split="test", transform=transform, download=True)
        self.train_len = self.train.__len__()
        self.valid_len = self.valid.__len__()
        self.test_len = self.test.__len__()

    def __getitem__(self, index):
        if (index < self.train_len):
            return self.train[index][0]
        elif (index < self.train_len + self.valid_len):
            return self.valid[index - self.train_len][0]
        else:
            return self.test[index - self.train_len - self.valid_len][0]

    def __len__(self):
        return self.train_len + self.valid_len + self.test_len