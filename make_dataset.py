from torchvision import transforms
from PIL import Image
import torch.utils.data as data
import glob
import numpy as np
import torch
from xml.etree import ElementTree
from torchvision.transforms import functional as F
import os


include_tags = ["blonde hair", "brown hair", "black hair", "blue hair",
            "pink hair", "purple hair", "green hair", "red hair",
            "silver hair", "white hair", "orange hair", "aqua hair",
            "grey hair", "long hair", "short hair", "twintails",
            "drill hair", "ponytail", "blush", "smile",
            "open mouth", "hat", "ribbon", "glasses",
            "blue eyes", "red eyes", "brown eyes", "green eyes",
            "purple eyes", "yellow eyes", "pink eyes", "aqua eyes",
            "black eyes", "orange eyes"]
tag2idx = {include_tags[i]:i for i in range(len(include_tags))}


def make_datapath_list():
    img_file_list = []
    xml_file_list = []
    # for img_path in glob.glob("./data/dataset2005/*"):
    #     img_file_list.append(img_path)
    # for xml_path in glob.glob("./data/xml2005/*"):
    #     xml_file_list.append(xml_path)
    for img_path in glob.glob("./data/_dataset2005/*"):
        img_file_list.append(img_path)
    for xml_path in glob.glob("./data/_xml2005/*"):
        xml_file_list.append(xml_path)
    img_file_list = sorted(img_file_list)
    xml_file_list = sorted(xml_file_list)
    return img_file_list, xml_file_list

class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((resize,resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, img):
        return self.data_transform(img)

def generate_random_vec(batch_size):
    idx1 = np.arange(batch_size)
    vec = np.zeros((batch_size, 34))
    idx2 = np.random.choice(range(0,13),batch_size)
    vec[idx1,idx2] = 1
    idx2 = np.random.choice(range(24,34))
    vec[idx1,idx2] = 1
    vec[:,13:23] += (np.random.rand(batch_size, 10) < 0.25)
    return torch.Tensor(vec)

def features2vec(features):
    return torch.Tensor([1 if include_tags[i] in features else 0 for i in range(34)])

def vec2features(vec):
    return [include_tags[i] for i in range(34) if vec[i] == 1]

class AnimeFaceDataset(data.Dataset):
    def __init__(self, img_file_list, xml_file_list, transform):
        self.img_file_list = img_file_list
        self.xml_file_list = xml_file_list
        self.transform = transform

    def color_transform(self, x):
        x = F.adjust_saturation(x, 2.5)
        x = F.adjust_gamma(x, 0.7)
        x = F.adjust_contrast(x, 1.2)
        return x
    
    def __len__(self):
        return len(self.img_file_list)
    
    def __getitem__(self, index):
        img_path = self.img_file_list[index]
        img = Image.open(img_path)
        if len(np.array(img).shape) == 2:
            img = Image.fromarray(np.repeat(np.array(img)[:,:,None],3,axis=2))
        elif np.array(img).shape[2] == 4:
            img = Image.fromarray(np.array(img)[:,:,0:3])

        img = self.color_transform(img)
        img = self.transform(img)

        xml_path = self.xml_file_list[index]
        root = ElementTree.parse(xml_path).getroot()
        features = [elem.text for elem in root[4]]
        vec = features2vec(features)
        
        return img, vec

def make_dataloader(batch_size=64, resize=128):
    img_file_list, xml_file_list = make_datapath_list()
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = ImageTransform(resize, mean, std)
    dataloader = data.DataLoader(AnimeFaceDataset(img_file_list, xml_file_list, transform),
        batch_size=batch_size, drop_last=True, shuffle=True, num_workers=os.cpu_count())
    return dataloader

def denorm(img):
    out = img * 0.5 + 0.5
    return out.clamp(0, 1)

if __name__=="__main__":
    dataloader = make_dataloader()
    batch_iterator = iter(dataloader)

    img, vec = next(batch_iterator)
    print(img.size())
    print(vec.size())

    img = denorm(img)

    import matplotlib.pyplot as plt
    print(vec2features(vec[0]))
    plt.imshow(img[0].detach().numpy().transpose(1,2,0))
    plt.show()
