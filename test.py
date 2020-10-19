import torch
from torchvision import utils
from PIL import Image
import matplotlib.pyplot as plt

from make_dataset import *
import i2v
from generate_image import generate_image

import time
start = time.time()

dummy_tags = ["blue hair", "blue eyes", "long hair"]
dummy_vec = features2vec(dummy_tags)
batch_size = 10

# illust2vec読み込み
illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

print(time.time()-start)

for i in range(10):
    # 画像生成
    output = generate_image(batch_size, dummy_vec)
    output = denorm(output)
    images = (output.detach().numpy().transpose(0,2,3,1)*255).astype(np.uint8)
    images = [Image.fromarray(img) for img in images]

    # 画像選択
    out = illust2vec.estimate_plausible_tags(images, threshold=0.25)
    tag_list = [[tag for tag, prob in o["general"]] for o in out]
    tag_list = [[tag for tag in tags if tag in include_tags] for tags in tag_list]
    print(*tag_list,sep="\n")
    tag_vec = torch.cat([features2vec(tags).view(1,-1) for tags in tag_list], dim=0)
    score = (tag_vec * dummy_vec).sum(dim=1)
    print(score.size())
    print(score)
    # tags = [a for a,b in out if a in include_tags]
    grid = utils.make_grid(output, nrow=5)

    print(time.time()-start)

    plt.imshow(grid.detach().numpy().transpose(1,2,0))
    plt.show()

    idx = np.argmax(score.detach().numpy())
    if score[idx] >= len(dummy_tags)-1:
        plt.imshow(output[idx].detach().numpy().transpose(1,2,0))
        plt.show()
        break
    else:
        print("fail")
        continue
