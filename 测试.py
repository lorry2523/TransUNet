import numpy as np
data = np.load('./model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz')
keys = list(data.keys())
print(f"总共有 {len(keys)} 个键")
print("前20个键:", keys[:20])