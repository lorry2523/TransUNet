import torch
import numpy as np
import matplotlib.pyplot as plt
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from scipy.ndimage import zoom
from skimage import morphology   # 新增

# 参数设置（与训练时一致）
img_size = 512
vit_patches_size = 16
vit_name = 'R50-ViT-B_16'
num_classes = 2
n_skip = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = num_classes
config_vit.n_skip = n_skip
# === 关键修复：手动设置 grid，与训练脚本完全一致 ===
if vit_name.find('R50') != -1:
    config_vit.patches.grid = (img_size // vit_patches_size, img_size // vit_patches_size)

model = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes).to(device)

# 加载训练好的权重（消除 FutureWarning）
snapshot_path = 'G:/Strain_Project/model/TU_Synapse512/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs16_lr0.0001_512/best_model.pth'
checkpoint = torch.load(snapshot_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
# 加载测试样本（比如索引 0）
data_path = './data/Synapse/test_vol_h5/1.npz'
data = np.load(data_path)
image = data['image']      # (H,W,3)
label = data['label']      # (H,W)

# 预处理：缩放到模型输入尺寸
h, w = image.shape[:2]
if h != img_size or w != img_size:
    image_resized = zoom(image, (img_size/h, img_size/w, 1), order=3)
    label_resized = zoom(label, (img_size/h, img_size/w), order=0)
else:
    image_resized, label_resized = image, label

# 转换为 tensor 并推理
input_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

# 将预测结果缩放回原图尺寸（如果之前缩放过）
if h != img_size or w != img_size:
    pred = zoom(pred, (h/img_size, w/img_size), order=0)

# ------------------ 形态学闭运算 ------------------
def morphological_closing(mask, kernel_size=5):
    """应用形态学闭运算，kernel_size为结构元素直径（奇数）"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    selem = morphology.disk(kernel_size // 2)
    closed = morphology.binary_closing(mask, selem)
    return closed.astype(np.uint8)

pred_closed = morphological_closing(pred, kernel_size=5)   # 可调整核大小

# ------------------ 可视化（叠加显示） ------------------
def overlay_mask(image, mask, color=(1,0,0), alpha=0.5):
    """
    将 mask 以半透明颜色叠加到图像上
    image: (H,W,3) uint8
    mask: (H,W) 二值 0/1
    color: 叠加颜色 (R,G,B) 范围 0~1
    alpha: 透明度
    """
    img_norm = image.astype(np.float32) / 255.0
    mask_layer = np.zeros_like(img_norm)
    mask_layer[mask > 0] = color
    blended = (1 - alpha) * img_norm + alpha * mask_layer
    return blended

# 真实 mask 叠加
gt_overlay = overlay_mask(image, label, color=(0,1,0), alpha=0.5)   # 绿色
# 原始预测 mask 叠加
pred_overlay = overlay_mask(image, pred, color=(1,0,0), alpha=0.5)   # 红色
# 闭运算后预测 mask 叠加
pred_closed_overlay = overlay_mask(image, pred_closed, color=(1,0.5,0), alpha=0.5)   # 橙色

# 显示
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(gt_overlay)
plt.title('Ground Truth Overlay (Green)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(pred_overlay)
plt.title('Prediction Overlay (Red)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(pred_closed_overlay)
plt.title('Closed Prediction Overlay (Orange)')
plt.axis('off')

plt.tight_layout()
plt.show()