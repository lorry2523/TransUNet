import os
import cv2
import numpy as np
from tqdm import tqdm


def convert_split(images_dir, masks_dir, save_dir, txt_path):
    """
    将图像和mask转换为npz文件，并生成文件列表txt
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有图像文件（支持png/jpg）
    img_exts = ['*.png', '*.jpg', '*.jpeg']
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        raise FileNotFoundError(f'未找到图像文件: {images_dir}')

    file_names = []
    for idx, img_path in enumerate(tqdm(img_paths, desc=f'Converting {os.path.basename(save_dir)}')):
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f'跳过无法读取的图像: {img_path}')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 构造对应mask路径（假设文件名相同）
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]
        mask_path = os.path.join(masks_dir, base)
        if not os.path.exists(mask_path):
            # 尝试其他扩展名
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_path = os.path.join(masks_dir, name + ext)
                if os.path.exists(alt_path):
                    mask_path = alt_path
                    break
        if not os.path.exists(mask_path):
            print(f'跳过 {img_path}: 找不到对应mask')
            continue

        # 读取mask（灰度）
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f'跳过 {img_path}: 无法读取mask')
            continue

        # 确保mask是0和1的二值图像（如果原始是0/255则转换）
        unique = np.unique(mask)
        if np.array_equal(unique, [0, 255]):
            mask = (mask > 127).astype(np.uint8)
        elif not np.array_equal(unique, [0, 1]):
            print(f'警告: {mask_path} 包含异常值 {unique}，将二值化')
            mask = (mask > 0).astype(np.uint8)

        # 保存为npz
        save_path = os.path.join(save_dir, f'{idx}.npz')
        np.savez(save_path, image=image, label=mask)
        file_names.append(str(idx))

    # 写入txt列表
    with open(txt_path, 'w') as f:
        f.write('\n'.join(file_names))
    print(f'生成 {len(file_names)} 个样本 -> {save_dir}\n列表文件: {txt_path}')


if __name__ == '__main__':
    import glob  # 放在这里避免干扰

    # ===== 请根据你的实际路径修改 =====
    dataset_root = 'F:/Strain_Project/data'  # 你的数据集根目录（相对于项目根目录）

    # 训练集转换
    convert_split(
        images_dir=os.path.join(dataset_root, 'train', 'images'),
        masks_dir=os.path.join(dataset_root, 'train', 'masks'),
        save_dir='./data/Synapse/train_npz',
        txt_path='./lists/lists_Synapse/train.txt'
    )

    # 验证集转换（官方测试集使用test_vol.txt）
    convert_split(
        images_dir=os.path.join(dataset_root, 'val', 'images'),
        masks_dir=os.path.join(dataset_root, 'val', 'masks'),
        save_dir='./data/Synapse/test_vol_h5',
        txt_path='./lists/lists_Synapse/test_vol.txt'
    )