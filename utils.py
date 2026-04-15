import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 移除 batch 维度
    if isinstance(image, torch.Tensor):
        image = image.squeeze(0).cpu().detach().numpy()
        label = label.squeeze(0).cpu().detach().numpy()
    else:
        image = image.squeeze(0) if image.ndim == 4 else image
        label = label.squeeze(0) if label.ndim == 4 else label

    # 获取原始尺寸
    orig_h, orig_w = image.shape[:2]

    # 图像缩放至模型输入尺寸
    if orig_h != patch_size[0] or orig_w != patch_size[1]:
        if image.ndim == 3:  # 彩色
            image_resized = zoom(image, (patch_size[0]/orig_h, patch_size[1]/orig_w, 1), order=3)
        else:  # 灰度
            image_resized = zoom(image, (patch_size[0]/orig_h, patch_size[1]/orig_w), order=3)
    else:
        image_resized = image

    # 转换为模型输入格式
    if image_resized.ndim == 3:  # 彩色
        input_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()
    else:  # 灰度
        input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        out = net(input_tensor)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0).cpu().numpy()

    # 将预测结果上采样回原始尺寸
    if orig_h != patch_size[0] or orig_w != patch_size[1]:
        pred = zoom(pred, (orig_h / patch_size[0], orig_w / patch_size[1]), order=0)

    # 计算指标（与原始相同）
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(pred == i, label == i))
    return metric_list


def compute_metrics(pred, label, num_classes=2):
    """
    计算前景类（类别1）的评估指标
    Args:
        pred: 模型预测标签 (B, H, W) 或 (H, W)，取值0或1
        label: 真实标签 (B, H, W) 或 (H, W)
        num_classes: 总类别数，默认2（背景0，前景1）
    Returns:
        iou, dice, pa: 前景类的交并比、Dice系数、像素精度
    """
    # 确保是 PyTorch Tensor 且在 CPU 上计算
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)

    pred = pred.flatten().long()
    label = label.flatten().long()

    # 混淆矩阵计算（只取有效的类别范围）
    mask = (label >= 0) & (label < num_classes)
    pred, label = pred[mask], label[mask]

    cm = torch.bincount(num_classes * label + pred, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    tp = cm[1, 1]
    fp = cm[0, 1] + cm[1, 0]  # 注意：对于二分类，背景误分为前景和前景误分为背景都算FP
    fn = cm[1, 0]

    # 添加极小值防止除零
    smooth = 1e-6
    iou = tp / (tp + fp + fn + smooth)
    dice = (2 * tp) / (2 * tp + fp + fn + smooth)
    pa = tp / (tp + fp + smooth)

    return iou.item(), dice.item(), pa.item()