import math
from functools import partial

import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
import jittor as jt
from jittor import nn

def box_iou(boxes1, boxes2):
    """
    计算 boxes1 和 boxes2 之间的 IoU
    Args:
        boxes1: jt.Var[N, 4], 第一组边界框 (x1, y1, x2, y2)
        boxes2: jt.Var[M, 4], 第二组边界框 (x1, y1, x2, y2)
    Returns:
        jt.Var[N, M], IoU 张量
    """

    # raise ValueError(f"boxes1 必须是形状为 [N,] 的一维张量，但得到 {boxes1.shape}") 12321 4
    # raise ValueError(f"boxes2 必须是形状为 [M,] 的一维张量，但得到 {boxes2.shape}") 1 4
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min_v=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter 
    iou = inter / union
    return iou

def box_area(boxes):
    """
    计算每个框的面积
    boxes: jt.Var[N, 4] - (x1, y1, x2, y2)
    return: jt.Var[N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# def bbox_iou(box_a, box_b):
#     """
#     计算两组边界框之间的IoU (Jittor兼容版本)
#     Args:
#         box_a: 边界框张量，形状为 [N, 4]
#         box_b: 边界框张量，形状为 [M, 4]
#     Returns:
#         IoU张量，形状为 [N, M]
#     """
#     # raise ValueError(f"box_a 必须是形状为 [N,] 的一维张量，但得到 {box_a.shape}") 12321 4
#     # raise ValueError(f"box_a 必须是形状为 [M,] 的一维张量，但得到 {box_b.shape}") 1 4
#     # 确保输入是Jittor张量
#     if not isinstance(box_a, jt.Var):
#         box_a = jt.array(box_a)
#     if not isinstance(box_b, jt.Var):
#         box_b = jt.array(box_b)
    
#     # 检查输入形状
#     if box_a.ndim != 2 or box_a.shape[1] != 4:
#         raise ValueError(f"box_a 必须是形状为 [N, 4] 的二维张量，但得到 {box_a.shape}")
#     if box_b.ndim != 2 or box_b.shape[1] != 4:
#         raise ValueError(f"box_b 必须是形状为 [M, 4] 的二维张量，但得到 {box_b.shape}")
    
#     # 计算面积
#     area_a = box_area(box_a)  # [N]
#     area_b = box_area(box_b)  # [M]
#     # ValueError(f"area_b 必须是形状为 [M, 4] 的二维张量，但得到 {box_b.shape}")
    
#     # 处理空输入
#     if box_a.shape[0] == 0 or box_b.shape[0] == 0:
#         return jt.zeros((box_a.shape[0], box_b.shape[0]), dtype=box_a.dtype)
    
#     assert box_a.ndim == 2 and box_b.ndim == 2, "边界框必须是[N,4]格式"
    
#     # 扩展维度以支持广播 [N,4] -> [N,1,4]
#     box_a = box_a.unsqueeze(1)  # 显式扩展维度避免隐式广播问题
    
#     # 计算交集区域的左上角坐标
#     lt = jt.maximum(box_a[..., :2], box_b[..., :2])
#     rb = jt.minimum(box_a[..., 2:], box_b[..., 2:])
    

#     wh = (rb - lt).clamp(min_v=0)  # [N, M, 2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

#     iou = inter / (area_a[:, None] + area_b - inter + 1e-6)  # 避免除以零
#     # raise ValueError(f"iou 必须是形状为 [N,M] 的一维张量，但得到 {iou.shape}")
#     return iou

# def box_area(boxes: jt.Var) -> jt.Var:
#     """
#     Jittor兼容的边界框面积计算
#     自动修正无效边界框并添加维度保护
#     Args:
#         boxes: 边界框张量，形状为 [N, 4]
#     Returns:
#         面积张量，形状为 [N, 1]
#     """
#     # 检查输入是否为二维张量且有 4 列
#     if boxes is None or boxes.ndim != 2 or boxes.shape[1] != 4:
#         raise ValueError(f"boxes 必须是形状为 [N, 4] 的二维张量，但得到 {boxes.shape if boxes is not None else None}")
    

#     boxes = jt.dfs_to_numpy(boxes)
#     area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     return jt.array(area)



# def bbox2loc(src_bbox, dst_bbox):
#     # 检查输入形状
#     if src_bbox.ndim != 2 or src_bbox.shape[1] != 4:
#         raise ValueError(f"anchor 必须是形状为 [M, 4] 的二维张量，但得到 {src_bbox.shape}")
#     if dst_bbox.ndim != 2 or dst_bbox.shape[1] != 4:
#         raise ValueError(f"bbox 必须是形状为 [N, 4] 的二维张量，但得到 {dst_bbox.shape}")
#     # print(f"anchor 必须是形状为 [M, 4] 的二维张量，但得到 {anchor.shape}")

#     width = src_bbox[:, 2] - src_bbox[:, 0]
#     height = src_bbox[:, 3] - src_bbox[:, 1]
#     ctr_x = src_bbox[:, 0] + 0.5 * width
#     ctr_y = src_bbox[:, 1] + 0.5 * height

#     base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
#     base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
#     base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
#     base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

#     eps = np.finfo(height.dtype).eps
#     width = np.maximum(width, eps)
#     height = np.maximum(height, eps)
    
#     base_ctr_x = jt.dfs_to_numpy(base_ctr_x)
#     base_ctr_y = jt.dfs_to_numpy(base_ctr_y)
#     # print("-----------------------------anchor shape:", base_ctr_x.shape) 128
#     # print("-----------------------------bbox[argmax_ious] shape:", ctr_x.shape) 12321
#     dx = (base_ctr_x - ctr_x) / width
#     dy = (base_ctr_y - ctr_y) / height
#     dw = np.log(base_width / width)
#     dh = np.log(base_height / height)

#     loc = np.vstack((dx, dy, dw, dh)).transpose()
#     return loc

def bbox2loc(src_bbox, dst_bbox):
    # 确保输入是Jittor张量
    if not isinstance(src_bbox, jt.Var):
        src_bbox = jt.array(src_bbox)
    if not isinstance(dst_bbox, jt.Var):
        dst_bbox = jt.array(dst_bbox)
    
    # 数值稳定性保护
    widths = jt.maximum(src_bbox[:, 2] - src_bbox[:, 0], 1.0)
    heights = jt.maximum(src_bbox[:, 3] - src_bbox[:, 1], 1.0)
    ctr_x = src_bbox[:, 0] + 0.5 * widths
    ctr_y = src_bbox[:, 1] + 0.5 * heights

    base_widths = jt.maximum(dst_bbox[:, 2] - dst_bbox[:, 0], 1.0)
    base_heights = jt.maximum(dst_bbox[:, 3] - dst_bbox[:, 1], 1.0)
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_widths
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_heights
    
    # 安全计算偏移量
    dx = (base_ctr_x - ctr_x) / widths
    dy = (base_ctr_y - ctr_y) / heights
    dw = jt.log(jt.maximum(base_widths / widths, 1e-5))
    dh = jt.log(jt.maximum(base_heights / heights, 1e-5))
    
    return jt.stack([dx, dy, dw, dh], dim=1)

class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample       = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio      = pos_ratio

    def __call__(self, bbox, anchor):
        # # 展平边界框：去除冗余维度
        # if bbox.ndim == 3 and bbox.shape[0] == 1:
        #     bbox = bbox.squeeze(0)  # 从 [1,1,4] -> [1,4]
        # elif bbox.ndim == 3:
        #     bbox = bbox.reshape(-1, 4)  # 从 [K,M,4] -> [K*M, 4]
        # # 验证形状是否符合 [N,4]
        # if bbox.ndim != 2 or bbox.shape[1] != 4:
        #     raise ValueError(f"转换后bbox形状仍无效: {bbox.shape}")
        
        # 1. 确保anchor是(N, 4)（锚框数量N，4个坐标）
        anchor = np.reshape(anchor, (-1, 4))  # 强制转为二维(N,4)
        # 2. 确保bbox是(M, 4)（真实框数量M，4个坐标）
        bbox = np.reshape(bbox, (-1, 4))  # 强制转为二维(M,4)

        # 检查输入形状
        if anchor.ndim != 2 or anchor.shape[1] != 4:
            raise ValueError(f"anchor 必须是形状为 [M, 4] 的二维张量，但得到 {anchor.shape}")
        if bbox.ndim != 2 or bbox.shape[1] != 4:
            raise ValueError(f"bbox 必须是形状为 [N, 4] 的二维张量，但得到 {bbox.shape}")
        # raise ValueError(f"anchor 必须是形状为 [M, 4] 的二维张量，但得到 {anchor.shape}") # (12321, 4)
        # raise ValueError(f"bbox 必须是形状为 [N, 4] 的二维张量，但得到 {bbox.shape}") # (1, 4)

        # 处理空 bbox
        if bbox.shape[0] == 0:
            label = np.full((anchor.shape[0],), -1, dtype=np.int32)
            loc = np.zeros_like(anchor)
            return loc, label

        argmax_ious, label = self._create_label(anchor, bbox)
        
        # 确保 argmax_ious 是 numpy 类型（否则不能索引）
        # if isinstance(argmax_ious, jt.Var):
        #     argmax_ious = argmax_ious.dfs_to_numpy()
        
        if (label > 0).any():
            # if anchor.ndim != 2 or anchor.shape[1] != 4:
            #     raise ValueError(f"anchor---bbox2loc 必须是形状为 [M, 4] 的二维张量，但得到 {anchor.shape}")
            # if bbox.ndim != 2 or bbox.shape[1] != 4:
            #     raise ValueError(f"bbox---=bbox2loc 必须是形状为 [N, 4] 的二维张量，但得到 {bbox.shape}")
            # raise ValueError(f"bbox---=bbox2loc 必须是形状为 [N, 4] 的二维张量，但得到 {bbox.shape}") # 1 4
            # raise ValueError(f"argmax_ious 必须是形状为 [N,] 的一维张量，但得到 {argmax_ious.shape}") # argmax_ious shape[2,12321,1,]
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label

    def _calc_ious(self, anchor, bbox):
        #----------------------------------------------#
        #   anchor和bbox的iou
        #   获得的ious的shape为[num_anchors, num_gt]
        #----------------------------------------------#
        ious = box_iou(anchor, bbox)

        if len(bbox)==0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        #---------------------------------------------------------#
        #   获得每一个先验框最对应的真实框  [num_anchors, ]
        #---------------------------------------------------------#
        
        ious = jt.dfs_to_numpy(ious)
        argmax_ious = ious.argmax(axis=1)
        # raise ValueError(f"ious 必须是形状为 [N,M] 的张量，但得到 {ious.shape}，它的类型为{type(ious)}")
        # raise ValueError(f"argmax_ious 必须是形状为 [N,] 的张量，但得到 {argmax_ious.shape}，它的类型为{type(argmax_ious)}")
        #---------------------------------------------------------#
        #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
        #---------------------------------------------------------#
        max_ious = np.max(ious, axis=1)
        #---------------------------------------------------------#
        #   获得每一个真实框最对应的先验框  [num_gt, ]
        #---------------------------------------------------------#
        gt_argmax_ious = ious.argmax(axis=0)
        #---------------------------------------------------------#
        #   保证每一个真实框都存在对应的先验框
        #---------------------------------------------------------#
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i

        return argmax_ious, max_ious, gt_argmax_ious
#     def _calc_ious(self, anchor, bbox):
#         #----------------------------------------------#
#         #   anchor和bbox的iou
#         #   获得的ious的shape为[num_anchors, num_gt]
#         #----------------------------------------------#
#         ious = bbox_iou(anchor, bbox)
#         # raise ValueError(f"ious 必须是形状为 [N,M] 的一维张量，但得到 {ious.shape}") #  [12321,12321,1,] -》 12321 1

#         if len(bbox)==0:
#             return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
#         #---------------------------------------------------------#
#         #   获得每一个先验框最对应的真实框  [num_anchors, ]
#         #---------------------------------------------------------#
#         ious = jt.array(ious)
#         # raise ValueError(f"ious 必须是形状为 [N,] 的一维张量，但得到 {ious.shape}") # [12321,12321,1,]
#         argmax_ious = jt.array(ious.argmax(dim=1)).reshape(-1)
#         # raise ValueError(f"argmax_ious 必须是形状为 [N,] 的一维张量，但得到 {argmax_ious.shape}") # 形状为 [N,] 的一维张量，但得到 [24642,]
#         #---------------------------------------------------------#
#         #   找出每一个先验框最对应的真实框的iou  [num_anchors, ]
#         #---------------------------------------------------------#
#         max_ious = jt.max(ious, dim=1).reshape(-1)
#         #---------------------------------------------------------#
#         #   获得每一个真实框最对应的先验框  [num_gt, ]
#         #---------------------------------------------------------#
#         gt_argmax_ious = jt.array(ious.argmax(dim=0))
#         #---------------------------------------------------------#
#         #   保证每一个真实框都存在对应的先验框
#         #---------------------------------------------------------#
#         for i in range(len(gt_argmax_ious)):
#             idx_var = gt_argmax_ious[i].flatten()  # 展平多维张量
#             if idx_var.numel() == 0:  # 空张量跳过
#                 continue
#             elif idx_var.numel() == 1:  # 单元素直接转换
#                 idx = int(idx_tensor.item())
#             else:  # 多元素取首值（按业务需求可改为最大值/众数）
#                 idx = int(idx_var[0].item())

#             # 边界校验后赋值
#             if 0 <= idx < len(argmax_ious):
#                 argmax_ious[idx] = i  # 赋值为标量而非序列

#         return argmax_ious, max_ious, gt_argmax_ious
        
    def _create_label(self, anchor, bbox):
        # ------------------------------------------ #
        #   1是正样本，0是负样本，-1忽略
        #   初始化的时候全部设置为-1
        # ------------------------------------------ #
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # ------------------------------------------------------------------------ #
        #   argmax_ious为每个先验框对应的最大的真实框的序号         [num_anchors, ]
        #   max_ious为每个真实框对应的最大的真实框的iou             [num_anchors, ]
        #   gt_argmax_ious为每一个真实框对应的最大的先验框的序号    [num_gt, ]
        # ------------------------------------------------------------------------ #
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)
        
        # ----------------------------------------------------- #
        #   如果小于门限值则设置为负样本
        #   如果大于门限值则设置为正样本
        #   每个真实框至少对应一个先验框
        # ----------------------------------------------------- #
        # flag1 = max_ious < self.neg_iou_thresh
        # label[flag1] = 0
        # flag2 = max_ious >= self.pos_iou_thresh
        # label[flag2] = 1
        # 生成布尔索引时显式压缩维度
        flag1 = (max_ious < self.neg_iou_thresh).reshape(-1)  # 展平为一维
        flag2 = (max_ious >= self.pos_iou_thresh).reshape(-1)
        flag1 = jt.array(flag1)
        label = jt.array(label)
        # 赋值前校验长度
        assert flag1.numel() == label.numel(), "布尔索引长度与标签长度不匹配"
        label = jt.where(flag1, 0, label)       # 负样本置0
        label = jt.where(flag2, 1, label)        # 正样本置1
        
        if len(gt_argmax_ious)>0:
            label[gt_argmax_ious] = 1

        # ----------------------------------------------------- #
        #   判断正样本数量是否大于128，如果大于则限制在128
        # ----------------------------------------------------- #
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # ----------------------------------------------------- #
        #   平衡正负样本，保持总数量为256
        # ----------------------------------------------------- #
        label = jt.dfs_to_numpy(label)
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label
#     def _create_label(self, anchor, bbox):
#             label = np.empty((len(anchor),), dtype=np.int32)
#             label.fill(-1)

#             argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

#             label[max_ious < self.neg_iou_thresh] = 0
#             label[max_ious >= self.pos_iou_thresh] = 1
#             if len(gt_argmax_ious) > 0:
#                 label[gt_argmax_ious] = 1

#             n_pos = int(self.pos_ratio * self.n_sample)
#             pos_index = np.where(label == 1)[0]
#             if len(pos_index) > n_pos:
#                 disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
#                 label[disable_index] = -1

#             n_neg = self.n_sample - np.sum(label == 1)
#             neg_index = np.where(label == 0)[0]
#             if len(neg_index) > n_neg:
#                 disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
#                 label[disable_index] = -1

#             return argmax_ious, label



class ProposalTargetCreator(object):
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        # raise ValueError(f"bbox 必须是形状为 [N,M] 的张量，但得到 {bbox.shape}，它的类型为{type(bbox)}") # 1 4
        # raise ValueError(f"roi 必须是 {roi.shape}，它的类型为{type(roi)}") # 600 4
        # bbox = bbox.squeeze(0)
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0)
        # ----------------------------------------------------- #
        #   计算建议框和真实框的重合程度
        # ----------------------------------------------------- #
        iou = box_iou(roi, bbox)
        
        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框  [num_roi, ]
            #---------------------------------------------------------#
            iou = jt.dfs_to_numpy(iou)
            gt_assignment = iou.argmax(axis=1)
            #---------------------------------------------------------#
            #   获得每一个建议框最对应的真实框的iou  [num_roi, ]
            #---------------------------------------------------------#
            max_iou = iou.max(axis=1)
            #---------------------------------------------------------#
            #   真实框的标签要+1因为有背景的存在
            #---------------------------------------------------------#
            gt_roi_label = label[gt_assignment] + 1

        #----------------------------------------------------------------#
        #   满足建议框和真实框重合程度大于neg_iou_thresh_high的作为负样本
        #   将正样本的数量限制在self.pos_roi_per_image以内
        #----------------------------------------------------------------#
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)

        #-----------------------------------------------------------------------------------------------------#
        #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
        #   将正样本的数量和负样本的数量的总和固定成self.n_sample
        #-----------------------------------------------------------------------------------------------------#
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
            
        #---------------------------------------------------------#
        #   sample_roi      [n_sample, 4]
        #   gt_roi_loc      [n_sample, 4]
        #   gt_roi_label    [n_sample, ]
        #---------------------------------------------------------#
        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox)==0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label

class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train    = model_train
        self.optimizer      = optimizer

        self.rpn_sigma      = 1
        self.roi_sigma      = 1

        self.anchor_target_creator      = AnchorTargetCreator()
        self.proposal_target_creator    = ProposalTargetCreator()

        self.loc_normalize_std          = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        # roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma 128*4
        # pred_loc = jt.dfs_to_numpy(pred_loc)
        # raise ValueError(f"pred_loc 是形状为  {pred_loc.shape}，它的类型为{type(pred_loc)}") #  (12321, 4)，numpy.ndarray
        # raise ValueError(f"gt_label 是形状为  {gt_label.shape}，它的类型为{type(gt_label)}") #  [12321,]，它的类型为<class 'jittor_core.Var'>
        # raise ValueError(f"gt_loc 是形状为  {gt_loc.shape}，它的类型为{type(gt_loc)}") # [12321,4,]，它的类型为<class 'jittor.jittor_core.Var'>
        # print(gt_label)
        pred_loc = jt.dfs_to_numpy(pred_loc)
        gt_loc   = jt.dfs_to_numpy(gt_loc)
        gt_label = jt.dfs_to_numpy(gt_label)
        # raise ValueError(f"pred_loc 是形状为  {pred_loc.shape}，它的类型为{type(pred_loc)}") #   (12321, 4)，它的类型为<class 'numpy.ndarray'>
        # raise ValueError(f"gt_label 是形状为  {gt_label.shape}，它的类型为{type(gt_label)}") #   (12321,)，它的类型为<class 'numpy.ndarray'>
        # raise ValueError(f"gt_loc 是形状为  {gt_loc.shape}，它的类型为{type(gt_loc)}") #   (12321, 4)，它的类型为<class 'numpy.ndarra
        

        mask = (gt_label > 0).reshape(-1) 
        if not mask.any():
            return jt.float32(0)  # 无正样本时返回0损失
        # raise ValueError(f"mask 是形状为  {mask.shape}，它的类型为{type(mask)},{mask}") #  (12321,)
        
        pred_loc = pred_loc[mask]
        gt_loc   = gt_loc[mask]
        # pred_loc    = pred_loc[gt_label > 0]
        # gt_loc      = gt_loc[gt_label > 0]
        # raise ValueError(f"pred_loc 是形状为  {pred_loc.shape}，它的类型为{type(pred_loc)}") #    (N, 4)，它的类型为<class 'numpy.ndarray'>
        # raise ValueError(f"gt_label 是形状为  {gt_label.shape}，它的类型为{type(gt_label)}") #  (12321,)，它的类型为<class 'numpy.ndarray'>
        # raise ValueError(f"gt_loc 是形状为  {gt_loc.shape}，它的类型为{type(gt_loc)}") #    (M, 4)，它的类型为<class 'numpy.ndarray'>
        # print(f"pred_loc 是形状为  {pred_loc.shape}")
        # print(f"gt_loc 是形状为  {gt_loc.shape}")
        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)
        regression_diff = jt.array(regression_diff)
        regression_diff = regression_diff.abs().float()
        regression_loss = jt.where(
                regression_diff < (1. / sigma_squared),
                0.5 * sigma_squared * regression_diff ** 2,
                regression_diff - 0.5 / sigma_squared
            )
        regression_loss = regression_loss.sum()
        gt_label = jt.array(gt_label)
        num_pos         = (gt_label > 0).sum().float()
        
        regression_loss /= jt.maximum(num_pos, jt.ones_like(num_pos))
        return regression_loss

        
    def execute(self, imgs, bboxes, labels, scale):
        # imgs = jt.dfs_to_numpy(imgs)
        # bboxes = jt.dfs_to_numpy(bboxes)
        # labels = jt.dfs_to_numpy(labels)
        # scale = jt.dfs_to_numpy(scale)
        # raise ValueError(f'================imgs:{imgs.shape}')
        n           = imgs.shape[0]
        img_size    = imgs.shape[2:]
        #-------------------------------#
        #   获取公用特征层
        #-------------------------------#
        base_feature = self.model_train(imgs, mode = 'extractor')

        # -------------------------------------------------- #
        #   利用rpn网络获得调整参数、得分、建议框、先验框
        # -------------------------------------------------- #
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x = [base_feature, img_size], scale = scale, mode = 'rpn')
        
        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all  = 0, 0, 0, 0
        sample_rois, sample_indexes, gt_roi_locs, gt_roi_labels                 = [], [], [], []
        for i in range(n):
            bbox        = bboxes[i]
            label       = labels[i]
            rpn_loc     = rpn_locs[i]
            rpn_score   = rpn_scores[i]
            roi         = rois[i]
            # -------------------------------------------------- #
            #   利用真实框和先验框获得建议框网络应该有的预测结果
            #   给每个先验框都打上标签
            #   gt_rpn_loc      [num_anchors, 4]
            #   gt_rpn_label    [num_anchors, ]
            # -------------------------------------------------- #
            gt_rpn_loc, gt_rpn_label    = self.anchor_target_creator(bbox, anchor[0].cpu().numpy())
            gt_rpn_loc                  = jt.array(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label                = jt.array(gt_rpn_label).type_as(rpn_locs).long()
            
            # raise ValueError(f"gt_rpn_loc 是形状为  {gt_rpn_loc.shape}，它的类型为{type(gt_rpn_loc)}") #  [12321,4,]
            # raise ValueError(f"gt_rpn_label 是形状为  {gt_rpn_label.shape}，它的类型为{type(gt_rpn_label)}")  # [12321,]
            
            # -------------------------------------------------- #
            #   分别计算建议框网络的回归损失和分类损失
            # -------------------------------------------------- #
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = nn.cross_entropy_loss(rpn_score, gt_rpn_label, ignore_index=-1)
  
            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            # ------------------------------------------------------ #
            #   利用真实框和建议框获得classifier网络应该有的预测结果
            #   获得三个变量，分别是sample_roi, gt_roi_loc, gt_roi_label
            #   sample_roi      [n_sample, ]
            #   gt_roi_loc      [n_sample, 4]
            #   gt_roi_label    [n_sample, ]
            # ------------------------------------------------------ #
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(jt.array(sample_roi).type_as(rpn_locs))
            sample_indexes.append(jt.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(jt.array(gt_roi_loc).type_as(rpn_locs))
            gt_roi_labels.append(jt.array(gt_roi_label).type_as(rpn_locs).long())
            
            # raise ValueError(f"sample_roi 是形状为  {sample_roi.shape}，它的类型为{type(sample_roi)}") #  [128,4,]
            # raise ValueError(f"gt_roi_loc 是形状为  {gt_roi_loc.shape}，它的类型为{type(gt_roi_loc)}") #  [128,4,]
            # raise ValueError(f"gt_roi_label 是形状为  {gt_roi_label.shape}，它的类型为{type(gt_roi_label)}") #  [128,1,]
        # 例如统一到 128 个 RoIs
        max_rois = 128
        padded_sample_rois = []
        for rois in sample_rois:
            if rois.shape[0] > max_rois:
                rois = rois[:max_rois]  # 截断
            elif rois.shape[0] < max_rois:
                pad = jt.zeros((max_rois - rois.shape[0], 4))  # 补零
                rois = jt.concat([rois, pad], dim=0)
            padded_sample_rois.append(rois)

        sample_rois = jt.stack(padded_sample_rois, dim=0)  # shape = [B, max_rois, 4]

        # sample_rois     = jt.stack(sample_rois, dim=0)
        max_rois = 128
        padded_sample_indexes = []
        for idx in sample_indexes:
            if idx.shape[0] > max_rois:
                idx = idx[:max_rois]
            elif idx.shape[0] < max_rois:
                pad = jt.zeros((max_rois - idx.shape[0],), dtype=idx.dtype)
                idx = jt.concat([idx, pad], dim=0)
            padded_sample_indexes.append(idx)

        sample_indexes = jt.stack(padded_sample_indexes, dim=0)

        # sample_indexes  = jt.stack(sample_indexes, dim=0)
        roi_cls_locs, roi_scores = self.model_train([base_feature, sample_rois, sample_indexes, img_size], mode = 'head')
        
        for i in range(n):
            # ------------------------------------------------------ #
            #   根据建议框的种类，取出对应的回归预测结果
            # ------------------------------------------------------ #
            n_sample = roi_cls_locs.size()[1]
            
            roi_cls_loc     = roi_cls_locs[i]
            roi_score       = roi_scores[i]
            gt_roi_loc      = gt_roi_locs[i]
            gt_roi_label    = gt_roi_labels[i]
        
            n_sample = roi_cls_loc.shape[0]

            
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            # roi_cls_loc = roi_cls_loc.view(-1, n_sample,  4)
            # print(f"gt_roi_label的形状：{gt_roi_label.shape}")
            
            if isinstance(gt_roi_label, np.ndarray):
                gt_roi_label = jt.array(gt_roi_label)
            gt_roi_label = gt_roi_label.reshape(-1)
            
            valid_mask = gt_roi_label >= 0
            valid_indices = jt.where(valid_mask)[0]
            if valid_indices.shape[0] == 0:
                continue

            gt_roi_label_valid = gt_roi_label[valid_indices]
            roi_score_valid = roi_score[valid_indices]
            gt_roi_loc_valid = gt_roi_loc[valid_indices]
            roi_cls_loc_valid = roi_cls_loc[valid_indices, gt_roi_label_valid]

            roi_cls_loss = nn.CrossEntropyLoss()(roi_score_valid, gt_roi_label_valid)
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_cls_loc_valid, gt_roi_loc_valid, gt_roi_label_valid, self.roi_sigma)

            roi_cls_loss_all += roi_cls_loss
            roi_loc_loss_all += roi_loc_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses.append(sum(losses))
        return losses

            
      

            # roi_loc     = roi_cls_loc[jt.arange(0, n_sample), gt_roi_label]
#             # roi_loc = roi_loc[0]
#             # # 从 roi_cls_loc 中根据类别索引，取出每个 RoI 对应的预测框
#             roi_loc = roi_cls_loc[jt.arange(n_sample), gt_roi_label] 

            
            
            
#             # print(roi_loc.shape)
#             # print(roi_loc)
#             # print(gt_roi_label.data.shape) 
#             # print(gt_roi_label.data)
            
            
            

#             # -------------------------------------------------- #
#             #   分别计算Classifier网络的回归损失和分类损失
#             # -------------------------------------------------- # 
#             # raise ValueError(f"roi_loc 是形状为  {roi_loc.shape}，它的类型为{type(roi_loc)}")#  [128,4,]
#             # raise ValueError(f"gt_roi_loc 是形状为  {gt_roi_loc.shape}，它的类型为{type(gt_roi_loc)}") #  [128,4,]
#             # raise ValueError(f"gt_roi_label.data 是形状为  {gt_roi_label.data.shape}，它的类型为{type(gt_roi_label.data)}") #  (128, 1)，它的类型为<class 'numpy.ndarray'>
#             # raise ValueError(f"self.roi_sigma 是形状为  {self.roi_sigma.shape}，它的类型为{type(self.roi_sigma)}")
#             # gt_roi_label.data = gt_roi_label.data.squeeze(-1)
#             roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            
#             roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

#             roi_loc_loss_all += roi_loc_loss
#             roi_cls_loss_all += roi_cls_loss
            
#         losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
#         losses = losses + [sum(losses)]
#         return losses

    def train_step(self, imgs, bboxes, labels, scale, fp16=False, scaler=None):
        self.optimizer.zero_grad()
        losses = self.execute(imgs, bboxes, labels, scale)
        total_loss = losses[-1]
        # 添加梯度裁剪防止爆炸
        self.optimizer.clip_grad_norm(self.model_train.parameters(), 1.0)

        self.optimizer.step(total_loss)

            
        return losses

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                n = jt.array(m.weight.data)
                jt.init.gauss_(n, 0.0, init_gain)
            elif init_type == 'xavier':
                jt.init.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                jt.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                jt.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            jt.init.normal_(m.weight.data, 1.0, 0.02)
            jt.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
