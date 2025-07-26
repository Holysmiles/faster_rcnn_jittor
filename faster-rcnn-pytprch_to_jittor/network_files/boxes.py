import jittor as jt
from jittor import nn
from jittor import Var
from typing import Tuple


def nms(boxes, scores, iou_threshold):
    # type: (jt.Var,jt.Var,  float) -> jt.Var
    """
    使用 Jittor 实现非极大值抑制（NMS）

    参数
    ----------
    boxes : jt.Var[N, 4]
        每个 box 的坐标，格式为 (x1, y1, x2, y2)
    scores : jt.Var[N]
        每个 box 对应的置信分数
    iou_threshold : float
        IoU 阈值，超过该值则进行抑制

    返回
    -------
    keep : jt.Var[int]
        被保留下来的 box 索引（按得分降序排列）
    """
    boxes_scores = jt.concat([boxes, scores.unsqueeze(1)], dim=1)   # [N, 4] + [N, 1] => [N, 5]

    return jt.nms(boxes_scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (jt.Var, jt.Var, jt.Var, float) -> jt.Var
    """
    批处理非极大值抑制（batched NMS）

    每个 idx 表示一个类别标签，同类之间才做 NMS，不同类之间互不干扰。

    参数
    ----------
    boxes : jt.Var[N, 4]
        box 坐标，格式为 (x1, y1, x2, y2)
    scores : jt.Var[N]
        每个 box 的得分
    idxs : jt.Var[N]
        每个 box 的类别索引
    iou_threshold : float
        IoU 阈值，用于抑制

    返回
    -------
    keep : jt.Var[int64]
        保留的 box 的索引（按得分降序排列）
    """
    if boxes.numel() == 0:
        return jt.empty((0,), dtype='int64')

    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax），用于偏移
    max_coordinate = boxes.max()

    # 为每个类别加一个较大的偏移，避免不同类别之间重叠
    # to(): Performs Tensor dtype and/or device conversion
    # 为每一个类别/每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    idxs = jt.to(idxs, boxes.dtype)
    offsets = idxs * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象
    boxes_for_nms = jt.add_(boxes , offsets[:, None])

    # 调用 Jittor 内置 nms 函数
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    # type : (jt.Var, float) -> jt.Var
    """
    Jittor 实现的 remove_small_boxes
    参数:
        boxes: jt.Var[N, 4]，每行为 (x1, y1, x2, y2)
        min_size: float，最小宽/高阈值
    返回:
        keep: jt.Var[int]，满足条件的框的索引
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]

    keep_mask = jt.logical_and(ws >= min_size, hs >= min_size)
    keep = jt.where(keep_mask)[0]
    return keep

def clip_boxes_to_image(boxes, size):
    # type: (jt.Var, Tuple[int, int]) -> jt.Var
    """
    Clip boxes so that they lie inside an image of size `size`.
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    # Jittor不支持torchvision._is_tracing()，直接使用clamp方法
    boxes_x = jt.clamp(boxes_x,min_v=0, max_v=width)   # 限制x坐标范围在[0,width]之间
    boxes_y = jt.clamp(boxes_y,min_v=0, max_v=height)  # 限制y坐标范围在[0,height]之间

    clipped_boxes = jt.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

def box_area(boxes):
    """
    计算每个框的面积
    boxes: jt.Var[N, 4] - (x1, y1, x2, y2)
    return: jt.Var[N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    计算 boxes1 和 boxes2 之间的 IoU
    boxes1: jt.Var[N, 4], boxes2: jt.Var[M, 4]
    return: jt.Var[N, M]
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    lt = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min_v=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou
