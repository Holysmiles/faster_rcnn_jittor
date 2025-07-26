import jittor as jt
from jittor import nn
from typing import List, Dict, Tuple, Union, Optional


def _filter_input(x: Dict[str, jt.Tensor], featmap_names: List[str]) -> Dict[str, jt.Tensor]:
    """过滤出需要使用的特征层（对应PyTorch的特征图筛选逻辑）"""
    return {k: v for k, v in x.items() if k in featmap_names}


def _setup_scales(x: Dict[str, jt.Tensor],
                  image_shapes: List[Tuple[int, int]],
                  canonical_scale: int,
                  canonical_level: int) -> Tuple[List[float], jt.Module]:
    """
    计算每个特征层的尺度因子及RoI到特征层的映射关系
    对应PyTorch中尺度初始化逻辑（基于FPN论文的启发式方法）
    """
    # 提取特征层名称并按分辨率排序（从高到低）
    featmap_names = sorted(x.keys())
    featmap_sizes = [x[k].shape[-2:] for k in featmap_names]  # [(H1, W1), (H2, W2), ...]

    # 计算每个特征层的缩放因子（特征层尺寸 / 原始图像尺寸）
    # 取第一张图像的尺寸作为基准（batch内图像已统一预处理）
    orig_h, orig_w = image_shapes[0]
    scales = []
    for (h, w) in featmap_sizes:
        scale = w / orig_w  # 宽度方向缩放因子（与高度方向一致，因图像等比缩放）
        scales.append(scale)

    # 定义RoI到特征层的映射器（基于FPN论文公式1）
    class LevelMapper(nn.Module):
        def __init__(self, scales: List[float], canonical_scale: int, canonical_level: int):
            super().__init__()
            self.scales = jt.array(scales, dtype=jt.float32)
            self.log2_scales = jt.log2(self.scales)  # 预计算log2(scale)
            self.canonical_level = canonical_level
            self.canonical_scale = canonical_scale

        def execute(self, roi_areas: jt.Tensor) -> jt.Tensor:
            """根据RoI面积计算目标特征层索引"""
            # 计算RoI的平方根面积（对应FPN论文中的s）
            roi_sizes = jt.sqrt(roi_areas)
            # 公式1：k = k0 + log2(s / 224)，k0=canonical_level
            target_levels = self.canonical_level + jt.log2(roi_sizes / self.canonical_scale)
            # 限制在有效特征层范围内
            target_levels = jt.clamp(target_levels, 0, len(self.scales) - 1).long()
            return target_levels

    return scales, LevelMapper(scales, canonical_scale, canonical_level)


def _multiscale_roi_align(x: Dict[str, jt.Tensor],
                          boxes: List[jt.Tensor],
                          output_size: Tuple[int, int],
                          sampling_ratio: int,
                          scales: List[float],
                          map_levels: jt.Module) -> jt.Tensor:
    """执行多尺度RoIAlign池化（核心逻辑）"""
    # 1. 收集所有图像的boxes并记录归属
    num_rois_per_image = [b.shape[0] for b in boxes]
    all_rois = jt.cat(boxes, dim=0)  # [总RoI数, 4] (x1,y1,x2,y2)
    device = all_rois.device

    # 2. 计算每个RoI的面积（用于确定目标特征层）
    roi_wh = all_rois[:, 2:4] - all_rois[:, 0:2]  # [总RoI数, 2] (w, h)
    roi_areas = roi_wh[:, 0] * roi_wh[:, 1]  # [总RoI数] 面积
    target_levels = map_levels(roi_areas)  # [总RoI数] 每个RoI的目标特征层索引

    # 3. 按特征层分组RoI
    feat_names = sorted(x.keys())  # 特征层名称（与scales顺序对应）
    roi_indices_per_level = {i: [] for i in range(len(feat_names))}
    for roi_idx, level in enumerate(target_levels):
        roi_indices_per_level[level.item()].append(roi_idx)

    # 4. 初始化RoIAlign算子（Jittor原生支持RoIAlign）
    roi_align = nn.RoIAlign(
        output_size=output_size,
        spatial_scale=1.0,  # 动态调整缩放因子
        sampling_ratio=sampling_ratio,
        aligned=True  # 对齐模式（与PyTorch保持一致）
    )

    # 5. 对每个特征层执行RoIAlign并收集结果
    pooled_results = []
    roi_indices_list = []
    for level, feat_name in enumerate(feat_names):
        roi_indices = roi_indices_per_level.get(level, [])
        if not roi_indices:
            continue  # 该层无RoI，跳过
        roi_indices = jt.array(roi_indices, dtype=jt.int64)

        # 提取该层的RoI
        rois = all_rois[roi_indices]  # [该层RoI数, 4]

        # 计算该层的空间缩放因子（RoI坐标从图像尺度→特征图尺度）
        scale = scales[level]
        rois_in_feat = rois * scale  # 转换RoI坐标到当前特征层

        # 执行RoIAlign
        feat = x[feat_name]  # [1, C, H, W]（假设batch_size=1）
        pooled = roi_align(feat, rois_in_feat)  # [该层RoI数, C, output_h, output_w]

        pooled_results.append(pooled)
        roi_indices_list.append(roi_indices)

    # 6. 按原始RoI顺序拼接结果
    total_rois = all_rois.shape[0]
    C = next(iter(x.values())).shape[1]  # 特征通道数
    output_h, output_w = output_size
    final_output = jt.zeros((total_rois, C, output_h, output_w), device=device)

    for pooled, indices in zip(pooled_results, roi_indices_list):
        final_output[indices] = pooled

    return final_output


class MultiScaleRoIAlign(nn.Module):
    """
    Jittor实现的多尺度RoIAlign池化（对应PyTorch的torchvision.ops.MultiScaleRoIAlign）

    功能：从多尺度特征图中提取RoI特征，通过启发式方法确定每个RoI对应的特征层
    （参考FPN论文公式1：根据RoI大小分配到不同分辨率的特征层）
    """
    __annotations__ = {
        "scales": Optional[List[float]],
        "map_levels": Optional[nn.Module]
    }

    def __init__(
            self,
            featmap_names: List[str],
            output_size: Union[int, Tuple[int, int], List[int]],
            sampling_ratio: int,
            *,
            canonical_scale: int = 224,
            canonical_level: int = 4,
    ):
        super().__init__()
        # 处理输出尺寸（统一为元组格式）
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, list):
            output_size = tuple(output_size)
        self.featmap_names = featmap_names
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.canonical_scale = canonical_scale  # 标准RoI尺寸（224）
        self.canonical_level = canonical_level  # 标准特征层（k0=4）

        # 动态计算的尺度和映射器（forward中初始化）
        self.scales: Optional[List[float]] = None
        self.map_levels: Optional[nn.Module] = None

    def execute(
            self,
            x: Dict[str, jt.Tensor],
            boxes: List[jt.Tensor],
            image_shapes: List[Tuple[int, int]],
    ) -> jt.Tensor:
        """
        前向传播：从多尺度特征图中提取RoI特征

        Args:
            x (Dict[str, jt.Tensor]): 特征图字典（键：特征层名称，值：特征张量[B, C, H, W]）
            boxes (List[jt.Tensor]): RoI列表（每个元素为单张图像的RoI，形状[N, 4]，格式(x1,y1,x2,y2)）
            image_shapes (List[Tuple[int, int]]): 原始图像尺寸列表（每个元素为(height, width)）

        Returns:
            jt.Tensor: 池化后的RoI特征（形状[总RoI数, C, output_h, output_w]）
        """
        # 过滤出需要使用的特征层
        x_filtered = _filter_input(x, self.featmap_names)

        # 初始化尺度和RoI-特征层映射器（仅首次调用时计算）
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )

        # 执行多尺度RoIAlign池化
        return _multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )