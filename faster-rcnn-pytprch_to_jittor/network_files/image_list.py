from typing import List, Tuple
import jittor as jt
from jittor import Var

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single Var.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Var, List[Tuple[int, int]]) -> None
        """
        Arguments:
            Vars (Var) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        """
                Jittor版本说明：
                1. 参数device可选，因为Jittor自动管理设备
                2. 保持接口兼容性，但实际不执行设备转移
                https://discuss.jittor.org/t/topic/329
                Jittor 里通过全局标志 jt.flags.use_cuda 来设置 var 是否在 gpu 上运算，不需要对单独的模型或者 tensor 设置
                """
        # Jittor统一内存管理，无需显式设备转移
        return ImageList(self.tensors, self.image_sizes)

