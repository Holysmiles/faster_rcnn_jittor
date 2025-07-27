import jittor as jt
from jittor import transform as jt_transform
from pycocotools.coco import COCO
import os
from PIL import Image
from typing import Callable, Optional, Tuple, Any
from pathlib import Path


class CocoDetection:
    """
    Jittor框架下的COCO检测数据集类（对应PyTorch的torchvision.datasets.CocoDetection）
    加载COCO格式的图像及标注，支持数据变换（图像和目标标注的预处理）
    """

    def __init__(
            self,
            root: Union[str, Path],
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        """
        Args:
            root: 图像存储的根目录（下载的COCO图像所在路径）
            annFile: COCO标注文件（.json格式）的路径
            transform: 仅对图像进行变换的函数（输入PIL图像，返回变换后的数据）
            target_transform: 仅对标注进行变换的函数（输入原始标注，返回变换后标注）
            transforms: 同时对图像和标注进行变换的函数（输入(image, target)，返回变换后的数据）
        """
        # 初始化COCO API（依赖pycocotools，与PyTorch版本一致）
        self.root = str(root)
        self.annFile = annFile
        self.coco = COCO(annFile)

        # 获取所有图像的ID（按COCO标注的图像ID排序）
        self.ids = list(sorted(self.coco.imgs.keys()))

        # 数据变换函数（优先级：transforms > transform + target_transform）
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    def _load_image(self, id: int) -> Image.Image:
        """加载指定ID的图像（从root目录读取）"""
        img_info = self.coco.loadImgs(id)[0]  # 获取图像元信息（包括文件名）
        img_path = os.path.join(self.root, img_info["file_name"])  # 图像完整路径
        return Image.open(img_path).convert("RGB")  # 读取并转为RGB格式

    def _load_target(self, id: int) -> Any:
        """加载指定ID图像的标注（返回COCO格式的标注列表）"""
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=id))  # 获取该图像的所有标注

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        按索引获取数据（图像+标注），支持数据变换

        Returns:
            tuple: (image, target)，其中image为变换后的图像张量，target为变换后的标注
        """
        # 获取当前索引对应的图像ID
        coco_id = self.ids[index]

        # 加载原始图像和标注
        image = self._load_image(coco_id)
        target = self._load_target(coco_id)

        # 应用数据变换（与PyTorch版本的变换逻辑一致）
        if self.transforms is not None:
            # 同时变换图像和标注
            image, target = self.transforms(image, target)
        else:
            # 分别变换图像和标注
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        """返回数据集大小（图像总数）"""
        return len(self.ids)

    def __repr__(self) -> str:
        """返回数据集信息（与PyTorch版本格式对齐）"""
        return (
            f"{self.__class__.__name__}("
            f"root={self.root!r}, "
            f"annFile={self.annFile!r}, "
            f"length={self.__len__()}"
            ")"
        )