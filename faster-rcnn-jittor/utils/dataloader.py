import cv2
import numpy as np

import jittor as jt
from PIL import Image

from jittor.dataset import Dataset
from utils.utils import cvtColor, preprocess_input



import cv2
import numpy as np
import jittor as jt
from PIL import Image
from jittor.dataset import Dataset
from utils.utils import cvtColor, preprocess_input  # 假设该函数存在


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        super().__init__()  # 初始化Jittor Dataset父类
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # [高, 宽]，例如[600, 600]
        self.train = train
        # 无需手动定义collate_fn等属性（通过set_attrs动态设置）

    def __len__(self):
        return self.length

    def rand(self, a=0, b=1):
        """生成[a, b)之间的随机数"""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """
        加载并预处理图像和标签
        返回：
            image_data: 预处理后的图像（形状为(input_shape[0], input_shape[1], 3)）
            box: 调整后的边界框（格式：(x1, y1, x2, y2, label)）
        """
        line = annotation_line.split()
        # 读取图像并转换为RGB（避免灰度图问题）
        try:
            image = Image.open(line[0])
        except Exception as e:
            raise FileNotFoundError(f"无法打开图像：{line[0]}，错误：{e}")
        image = cvtColor(image)  # 确保是RGB格式
        iw, ih = image.size  # 原始图像宽、高
        h, w = input_shape  # 目标高、宽（input_shape[0]是高，[1]是宽）

        # 解析边界框（格式：(x1, y1, x2, y2, label)）
        box = []
        for box_str in line[1:]:
            try:
                box.append(list(map(int, box_str.split(','))))
            except Exception as e:
                raise ValueError(f"边界框格式错误：{box_str}，错误：{e}")
        box = np.array(box, dtype=np.float32)  # 统一为float32类型

        if not random:
            # 验证模式：等比例缩放+灰条填充（确保输出形状为(h, w, 3)）
            scale = min(w / iw, h / ih)  # 缩放比例（确保图像完全放入目标尺寸）
            nw = int(iw * scale)  # 缩放后的宽
            nh = int(ih * scale)  # 缩放后的高
            dx = (w - nw) // 2  # 左右灰条宽度
            dy = (h - nh) // 2  # 上下灰条高度

            # 缩放图像并填充灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))  # 灰条背景
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, dtype=np.float32)  # 转换为数组

            # 调整边界框坐标（映射到新图像）
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * (nw / iw) + dx  # x坐标调整
                box[:, [1, 3]] = box[:, [1, 3]] * (nh / ih) + dy  # y坐标调整
                # 裁剪超出边界的框
                box[:, 0:2] = np.clip(box[:, 0:2], 0, [w, h])  # 左上角坐标不小于0
                box[:, 2:4] = np.clip(box[:, 2:4], 0, [w, h])  # 右下角坐标不大于图像尺寸
                # 过滤无效框（宽高需大于1）
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # 训练模式：随机增强（仍确保输出形状为(h, w, 3)）
        # 1. 随机缩放和扭曲
        new_ar = (iw / ih) * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)  # 随机缩放比例
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)  # 缩放

        # 2. 随机位置填充灰条
        dx = int(self.rand(0, w - nw))  # 随机x偏移（确保不超出边界）
        dy = int(self.rand(0, h - nh))  # 随机y偏移
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 3. 随机水平翻转
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 4. 色域变换（HSV调整）
        # 色域变换（修正后）
        image_data = np.array(image, np.uint8)
        # 拆分HSV通道（重命名为xxx_channel，避免与参数混淆）
        hue_channel, sat_channel, val_channel = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        # 计算变换参数（使用函数参数hue、sat、val，标量）
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1  # 现在形状(3,) * (3,) → 正确

        # 生成查找表（LUT）
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # 应用LUT到HSV通道（使用重命名后的数组）
        image_data = cv2.merge((
            cv2.LUT(hue_channel, lut_hue),
            cv2.LUT(sat_channel, lut_sat),
            cv2.LUT(val_channel, lut_val)
        ))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)


        # 5. 调整边界框坐标
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * (nw / iw) + dx  # x坐标调整
            box[:, [1, 3]] = box[:, [1, 3]] * (nh / ih) + dy  # y坐标调整
            if flip:  # 翻转时x坐标镜像
                box[:, [0, 2]] = w - box[:, [2, 0]]
            # 裁剪超出边界的框
            box[:, 0:2] = np.clip(box[:, 0:2], 0, [w, h])
            box[:, 2:4] = np.clip(box[:, 2:4], 0, [w, h])
            # 过滤无效框
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        # 最终确认图像形状（确保是(h, w, 3)）
        assert image_data.shape == (h, w, 3), f"训练模式图像形状错误：{image_data.shape}，预期{(h, w, 3)}"
        return image_data, box
    
    
    
    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, y    = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data    = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box         = box_data[:, :4]
        label       = box_data[:, -1]
        return image, box, label
#         index = index % self.length
#         image, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
#         image = jt.array(np.transpose(image, (2, 0, 1)).astype(np.float32))  # 转换为JT支持的格式
        
#         # 填充边界框
#         max_boxes = 50  # 设定最大框数
#         box_data = np.zeros((max_boxes, 5), dtype=np.float32)  # 设定边界框数据结构
        
#         # box_data = np.zeros((len(y), 5))
#         if len(y) > 0:
#             box_data[:min(len(y), max_boxes), :4] = y[:max_boxes]  # 填充有效框，超出部分截断

#         # box = box_data[:, :4]
#         # label = box_data[:, -1]
#         # return image, jt.array(box), jt.array(label)
#         return image, jt.array(box_data), jt.array(box_data[:, -1])
    
    def collate_batch(self, batch):
        images = []
        boxes = []
        labels = []
        for img, box, label in batch:
            images.append(img)
            boxes.append(box)
            labels.append(label)
        images = jt.stack(images, dim=0)  # 生成 [batch_size, 3, 600, 600]
        return images, boxes, labels

#     def __getitem__(self, index):
#         index = index % self.length  # 防止索引越界
#         # 加载图像和标签
#         image, y = self.get_random_data(
#             self.annotation_lines[index],
#             self.input_shape[0:2],  # 传入(高, 宽)
#             random=self.train
#         )

#         # 图像预处理：转置通道（HWC→CHW）+ 归一化
#         image = np.transpose(image, (2, 0, 1))  # 转换为(3, 高, 宽)
#         image = preprocess_input(np.array(image, dtype=np.float32))  # 归一化（假设输出float32）

#         # 边界框和标签处理（固定格式）
#         box_data = np.zeros((len(y), 5), dtype=np.float32)  # 5列：x1,y1,x2,y2,label
#         if len(y) > 0:
#             box_data[:len(y)] = y  # 填充有效框，无效部分用0填充

#         box = box_data[:, :4]  # 边界框坐标
#         label = box_data[:, -1].astype(np.int32)  # 标签（整数类型）
        
#         # 新增：检查图像形状（必须是3维：(3, 高, 宽)）
#         assert len(image.shape) == 3 and image.shape[0] == 3, \
#         f"单张图像形状错误：实际{image.shape}，预期(3, {self.input_shape[0]}, {self.input_shape[1]})"

#         return image, box, label





def generate_batch(dataset, batch_size):
    batch_images = []
    batch_boxes = []
    batch_labels = []
    for i, (img, box, label) in enumerate(dataset):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)  # 确保 [3, 600, 600]
        batch_images.append(img)
        batch_boxes.append(box)
        batch_labels.append(label)

        if len(batch_images) == batch_size:
            yield frcnn_dataset_collate(list(zip(batch_images, batch_boxes, batch_labels)))
            batch_images, batch_boxes, batch_labels = [], [], []

    if batch_images:
        yield frcnn_dataset_collate(list(zip(batch_images, batch_boxes, batch_labels)))

def init_dataloader(train_lines, val_lines):
    train_dataset = FRCNNDataset(train_lines, input_shape=[600, 600], train=True)
    train_dataset.set_attrs(
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    val_dataset = FRCNNDataset(val_lines, input_shape=[600, 600], train=False)
    val_dataset.set_attrs(
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return train_dataset, val_dataset





# =======================================================================================================================================



# class FRCNNDataset(Dataset):
#     def __init__(self, annotation_lines, input_shape = [600, 600], train = True):
#         super().__init__() 
#         self.annotation_lines   = annotation_lines
#         self.length             = len(annotation_lines)
#         self.input_shape        = input_shape
#         self.train              = train
#         # self.collate_fn = frcnn_dataset_collate  # 添加collate_fn属性，初始化为None

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         index       = index % self.length
#         #---------------------------------------------------#
#         #   训练时进行数据的随机增强
#         #   验证时不进行数据的随机增强
#         #---------------------------------------------------#
#         image, y    = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
#         image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
#         box_data    = np.zeros((len(y), 5))
#         if len(y) > 0:
#             box_data[:len(y)] = y

#         box         = box_data[:, :4]
#         label       = box_data[:, -1]
#         return image, box, label
    
    

#     def rand(self, a=0, b=1):
#         return np.random.rand()*(b-a) + a

#     def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
#         line = annotation_line.split()
#         #------------------------------#
#         #   读取图像并转换成RGB图像
#         #------------------------------#
#         image   = Image.open(line[0])
#         image   = cvtColor(image)
#         #------------------------------#
#         #   获得图像的高宽与目标高宽
#         #------------------------------#
#         iw, ih  = image.size
#         h, w    = input_shape
#         #------------------------------#
#         #   获得预测框
#         #------------------------------#
#         box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

#         if not random:
#             scale = min(w/iw, h/ih)
#             nw = int(iw*scale)
#             nh = int(ih*scale)
#             dx = (w-nw)//2
#             dy = (h-nh)//2

#             #---------------------------------#
#             #   将图像多余的部分加上灰条
#             #---------------------------------#
#             image       = image.resize((nw,nh), Image.BICUBIC)
#             new_image   = Image.new('RGB', (w,h), (128,128,128))
#             new_image.paste(image, (dx, dy))
#             image_data  = np.array(new_image, np.float32)

#             #---------------------------------#
#             #   对真实框进行调整
#             #---------------------------------#
#             if len(box)>0:
#                 np.random.shuffle(box)
#                 box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
#                 box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
#                 box[:, 0:2][box[:, 0:2]<0] = 0
#                 box[:, 2][box[:, 2]>w] = w
#                 box[:, 3][box[:, 3]>h] = h
#                 box_w = box[:, 2] - box[:, 0]
#                 box_h = box[:, 3] - box[:, 1]
#                 box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

#             return image_data, box
                
#         #------------------------------------------#
#         #   对图像进行缩放并且进行长和宽的扭曲
#         #------------------------------------------#
#         new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
#         scale = self.rand(.25, 2)
#         if new_ar < 1:
#             nh = int(scale*h)
#             nw = int(nh*new_ar)
#         else:
#             nw = int(scale*w)
#             nh = int(nw/new_ar)
#         image = image.resize((nw,nh), Image.BICUBIC)

#         #------------------------------------------#
#         #   将图像多余的部分加上灰条
#         #------------------------------------------#
#         dx = int(self.rand(0, w-nw))
#         dy = int(self.rand(0, h-nh))
#         new_image = Image.new('RGB', (w,h), (128,128,128))
#         new_image.paste(image, (dx, dy))
#         image = new_image

#         #------------------------------------------#
#         #   翻转图像
#         #------------------------------------------#
#         flip = self.rand()<.5
#         if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

#         image_data      = np.array(image, np.uint8)
#         #---------------------------------#
#         #   对图像进行色域变换
#         #   计算色域变换的参数
#         #---------------------------------#
#         r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
#         #---------------------------------#
#         #   将图像转到HSV上
#         #---------------------------------#
#         hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
#         dtype           = image_data.dtype
#         #---------------------------------#
#         #   应用变换
#         #---------------------------------#
#         x       = np.arange(0, 256, dtype=r.dtype)
#         lut_hue = ((x * r[0]) % 180).astype(dtype)
#         lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#         lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

#         image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
#         image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

#         #---------------------------------#
#         #   对真实框进行调整
#         #---------------------------------#
#         if len(box)>0:
#             np.random.shuffle(box)
#             box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
#             box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
#             if flip: box[:, [0,2]] = w - box[:, [2,0]]
#             box[:, 0:2][box[:, 0:2]<0] = 0
#             box[:, 2][box[:, 2]>w] = w
#             box[:, 3][box[:, 3]>h] = h
#             box_w = box[:, 2] - box[:, 0]
#             box_h = box[:, 3] - box[:, 1]
#             box = box[np.logical_and(box_w>1, box_h>1)] 
        
#         return image_data, box

# # DataLoader中collate_fn使用
# def frcnn_dataset_collate(batch):
#     images = []
#     bboxes = []
#     labels = []
#     for img, box, label in batch:
#         images.append(img)
#         bboxes.append(box)
#         labels.append(label)
#     images = jt.array(np.array(images))
#     return images, bboxes, labels



