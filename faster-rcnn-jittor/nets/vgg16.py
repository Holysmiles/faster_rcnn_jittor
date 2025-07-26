# import torch
# import torch.nn as nn
# from torch.hub import load_state_dict_from_url
import jittor as jt
import jittor.nn as nn
# from load_state_dict_from_url import  load_state_dict_from_url

#--------------------------------------#
#   VGG16的结构
#--------------------------------------#
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #--------------------------------------#
        #   平均池化到7x7大小
        #--------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def execute(self, x):
        #--------------------------------------#
        #   特征提取
        #--------------------------------------#
        x = self.features(x)
        #--------------------------------------#
        #   平均池化
        #--------------------------------------#
        x = self.avgpool(x)
        #--------------------------------------#
        #   平铺后
        #--------------------------------------#
        x = jt.flatten(x, 1)
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                jt.init.gauss_(m.weight, 0, 0.01)
                # nn.init.normal_(m.weight, 0, 0.01)
                jt.init.constant_(m.bias, 0)

'''
假设输入图像为(600, 600, 3),随着cfg的循环,特征层变化如下：
600,600,3 -> 600,600,64 -> 600,600,64 -> 300,300,64 -> 300,300,128 -> 300,300,128 -> 150,150,128 -> 150,150,256 -> 150,150,256 -> 150,150,256 
-> 75,75,256 -> 75,75,512 -> 75,75,512 -> 75,75,512 -> 37,37,512 ->  37,37,512 -> 37,37,512 -> 37,37,512
到cfg结束,我们获得了一个37,37,512的特征层
'''

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


#--------------------------------------#
#   特征提取部分
#--------------------------------------#
def make_layers(cfg, batch_norm = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v 
    return nn.Sequential(*layers)

def decom_vgg16(pretrained = False):
    model = VGG(make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir = "./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分,最终获得一个37,37,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list(model.features)[:30]
    #----------------------------------------------------------------------------#
    #   获取分类部分,需要除去Dropout部分
    #----------------------------------------------------------------------------#
    classifier  = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]

    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier



import os
import requests
from tqdm import tqdm


def load_state_dict_from_url(url, model_dir="./model_data"):
    """
    从URL下载权重文件（适配PyTorch格式的.pth文件），并转换为Jittor兼容的状态字典

    Args:
        url: 权重文件的URL（如PyTorch官方模型链接）
        model_dir: 本地保存目录
    Returns:
        jittor兼容的状态字典（键为参数名，值为Jittor张量）
    """
    # 创建保存目录
    os.makedirs(model_dir, exist_ok=True)
    # 提取文件名（从URL中获取）
    filename = url.split("/")[-1]
    save_path = os.path.join(model_dir, filename)

    # 如果文件已存在，直接加载；否则下载
    if not os.path.exists(save_path):
        print(f"Downloading {url} to {save_path}...")
        # 发送HTTP请求（带进度条）
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

    # 加载PyTorch格式的.pth文件（Jittor支持直接加载PyTorch的state_dict）
    # 注意：PyTorch的.state_dict()中张量为torch.Tensor，需转换为Jittor张量
    state_dict = jt.load(save_path)  # Jittor的load函数可直接读取.pth文件

    # 转换所有值为Jittor张量（若原始为PyTorch张量）
    for k in state_dict:
        if isinstance(state_dict[k], jt.Tensor):
            continue  # 已为Jittor张量则跳过
        # 若为numpy数组或其他格式，转换为Jittor张量
        state_dict[k] = jt.array(state_dict[k])

    return state_dict

