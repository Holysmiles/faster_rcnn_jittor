# --------------------------------------------#
#   该部分代码用于查看网络结构、参数量及计算量
# --------------------------------------------#
import jittor as jt
from jittor import nn
# Jittor版本的计算量统计工具（需先安装：pip install jtflops）
from jtflops import get_model_complexity_info

# 假设已将FasterRCNN转换为Jittor版本（nets.frcnn中为Jittor实现）
from nets.frcnn import FasterRCNN

if __name__ == "__main__":
    input_shape = [600, 600]  # 输入图像尺寸 (H, W)
    num_classes = 21  # 类别数（含背景）



    jt.flags.use_cuda = 1 if jt.has_cuda else 0  # 设置全局默认设备

    # 初始化Jittor版本的FasterRCNN
    model = FasterRCNN(num_classes, backbone='vgg')
    model.to(device)  # 移动模型到目标设备（Jittor的to方法）

    # 1. 查看网络结构（模拟torchsummary的输出）
    print("Network Structure:")
    # 打印模型结构（Jittor的__str__方法会输出层结构）
    print(model)

    # 2. 计算参数量和计算量（对应thop的profile功能）
    # 输入尺寸：(3, H, W)，与PyTorch的输入格式一致
    input_size = (3, input_shape[0], input_shape[1])

    # 使用jtflops计算FLOPs和Params（Jittor专用工具）
    flops, params = get_model_complexity_info(
        model,
        input_size,
        as_strings=False,  # 先获取数值，后续统一格式化
        print_per_layer_stat=False  # 不打印每层统计（如需可设为True）
    )

    # 与原PyTorch代码保持一致：卷积操作算2次运算（乘法+加法）
    flops = flops * 2


    # 格式化输出（参考clever_format的"%.3f"格式）
    def clever_format(values, format="%.3f"):
        """简易格式化工具（将数值转为人类可读格式）"""
        units = ["", "K", "M", "G", "T"]
        unit_idx = 0
        value = values
        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1
        return f"{format % value}{units[unit_idx]}"


    flops_str = clever_format(flops)
    params_str = clever_format(params)

    # 输出结果（与原PyTorch代码输出格式一致）
    print('\nTotal GFLOPS: %s' % (flops_str))
    print('Total params: %s' % (params_str))