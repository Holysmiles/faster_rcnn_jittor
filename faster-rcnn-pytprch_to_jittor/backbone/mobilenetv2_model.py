from jittor import nn
import jittor as jt
from jittor import init
# from torchvision.ops import misc
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channel),
            nn.ReLU6()
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel, norm_layer=norm_layer),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            norm_layer(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def execute(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, weights_path=None, norm_layer=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1, norm_layer=norm_layer))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        if weights_path is None:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zero_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.one_(m.weight)
                    nn.init.zero_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.gauss_(m.weight, 0, 0.01)
                    nn.init.zero_(m.bias)
        else:
            # Jittor加载权重的方式与PyTorch不同
            import numpy as np
            params = jt.load(weights_path)
            self.load_state_dict(params)
            # self.load_state_dict(torch.load(weights_path))

    def execute(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = jt.flatten(x, 1)
        x = self.classifier(x)
        return x
if __name__ == '__main__':
    model = MobileNetV2(num_classes=1000, alpha=1.0, round_nearest=8, weights_path=None, norm_layer=None)
    print(model)