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

