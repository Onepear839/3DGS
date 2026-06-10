import json

notebook = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'accelerator': 'GPU',
        'colab': {'provenance': []}
    },
    'cells': []
}

def md(source_lines, cell_id):
    return {
        'cell_type': 'markdown',
        'metadata': {'id': cell_id},
        'source': [line + '\n' if not line.endswith('\n') else line for line in source_lines]
    }

def code(source_lines, cell_id):
    return {
        'cell_type': 'code',
        'metadata': {'id': cell_id},
        'source': [line + '\n' if not line.endswith('\n') else line for line in source_lines],
        'execution_count': None,
        'outputs': []
    }

cells = notebook['cells']

# Cell 1: Title
cells.append(md([
    '# 3D Gaussian Splatting + VGG感知损失增强训练',
    '',
    '**课程大作业 - 改进版**',
    '',
    '本笔记本在原始3DGS基础上增加了 **VGG感知损失**，使渲染结果在视觉感知质量上更优。',
    '',
    '对比实验：原始3DGS (lambda_vgg=0) vs 改进版 (lambda_vgg=0.2)'
], 'title'))

# Cell 2: GPU check
cells.append(md(['## 1. 检查GPU环境'], 'gpu_check'))

cells.append(code([
    'import torch',
    'print(f"PyTorch版本: {torch.__version__}")',
    'print(f"CUDA可用: {torch.cuda.is_available()}")',
    'if torch.cuda.is_available():',
    '    print(f"GPU型号: {torch.cuda.get_device_name(0)}")',
    '    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")'
], 'check_gpu'))

# Cell 3: Clone repo
cells.append(md([
    '## 2. 克隆项目代码',
    '',
    '从原始3DGS仓库克隆代码，然后应用改进。'
], 'clone_repo'))

cells.append(code([
    'import os',
    'from google.colab import drive',
    '',
    '# 挂载Google Drive用于保存模型',
    'drive.mount("/content/drive")',
    '',
    '# 克隆3DGS项目（含submodules）',
    '!git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive',
    '%cd gaussian-splatting',
], 'clone_code'))

# Cell 4: Install deps
cells.append(md([
    '## 3. 安装依赖',
    '',
    '编译CUDA自定义算子需要几分钟。'
], 'install_deps'))

cells.append(code([
    '# 安装PyTorch（Colab预装但版本确认）',
    '!pip install torch torchvision --quiet',
    '',
    '# 安装其他依赖',
    '!pip install tqdm plyfile opencv-python joblib --quiet',
    '',
    '# 编译CUDA算子（关键步骤，约2-3分钟）',
    '!pip install submodules/diff-gaussian-rasterization --quiet',
    '!pip install submodules/simple-knn --quiet',
    '!pip install submodules/fused-ssim --quiet',
    '',
    'print("\\n依赖安装完成!")'
], 'install_packages'))

# Cell 5: Create VGG loss module
cells.append(md([
    '## 4. 创建VGG感知损失模块',
    '',
    '这是本项目的核心改进。VGG感知损失使用预训练的VGG16网络提取多层特征，',
    '在特征空间计算L1距离，使渲染结果在感知质量上更接近真实图像。',
    '',
    '参考论文：Johnson et al. "Perceptual Losses for Real-Time Style Transfer" (ECCV 2016)'
], 'create_vgg'))

vgg_code = '''
#
# 改进: VGG感知损失 (Perceptual Loss)
# 基于Johnson et al. (2016)
# 使用预训练VGG16网络提取特征，计算特征空间的L1距离
#
import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:16].to(device).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        # 多层特征提取
        self.layers = {
            'relu1_2': 3,   # 低级边缘特征
            'relu2_2': 8,   # 中级纹理特征
            'relu3_3': 15,  # 高级形状特征
        }
        # 各层损失权重（归一化）
        self.layer_weights = {
            'relu1_2': 1.0 / 2.6,
            'relu2_2': 1.0 / 4.8,
            'relu3_3': 1.0 / 3.7,
        }
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        # ImageNet标准化
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        pred_features = pred_norm
        target_features = target_norm
        total_loss = 0.0
        for layer_name, layer_idx in self.layers.items():
            start_idx = 0 if layer_name == 'relu1_2' else (4 if layer_name == 'relu2_2' else 9)
            for i in range(start_idx, layer_idx + 1):
                pred_features = self.features[i](pred_features)
                target_features = self.features[i](target_features)
            layer_loss = nn.functional.l1_loss(pred_features, target_features)
            total_loss += self.layer_weights[layer_name] * layer_loss
        return total_loss

print("VGG感知损失模块已创建")
'''.strip().split('\n')

cells.append(code(
    ['%%writefile utils/vgg_loss.py'] + vgg_code,
    'create_vgg_loss'
))

# Cell 6: Modify train.py
cells.append(md([
    '## 5. 修改train.py集成感知损失',
    '',
    '在训练循环中引入VGG感知损失，与原始L1+SSIM损失合并。'
], 'modify_train'))

patch_code = '''
# 读取train.py
with open('train.py', 'r') as f:
    code = f.read()

# 1. 添加导入
old = 'from utils.image_utils import psnr'
new = 'from utils.image_utils import psnr\\nfrom utils.vgg_loss import VGGPerceptualLoss'
code = code.replace(old, new)

# 2. 添加lambda_vgg参数
old = 'parser.add_argument("--start_checkpoint", type=str, default = None)\\n    args = parser.parse_args(sys.argv[1:])'
new = '''parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--lambda_vgg", type=float, default=0.1, help="VGG perceptual loss weight")
    args = parser.parse_args(sys.argv[1:])'''
code = code.replace(old, new)

# 3. 初始化VGG损失
old = 'background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")'
new = '''background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    vgg_loss_fn = VGGPerceptualLoss(device="cuda") if opt.lambda_vgg > 0 else None'''
code = code.replace(old, new)

# 4. 在loss.backward()前加入感知损失
old = 'loss.backward()'
new = '''if vgg_loss_fn is not None and opt.lambda_vgg > 0:
            perceptual_loss = vgg_loss_fn(image, gt_image)
            loss += opt.lambda_vgg * perceptual_loss
        loss.backward()'''
code = code.replace(old, new)

with open('train.py', 'w') as f:
    f.write(code)

# 同时修改arguments/__init__.py
with open('arguments/__init__.py', 'r') as f:
    args_code = f.read()
old = 'self.optimizer_type = "default"\\n        super().__init__(parser, "Optimization Parameters")'
new = 'self.optimizer_type = "default"\\n        self.lambda_vgg = 0.0\\n        super().__init__(parser, "Optimization Parameters")'
args_code = args_code.replace(old, new)
with open('arguments/__init__.py', 'w') as f:
    f.write(args_code)

print("train.py已修改，VGG感知损失集成完成!")
'''.strip().split('\n')

cells.append(code(patch_code, 'patch_train'))

# Cell 7: Data preparation
cells.append(md([
    '## 6. 准备数据',
    '',
    '### 方式A：使用示例数据集（推荐快速体验）',
    '下载官方提供的Tanks&Temples数据集中的Truck场景。'
], 'data_prep'))

cells.append(code([
    '# 下载并解压示例数据集（Truck场景，约200MB）',
    'import urllib.request',
    'import zipfile',
    'import os',
    '',
    'data_dir = "/content/data"',
    'os.makedirs(data_dir, exist_ok=True)',
    '',
    'if not os.path.exists(f"{data_dir}/truck"):',
    '    print("正在下载Truck场景数据集...")',
    '    url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"',
    '    urllib.request.urlretrieve(url, f"{data_dir}/tandt_db.zip")',
    '    with zipfile.ZipFile(f"{data_dir}/tandt_db.zip", "r") as zip_ref:',
    '        zip_ref.extractall(data_dir)',
    '    print("数据集下载并解压完成!")',
    'else:',
    '    print("数据已存在")',
], 'download_data'))

cells.append(md([
    '### 方式B：上传自己的数据',
    '',
    '如果你有自己的照片，请先上传到Google Drive，然后运行下面的单元格将其复制到Colab。',
    '',
    '数据格式要求（COLMAP格式）：',
    '```',
    'your_scene/',
    '  input/    <- 原始照片',
    '  images/   <- COLMAP处理后的图片',
    '  sparse/   <- COLMAP重建结果',
    '```'
], 'upload_own'))

cells.append(code([
    '# 如果数据在Google Drive上，复制到Colab',
    '# import shutil',
    '# shutil.copytree("/content/drive/MyDrive/your_scene", "/content/data/your_scene")',
    '',
    '# 查看可用的数据集',
    '!ls -la /content/data/'
], 'load_drive_data'))

# Cell 8: Training original
cells.append(md([
    '## 7. 实验A：原始3DGS训练（无感知损失）',
    '',
    '作为对照组，先用 --lambda_vgg 0 训练原始3DGS。',
    '',
    '训练7000次迭代约需20分钟。'
], 'train_original'))

cells.append(code([
    '# 原始3DGS训练（lambda_vgg=0，不使用感知损失）',
    '!python train.py \\',
    '    -s /content/data/truck \\',
    '    -m /content/drive/MyDrive/3dgs_output/truck_original \\',
    '    --lambda_vgg 0 \\',
    '    --iterations 7000 \\',
    '    --test_iterations 7000 \\',
    '    --save_iterations 7000 \\',
    '    --quiet \\',
    '    --disable_viewer',
    '',
    'print("\\n原始3DGS训练完成!")'
], 'train_original_code'))

# Cell 9: Training improved
cells.append(md([
    '## 8. 实验B：改进版3DGS训练（+VGG感知损失）',
    '',
    '使用 --lambda_vgg 0.2 启用VGG感知损失，其他参数保持一致。'
], 'train_improved'))

cells.append(code([
    '# 改进版3DGS训练（lambda_vgg=0.2，加入感知损失）',
    '!python train.py \\',
    '    -s /content/data/truck \\',
    '    -m /content/drive/MyDrive/3dgs_output/truck_improved \\',
    '    --lambda_vgg 0.2 \\',
    '    --iterations 7000 \\',
    '    --test_iterations 7000 \\',
    '    --save_iterations 7000 \\',
    '    --quiet \\',
    '    --disable_viewer',
    '',
    'print("\\n改进版3DGS训练完成!")'
], 'train_improved_code'))

# Cell 10: Evaluation
cells.append(md([
    '## 9. 渲染评估与对比',
    '',
    '对两个模型分别渲染并计算PSNR、SSIM、LPIPS指标。'
], 'evaluate'))

cells.append(code([
    '# 渲染原始3DGS结果',
    '!python render.py \\',
    '    -m /content/drive/MyDrive/3dgs_output/truck_original \\',
    '    --quiet',
    '',
    '# 渲染改进版结果',
    '!python render.py \\',
    '    -m /content/drive/MyDrive/3dgs_output/truck_improved \\',
    '    --quiet',
    '',
    '# 计算评估指标',
    '!python metrics.py \\',
    '    -m /content/drive/MyDrive/3dgs_output/truck_original',
    '',
    '!python metrics.py \\',
    '    -m /content/drive/MyDrive/3dgs_output/truck_improved',
    '',
    'print("\\n评估完成!")'
], 'render_eval'))

# Cell 11: Visualization
cells.append(md([
    '## 10. 可视化对比',
    '',
    '将两个模型的渲染结果并排显示，并展示误差热力图。'
], 'visualize'))

viz_code = '''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def load_images(model_path, split="test", iteration=7000):
    render_dir = f"{model_path}/{split}/ours_{iteration}/renders"
    gt_dir = f"{model_path}/{split}/ours_{iteration}/gt"
    imgs = sorted(os.listdir(render_dir))
    if not imgs:
        return None, None
    fname = imgs[0]
    render = np.array(Image.open(f"{render_dir}/{fname}"))
    gt = np.array(Image.open(f"{gt_dir}/{fname}"))
    return render, gt

orig_render, orig_gt = load_images("/content/drive/MyDrive/3dgs_output/truck_original", "train")
impr_render, impr_gt = load_images("/content/drive/MyDrive/3dgs_output/truck_improved", "train")

if orig_render is not None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：原始3DGS
    axes[0, 0].imshow(orig_gt)
    axes[0, 0].set_title("真实图像 (Ground Truth)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(orig_render)
    axes[0, 1].set_title("原始3DGS渲染")
    axes[0, 1].axis("off")

    diff_orig = np.abs(orig_render.astype(float) - orig_gt.astype(float)).mean(axis=2)
    im = axes[0, 2].imshow(diff_orig, cmap="hot")
    axes[0, 2].set_title(f"误差热力图 (均值:{diff_orig.mean():.1f})")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # 第二行：改进版
    axes[1, 0].imshow(impr_gt)
    axes[1, 0].set_title("真实图像 (Ground Truth)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(impr_render)
    axes[1, 1].set_title("改进版3DGS渲染 (+VGG感知损失)")
    axes[1, 1].axis("off")

    diff_impr = np.abs(impr_render.astype(float) - impr_gt.astype(float)).mean(axis=2)
    im = axes[1, 2].imshow(diff_impr, cmap="hot")
    axes[1, 2].set_title(f"误差热力图 (均值:{diff_impr.mean():.1f})")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.suptitle("3DGS 改进前后对比 (VGG感知损失)", fontsize=16)
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/3dgs_output/comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("对比图已保存到 Google Drive")
else:
    print("渲染结果尚未生成，请先执行渲染步骤")
'''.strip().split('\n')

cells.append(code(viz_code, 'show_comparison'))

# Cell 12: Summary
cells.append(md([
    '## 11. 结果汇总',
    '',
    '| 指标 | 原始3DGS | 改进版 (+VGG感知损失) | 提升 |',
    '|------|---------|---------------------|------|',
    '| PSNR | (待填写) | (待填写) | - |',
    '| SSIM | (待填写) | (待填写) | - |',
    '| LPIPS | (待填写) | (待填写) | - |',
    '',
    '> LPIPS（感知相似度）越低越好，VGG感知损失主要优化此项指标。',
    '> PSNR可能小幅波动，但人眼观感会明显改善。',
    '',
    '---',
    '',
    '**模型文件已保存到:** /content/drive/MyDrive/3dgs_output/',
    '',
    '你可以下载 point_cloud.ply 文件到本地，用 SIBR 查看器打开查看。'
], 'summary'))

# Write JSON
with open('E:/Py Projects/gaussian-splatting-main/Colab_3DGS_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

# Validate
print('JSON已生成')

with open('E:/Py Projects/gaussian-splatting-main/Colab_3DGS_Training.ipynb', 'r', encoding='utf-8') as f:
    json.load(f)
print('JSON验证通过')

# Count
with open('E:/Py Projects/gaussian-splatting-main/Colab_3DGS_Training.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()
print(f'文件大小: {len(content)} 字节')
print(f'总cells: {len(notebook["cells"])}')