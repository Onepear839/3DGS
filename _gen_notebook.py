import json
import pathlib

cells = []

def md(src, cid):
    src = [l + ('\n' if not l.endswith('\n') else '') for l in src]
    return {'cell_type': 'markdown', 'metadata': {'id': cid}, 'source': src}

def code(src, cid):
    src = [l + ('\n' if not l.endswith('\n') else '') for l in src]
    return {'cell_type': 'code', 'metadata': {'id': cid}, 'source': src, 'execution_count': None, 'outputs': []}

# === CELL 1 ===
cells.append(md([
    '# 3D Gaussian Splatting + VGG感知损失增强训练',
    '',
    '**课程大作业 - 改进版**',
    '',
    '本笔记本在原始3DGS基础上增加了 **VGG感知损失**，使渲染结果在视觉感知质量上更优。',
    '',
    '对比实验：原始3DGS (lambda_vgg=0) vs 改进版 (lambda_vgg=0.2)'
], 'title'))

# === CELL 2: GPU check ===
cells.append(md(['## 1. 检查GPU环境'], 'gpu_check'))
cells.append(code([
    'import torch',
    'print(f"PyTorch版本: {torch.__version__}")',
    'print(f"CUDA可用: {torch.cuda.is_available()}")',
    'if torch.cuda.is_available():',
    '    print(f"GPU型号: {torch.cuda.get_device_name(0)}")',
    '    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")',
], 'check_gpu'))

# === CELL 3: Clone repo ===
cells.append(md(['## 2. 克隆项目代码', '', '从原始3DGS仓库克隆代码，然后应用改进。'], 'clone_repo'))
cells.append(code([
    '# 克隆3DGS项目（含submodules）',
    'import os, sys',
    'if not os.path.exists("gaussian-splatting"):',
    '    !git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive',
    '%cd gaussian-splatting',
    '',
    '# 创建输出目录（使用本地存储）',
    'OUTPUT_DIR = "/content/gaussian-splatting/output"',
    'os.makedirs(OUTPUT_DIR, exist_ok=True)',
    'print(f"输出目录: {OUTPUT_DIR}")',
], 'clone_code'))

# === CELL 4: Install deps ===
cells.append(md(['## 3. 安装依赖'], 'install_deps'))
cells.append(code([
    '!pip install torch torchvision --quiet',
    '!pip install tqdm plyfile opencv-python joblib --quiet',
    '',
    '# 编译CUDA算子（约2-3分钟）',
    '!pip install submodules/diff-gaussian-rasterization --quiet',
    '!pip install submodules/simple-knn --quiet',
    '!pip install submodules/fused-ssim --quiet',
    '',
    'print("\\n依赖安装完成!")',
], 'install_packages'))

# === CELL 5: Create VGG loss ===
cells.append(md([
    '## 4. 创建VGG感知损失模块',
    '',
    '这是本项目的核心改进。VGG感知损失使用预训练的VGG16网络提取多层特征，',
    '在特征空间计算L1距离，使渲染结果在感知质量上更接近真实图像。',
], 'create_vgg'))

vgg_code = [
    '%%writefile utils/vgg_loss.py',
    'import torch',
    'import torch.nn as nn',
    'import torchvision',
    '',
    'class VGGPerceptualLoss(nn.Module):',
    '    def __init__(self, device="cuda"):',
    '        super().__init__()',
    '        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)',
    '        self.features = vgg.features[:16].to(device).eval()',
    '        for p in self.features.parameters():',
    '            p.requires_grad = False',
    "        self.layers = {'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15}",
    "        self.layer_weights = {'relu1_2': 1.0/2.6, 'relu2_2': 1.0/4.8, 'relu3_3': 1.0/3.7}",
    '        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)',
    '        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)',
    '',
    '    def forward(self, pred, target):',
    '        if pred.dim() == 3:',
    '            pred = pred.unsqueeze(0)',
    '            target = target.unsqueeze(0)',
    '        pred_norm = (pred - self.mean) / self.std',
    '        target_norm = (target - self.mean) / self.std',
    '        pred_features = pred_norm',
    '        target_features = target_norm',
    '        total_loss = 0.0',
    "        for layer_name, layer_idx in self.layers.items():",
    "            start_idx = 0 if layer_name == 'relu1_2' else (4 if layer_name == 'relu2_2' else 9)",
    '            for i in range(start_idx, layer_idx + 1):',
    '                pred_features = self.features[i](pred_features)',
    '                target_features = self.features[i](target_features)',
    '            layer_loss = nn.functional.l1_loss(pred_features, target_features)',
    '            total_loss += self.layer_weights[layer_name] * layer_loss',
    '        return total_loss',
    '',
    'print("VGG感知损失模块已创建")',
]
cells.append(code(vgg_code, 'create_vgg_loss'))

# === CELL 6: Modify train.py ===
cells.append(md(['## 5. 修改train.py集成感知损失', '', '在训练循环中引入VGG感知损失。'], 'modify_train'))

patch_code = [
    "old = 'from utils.image_utils import psnr'",
    "new = 'from utils.image_utils import psnr\\nfrom utils.vgg_loss import VGGPerceptualLoss'",
    "with open('train.py') as f:",
    '    train_code = f.read()',
    'train_code = train_code.replace(old, new)',
    '',
    "old2 = 'parser.add_argument(\"--start_checkpoint\", type=str, default = None)\\n    args = parser.parse_args(sys.argv[1:])'",
    "new2 = 'parser.add_argument(\"--start_checkpoint\", type=str, default = None)\\n    parser.add_argument(\"--lambda_vgg\", type=float, default=0.1, help=\"VGG perceptual loss weight\")\\n    args = parser.parse_args(sys.argv[1:])'",
    'train_code = train_code.replace(old2, new2)',
    '',
    "old3 = 'background = torch.tensor(bg_color, dtype=torch.float32, device=\"cuda\")'",
    "new3 = 'background = torch.tensor(bg_color, dtype=torch.float32, device=\"cuda\")\\n    vgg_loss_fn = VGGPerceptualLoss(device=\"cuda\") if opt.lambda_vgg > 0 else None'",
    'train_code = train_code.replace(old3, new3)',
    '',
    "old4 = 'loss.backward()'",
    "new4 = 'if vgg_loss_fn is not None and opt.lambda_vgg > 0:\\n            perceptual_loss = vgg_loss_fn(image, gt_image)\\n            loss += opt.lambda_vgg * perceptual_loss\\n        loss.backward()'",
    'train_code = train_code.replace(old4, new4)',
    '',
    "with open('train.py', 'w') as f:",
    '    f.write(train_code)',
    '',
    "with open('arguments/__init__.py') as f:",
    '    args_code = f.read()',
    "old5 = 'self.optimizer_type = \"default\"\\n        super().__init__(parser, \"Optimization Parameters\")'",
    "new5 = 'self.optimizer_type = \"default\"\\n        self.lambda_vgg = 0.0\\n        super().__init__(parser, \"Optimization Parameters\")'",
    'args_code = args_code.replace(old5, new5)',
    "with open('arguments/__init__.py', 'w') as f:",
    '    f.write(args_code)',
    '',
    'print("train.py修改完成!")',
]
cells.append(code(patch_code, 'patch_train'))

# === CELL 7: Download data ===
cells.append(md([
    '## 6. 准备数据',
    '',
    '下载官方提供的Tanks&Temples数据集中的Truck场景（约200MB）。'
], 'data_prep'))
cells.append(code([
    'import urllib.request, zipfile, os',
    'data_dir = "/content/data"',
    'os.makedirs(data_dir, exist_ok=True)',
    'if not os.path.exists(f"{data_dir}/truck"):',
    '    print("正在下载Truck场景数据集...")',
    '    url = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"',
    '    urllib.request.urlretrieve(url, f"{data_dir}/tandt_db.zip")',
    '    with zipfile.ZipFile(f"{data_dir}/tandt_db.zip", "r") as zip_ref:',
    '        zip_ref.extractall(data_dir)',
    '    print("数据集下载完成!")',
    'else:',
    '    print("数据已存在")',
], 'download_data'))

# === CELL 8: Train original ===
cells.append(md([
    '## 7. 实验A：原始3DGS训练（无感知损失）',
    '',
    '作为对照组，先用 --lambda_vgg 0 训练原始3DGS。',
    '训练7000次迭代约需20分钟。'
], 'train_original'))
cells.append(code([
    '!python train.py \\',
    '    -s /content/data/truck \\',
    '    -m /content/gaussian-splatting/output/truck_original \\',
    '    --lambda_vgg 0 \\',
    '    --iterations 7000 \\',
    '    --test_iterations 7000 \\',
    '    --save_iterations 7000 \\',
    '    --quiet \\',
    '    --disable_viewer',
    '',
    'print("\\n原始3DGS训练完成!")',
], 'train_original_code'))

# === CELL 9: Train improved ===
cells.append(md([
    '## 8. 实验B：改进版3DGS训练（+VGG感知损失）',
    '',
    '使用 --lambda_vgg 0.2 启用VGG感知损失，其他参数保持一致。'
], 'train_improved'))
cells.append(code([
    '!python train.py \\',
    '    -s /content/data/truck \\',
    '    -m /content/gaussian-splatting/output/truck_improved \\',
    '    --lambda_vgg 0.2 \\',
    '    --iterations 7000 \\',
    '    --test_iterations 7000 \\',
    '    --save_iterations 7000 \\',
    '    --quiet \\',
    '    --disable_viewer',
    '',
    'print("\\n改进版3DGS训练完成!")',
], 'train_improved_code'))

# === CELL 10: Render & Evaluate ===
cells.append(md(['## 9. 渲染评估与对比', '', '对两个模型分别渲染并计算PSNR、SSIM、LPIPS指标。'], 'evaluate'))
cells.append(code([
    '!python render.py -m /content/gaussian-splatting/output/truck_original --quiet',
    '!python render.py -m /content/gaussian-splatting/output/truck_improved --quiet',
    '',
    '!python metrics.py -m /content/gaussian-splatting/output/truck_original',
    '!python metrics.py -m /content/gaussian-splatting/output/truck_improved',
    '',
    'print("\\n评估完成!")',
], 'render_eval'))

# === CELL 11: Visualize ===
cells.append(md(['## 10. 可视化对比'], 'visualize'))
viz_lines = [
    'import matplotlib.pyplot as plt',
    'import numpy as np',
    'from PIL import Image',
    'import os',
    '',
    'def load_images(model_path, split="test", iteration=7000):',
    '    render_dir = f"{model_path}/{split}/ours_{iteration}/renders"',
    '    gt_dir = f"{model_path}/{split}/ours_{iteration}/gt"',
    '    imgs = sorted(os.listdir(render_dir))',
    '    if not imgs:',
    '        return None, None',
    '    fname = imgs[0]',
    '    render = np.array(Image.open(f"{render_dir}/{fname}"))',
    '    gt = np.array(Image.open(f"{gt_dir}/{fname}"))',
    '    return render, gt',
    '',
    'orig_render, orig_gt = load_images("/content/gaussian-splatting/output/truck_original", "train")',
    'impr_render, impr_gt = load_images("/content/gaussian-splatting/output/truck_improved", "train")',
    '',
    'if orig_render is not None:',
    '    fig, axes = plt.subplots(2, 3, figsize=(15, 10))',
    '',
    '    axes[0, 0].imshow(orig_gt)',
    '    axes[0, 0].set_title("真实图像")',
    '    axes[0, 0].axis("off")',
    '',
    '    axes[0, 1].imshow(orig_render)',
    '    axes[0, 1].set_title("原始3DGS渲染")',
    '    axes[0, 1].axis("off")',
    '',
    '    diff_orig = np.abs(orig_render.astype(float) - orig_gt.astype(float)).mean(axis=2)',
    '    im = axes[0, 2].imshow(diff_orig, cmap="hot")',
    '    axes[0, 2].set_title(f"误差 (均值:{diff_orig.mean():.1f})")',
    '    axes[0, 2].axis("off")',
    '    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)',
    '',
    '    axes[1, 0].imshow(impr_gt)',
    '    axes[1, 0].set_title("真实图像")',
    '    axes[1, 0].axis("off")',
    '',
    '    axes[1, 1].imshow(impr_render)',
    '    axes[1, 1].set_title("改进版渲染 (+VGG感知损失)")',
    '    axes[1, 1].axis("off")',
    '',
    '    diff_impr = np.abs(impr_render.astype(float) - impr_gt.astype(float)).mean(axis=2)',
    '    im = axes[1, 2].imshow(diff_impr, cmap="hot")',
    '    axes[1, 2].set_title(f"误差 (均值:{diff_impr.mean():.1f})")',
    '    axes[1, 2].axis("off")',
    '    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)',
    '',
    '    plt.suptitle("3DGS改进前后对比 (VGG感知损失)", fontsize=16)',
    '    plt.tight_layout()',
    '    plt.savefig("/content/gaussian-splatting/output/comparison.png", dpi=150, bbox_inches="tight")',
    '    plt.show()',
    '    print("\\n对比图已保存!")',
    'else:',
    '    print("渲染结果尚未生成，请先执行渲染步骤")',
]
cells.append(code(viz_lines, 'show_comparison'))

# === CELL 12: Summary ===
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
], 'summary'))

# === CELL 13: Download results ===
cells.append(md([
    '## 12. 下载结果到本地',
    '',
    '打包所有结果文件供下载。如果你挂载了Google Drive，也可以手动复制到Drive。'
], 'download'))
cells.append(code([
    'import shutil, datetime',
    'timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")',
    'archive_name = f"3dgs_results_{timestamp}"',
    '',
    '# 创建压缩包',
    '!mkdir -p /content/{archive_name}',
    '!cp -r /content/gaussian-splatting/output/* /content/{archive_name}/ 2>/dev/null || true',
    '!cp /content/gaussian-splatting/output/comparison.png /content/{archive_name}/ 2>/dev/null || true',
    '',
    '!zip -r /content/{archive_name}.zip /content/{archive_name}/ > /dev/null 2>&1',
    '',
    'from google.colab import files',
    'files.download(f"/content/{archive_name}.zip")',
    '',
    'print(f"\\n已下载: {archive_name}.zip")',
    'print("包含: 训练模型 + 渲染结果 + 对比图")',
], 'download_results'))

# === WRITE ===
nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'accelerator': 'GPU',
        'colab': {'provenance': []}
    },
    'cells': cells
}

path = pathlib.Path('E:/Py Projects/gaussian-splatting-main/Colab_3DGS_Training.ipynb')
path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')

# Validate
data = json.loads(path.read_text(encoding='utf-8'))
size = len(path.read_bytes())
print(f"OK  cells={len(data['cells'])}  size={size} bytes")