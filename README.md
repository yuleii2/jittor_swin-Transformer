# Jittor 超声图像 BI-RADS 分类 | Swin Transformer + 渐进解冻

> 使用 **Jittor** 框架实现的乳腺超声图像 **BI-RADS 等级分类**基线方案，采用 **Swin Transformer** 主干与**渐进式解冻**训练策略，结合医学影像特有的数据增强与混合损失，提升在小样本与不平衡样本条件下的鲁棒性与泛化能力。

---

## 项目概述

**核心任务**  
基于 Jittor 搭建高性能的超声图像分类系统，准确识别乳腺超声图像的 **BI-RADS** 分类等级。

**解决思路**  
- 以 **Swin Transformer** 作为主干网络；  
- 采用**分阶段/渐进式解冻**策略（先训分类头 → 解冻深层 → 逐步扩展）；  
- 针对医学影像设计的增强：**病灶缩放、阴影条带、对比度扰动、噪声等**；  
- **混合损失**：标签平滑交叉熵 + 类别关系/分布约束，缓解类别边界模糊；  
- **TTA**（水平/垂直翻转、中心裁剪）提升推理稳定性；  
- 训练实用策略：**AMP 混合精度、余弦退火、AdamW（主干/头部分组学习率）、早停、梯度累积** 等。

---

## 环境与依赖

- OS：**Ubuntu 22.04.1 LTS**（示例环境）  
- GPU：**NVIDIA RTX 4090**（示例环境）  
- CUDA：**11.8 / 11.7**（均可）  
- Python：**≥ 3.8**  
- Jittor：**≥ 1.3.0**  
- 依赖生态：**jimm (Jittor Image Models)** 等

> 注：项目使用了 Jittor 推荐的 **jimm** 库，安装与使用请按其官方说明进行。

### 安装项目依赖
```bash
# 1) 通过 Conda 创建环境（基于 environment.yml）
conda env create -f environment.yml
conda activate <你的环境名>   # 将 <你的环境名> 替换为 environment.yml 中的 name

# 2) 若需要，可额外安装/升级 jittor 或其他包
# pip install -U jittor
```

---

## 数据准备与预处理

将原始训练集与标注整理好后，先进行类别均衡/可选离线增强，生成 `trainval_balanced.txt`（训练脚本会读取该文件）：

```bash
# 示例：在项目根目录执行
python code/balance_trainval.py   --src_txt  data/labels/trainval.txt   --img_root data/images/train   --dst_txt  data/labels/trainval_balanced.txt   --aug_root data/images/images_aug   --target_num 280
```

- `--src_txt`：原始标注，格式为 `img_path label`（相对或绝对路径均可）  
- `--dst_txt`：生成的均衡后标注文件  
- `--aug_root`：可选，指定离线增强后的图片输出目录  
- `--target_num`：长尾类别的目标样本数（按需调整）

目录示例：
```
data/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ trainval.txt
   └─ trainval_balanced.txt
```

---

## 训练

使用提供的训练脚本进行训练：
```bash
python code/test_5_train_1.py --cuda --amp 1
```

可在脚本/参数中按需补充常用配置，例如：
```bash
# 示例（按需）
# --epochs 100 --batch_size 32 --lr 5e-4 --weight_decay 1e-4
# --img_size 384 --swin_type base
# --freeze_stages 2 --unfreeze_at 10    # 渐进解冻
# --mixup 0.1 --cutmix 0.1
```

训练产物（可能因你的实现而异，一般包含）：
- `logs/train.log`：训练/验证曲线与关键信息  
- `weights/` 或 `runs/exp*/ckpts/`：保存的权重  
- `runs/exp*/samples/`：周期性可视化样例（如开启）

> 提示：由于数据规模较小，后期验证指标可能偏高。若将测试集并入训练仅用于演示或调参，请在正式评测中严格 **划分训练/验证/测试**，并复现实验设置。

---

## 推理 / 测试

使用测试/推理脚本：
```bash
# 最简用法（按你的 test.py 实现）：
python code/test.py

# 如需手动指定路径（按需扩展 test.py 参数）：
# python code/test.py #   --weights weights/best.ckpt #   --data    data/images/val #   --out     outputs/submission.csv #   --tta     1
```

---

## 权重下载（700MB 文件）

**推荐托管到 GitHub Releases**：上传后可在 README 中使用一键直链：
```bash
# 下载到本地 weights 目录
mkdir -p weights
curl -L -o weights/best.ckpt   "https://github.com/yuleii2/jittor_swin-Transformer/releases/download/v1.0/checkpoints1.pkl"

# 可选：校验和（请把下行的 expected_sha256 替换为真实值）
# echo "expected_sha256  weights/best.ckpt" | sha256sum -c -
```

> 若仓库尚未发布 Release，请先在 GitHub 仓库页面 **Draft a new release**，创建 tag（如 `v0.1.0`）并上传 `best.ckpt`。

---

## 模块说明（摘自实现要点）

- **模型构建**：Swin Transformer 主干，支持分阶段解冻与参数分组优化；  
- **医学数据增强**：病灶缩放、阴影条带、对比度扰动、噪声等；  
- **混合损失函数**：标签平滑交叉熵 + 类别关系/分布约束；  
- **动态训练策略**：解冻调度、余弦退火学习率、TTA；  
- **数据加载**：支持 `trainval_balanced.txt`，自适应绝对/相对路径；  
- **混合精度训练（AMP）**：提速并降低显存占用；  
- **早停机制**：基于验证损失监控，防止过拟合；  
- **优化器配置**：AdamW，主干与分类头区分学习率（建议 1:4）；  
- **训练流程控制**：完整训练/验证循环、模型保存、日志记录与梯度累积。

---

## 常见问题（FAQ）

- **`environment.yml` 用法**：请使用 `conda env create -f environment.yml` 创建，再 `conda activate <环境名>`。不要用 `pip install -r environment.yml`。  
- **jimm 安装问题**：先确认 Jittor 与 CUDA/驱动匹配，再安装 jimm。若使用 Conda，优先在全新环境中安装。  
- **验证率异常偏高**：数据规模较小且分布偏移时容易出现。请确认未将测试集混入正式训练流程；仅在调试/演示时可合并，最终评测需严格三划分。

---

## 致谢

- 感谢 **Jittor** 与 **jimm (Jittor Image Models)** 开源生态。

---

## 许可证

使用 **Apache-2.0**。请在仓库根目录放置 `LICENSE` 文件（或保留当前仓库内的 `LICENSE`）。
