#python balance_trainval.py     --src_txt   /home/ly/max/jittor/TrainSet/labels/trainval.txt     --img_root  /home/ly/max/jittor/TrainSet/images/train     --dst_txt   /home/ly/max/jittor/TrainSet/labels/trainval_balanced.txt     --aug_root  /home/ly/max/jittor/TrainSet/images/images_aug     --target_num 280
# -*- coding: utf-8 -*-
"""
平衡 trainval.txt：对样本数不足 target_num 的类别做复制 + 轻量增强
"""

import os, random, argparse, pathlib
from collections import defaultdict
from PIL import Image, ImageOps, ImageEnhance

# ----------------- 轻量增强 -----------------
def simple_augment(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
    if random.random() < 0.3:
        img = img.rotate(random.uniform(-15, 15), expand=True)
    if random.random() < 0.3:
        c = random.uniform(0.8, 1.2)
        l = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(ImageEnhance.Contrast(img).enhance(c)).enhance(l)
    return img

def to_full_path(path_str, root):
    """若 path_str 不是绝对路径，则与 root 拼接"""
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else root / p

def main(cfg):
    cfg.aug_root.mkdir(parents=True, exist_ok=True)

    # ---------- 读取 & 统计 ----------
    label2paths = defaultdict(list)
    with open(cfg.src_txt, 'r') as f:
        for line in f:
            path_str, lab = line.strip().split()
            full = to_full_path(path_str, cfg.img_root)
            label2paths[int(lab)].append(str(full))

    # ---------- 写新 txt ----------
    with open(cfg.dst_txt, 'w') as fout:
        # 1) 先写原行（保持顺序 & 兼容绝对/相对）
        with open(cfg.src_txt, 'r') as fin:
            for line in fin:
                fout.write(line)

        # 2) 对少数类补足
        for lab, paths in label2paths.items():
            deficit = cfg.target_num - len(paths)
            if deficit <= 0:
                continue
            print(f"[INFO] 类别 {lab} 需补 {deficit} 张")
            for i in range(deficit):
                src_path = pathlib.Path(random.choice(paths))
                try:
                    img = Image.open(src_path).convert("RGB")
                except FileNotFoundError:
                    print(f"[WARN] 找不到 {src_path}，跳过")
                    continue
                aug = simple_augment(img)

                new_name = f"{src_path.stem}__aug{i:02d}{src_path.suffix}"
                new_path = cfg.aug_root / new_name
                aug.save(new_path, quality=95)

                fout.write(f"{new_path} {lab}\n")

    print(f"[DONE] 写入平衡后的标注 → {cfg.dst_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_txt",   type=pathlib.Path, required=True)
    parser.add_argument("--img_root",  type=pathlib.Path, required=True)
    parser.add_argument("--dst_txt",   type=pathlib.Path, required=True)
    parser.add_argument("--aug_root",  type=pathlib.Path, required=True)
    parser.add_argument("--target_num", type=int, default=280)
    args = parser.parse_args()
    main(args)