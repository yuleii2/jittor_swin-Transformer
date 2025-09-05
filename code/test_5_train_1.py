#python test_5_train_1.py --cuda --amp
# ===============================================================
# 1. 通用依赖
# ===============================================================
import math, random, logging, argparse, gc
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    CenterCrop, Resize, ToTensor, ImageNormalize)
from tqdm import tqdm

# ===============================================================
# 2. timm → jimm -------------------------------------------------
# ===============================================================
import jimm
from jimm.models import registry as _reg
jimm.create_model = lambda n, **kw: _reg.model_entrypoint(n)(**kw)
WEIGHT = "/home/ly/.cache/torch/hub/checkpoints/swin_large_patch4_window12_384_22k.pth"
from jimm.models.swin_transformer import default_cfgs
default_cfgs["swin_large_patch4_window12_384_in22k"]["url"] = f"file://{WEIGHT}"

# ===============================================================
# 3. 路径 & 随机种子
# ===============================================================
DATA_ROOT   = Path("/home/ly/max/jittor/TrainSet")
IMG_TRAIN   = DATA_ROOT / "images/train"                     # 原始训练图片
LABEL_TRAIN = DATA_ROOT / "labels/trainval_balanced.txt"     # ★ 新训练标注
LABEL_VAL   = DATA_ROOT / "labels/val.txt"                   # 验证标注
OUTPUT_ROOT = Path("/home/ly/max/jittor/output/test_5_train_1")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); jt.seed(seed)

# ===============================================================
# 4. 工具函数
# ===============================================================
def clip_grad_norm_(params, opt, max_norm, eps=1e-6):
    total = 0.
    for p in params:
        g = p.opt_grad(opt)
        if g is not None:
            total += float(jt.sum(g**2))
    total = math.sqrt(total)
    if total > max_norm:
        coef = max_norm / (total + eps)
        for p in params:
            g = p.opt_grad(opt)
            if g is not None:
                g *= coef
    return total

def logger_to(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("train"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(p / "train.log")):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg

class EarlyStop:
    def __init__(self, pat=12, delta=1e-4):
        self.best = float("inf"); self.wait = 0
        self.pat, self.delta = pat, delta; self.best_epoch = 0
    def step(self, v, ep):
        if v < self.best - self.delta:
            self.best = v; self.wait = 0; self.best_epoch = ep
            return False
        self.wait += 1
        return self.wait >= self.pat

# ===============================================================
# 5. 数据集 & 增强
# ===============================================================
class ImageFolder(Dataset):
    """
    读取 'path label' 格式文本：
      - 若路径是绝对路径，直接打开
      - 若为相对路径，则与 root 拼接
    """
    def __init__(self, root, ann, tf=None):
        super().__init__(); self.root = Path(root); self.tf = tf
        with open(ann) as f:
            self.samples = [l.strip().split() for l in f]
        self.samples = [(p, int(c)) for p, c in self.samples]
        self.total_len = len(self.samples)

    def __getitem__(self, i):
        p, c = self.samples[i]
        pth = Path(p)
        full = pth if pth.is_absolute() else (self.root / pth)
        img = Image.open(full).convert("RGB")
        if self.tf:
            img = self.tf(img)
        return jt.array(img), c

from jittor.transform import (
    Compose, RandomResizedCrop, RandomHorizontalFlip,
    RandomAffine,  # ← 新增
)

class MedicalAug:
    """
    ● Step-1  RandomAffine:
        - scale 0.75–1.35  ↔  模拟病灶大小差异（放大/缩小）
        - translate ±8 %    ↔  病灶不一定在正中心
        - degrees 0         ↔  旋转仍交给后续小角度随机
    ● Step-2  RandomResizedCrop: 二次微调视野到 384×384
    ● Step-3  RandomHorizontalFlip: 超声左右翻
    ● Step-4  轻度旋转 / 对比度 / 噪声 / 阴影
    """
    def __init__(self):
        self.base = Compose([
            RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),   # 平移比例
                scale=(0.5, 1.5)),      # 缩放比例
            RandomResizedCrop((384, 384), (0.6, 1.0)),
            RandomHorizontalFlip()
        ])

    # ---------- 造阴影条带 ----------
    def add_shadow(self, img: Image.Image):
        arr = np.array(img, np.float32)
        h, w = arr.shape[:2]
        y  = random.randint(40, h // 2)
        h_s = random.randint(30, h // 3)
        mask = np.ones((h, w), np.float32)
        mask[y:y + h_s] *= random.uniform(0.3, 0.6)
        arr = np.clip(arr * mask[..., None], 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # ---------- 主调用 ----------
    def __call__(self, img: Image.Image):
        img = self.base(img)

        # 小角度旋转
        if random.random() < 0.4:
            img = img.rotate(random.randint(-8, 8))

        # 轻对比度扰动
        if random.random() < 0.4:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

        # 轻高斯噪声
        if random.random() < 0.4:
            arr = np.array(img, np.float32)
            arr += np.random.randn(*arr.shape) * 6
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # 加阴影
        if random.random() < 0.3:
            img = self.add_shadow(img)

        return img


# ===============================================================
# 6. 模型（渐进解冻）
# ===============================================================
class NetLarge(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.backbone = jimm.create_model(
            "swin_large_patch4_window12_384_in22k",
            pretrained=True, num_classes=nc)
        self.head_drop = nn.Dropout(0.5)

    def freeze_layers(self, ep):
        if ep < 5:                                        # 只训练 head
            for p in self.backbone.parameters(): p.stop_grad()
        elif ep < 15:                                    # + stage-4
            for n,p in self.backbone.named_parameters():
                p.start_grad() if ("layers.3" in n or "head" in n) else p.stop_grad()
        elif ep < 25:                                    # + stage-3 部分
            for n,p in self.backbone.named_parameters():
                if "layers.3" in n or "head" in n:
                    p.start_grad()
                elif "layers.2" in n and any(f"blocks.{i}" in n for i in [4,5,6]):
                    p.start_grad()
                else:
                    p.stop_grad()
        else:                                            # stage-3 全开
            for n,p in self.backbone.named_parameters():
                if any(s in n for s in ["layers.2","layers.3","head"]):
                    p.start_grad()
                else:
                    p.stop_grad()

    def execute(self, x):
        y = self.backbone(x)
        y = y[0] if isinstance(y, (tuple,list)) else y
        return self.head_drop(y)

# ===============================================================
# 7. Loss & Mix-Up
# ===============================================================
def mixup_data(x, y, ep, max_ep):
    alpha = 0.1 + 0.3 * ep / max_ep
    lam   = jt.array(np.random.beta(alpha, alpha, size=x.size(0))).float32()
    lam   = lam.view(-1,1,1,1)
    idx   = jt.randperm(x.shape[0]).stop_grad()
    return lam * x + (1-lam) * x[idx], y, y[idx], lam.squeeze()

class LSCE(nn.Module):
    def __init__(self, s=.1): super().__init__(); self.s=s
    def execute(self, pred, tar):
        tar = jt.array(tar).long(); C = pred.shape[1]
        one = jt.zeros_like(pred).scatter_(1, tar.unsqueeze(1), jt.float32(1))
        one = one*(1-self.s) + self.s/C
        return (-jt.sum(one * jt.nn.log_softmax(pred,1),1)).mean()

class BI_RADSLoss(nn.Module):
    def __init__(self, s=.1):
        super().__init__(); self.base=LSCE(s); self.epoch=0
        self.R = jt.array([
            [0.85,0.15,0,0,0,0],
            [0.10,0.75,0.15,0,0,0],
            [0,0.10,0.75,0.15,0,0],
            [0,0,0.15,0.75,0.10,0],
            [0,0,0,0.15,0.75,0.10],
            [0,0,0,0,0.05,0.95]], dtype=jt.float32).stop_grad()
    def execute(self, pred, tar):
        base = self.base(pred, tar)
        cont = jt.mean(jt.sum((nn.softmax(pred,1) - self.R[tar])**2,1))
        w = 0.1 if 15<=self.epoch<25 else 0.3
        return base + w*cont

def mixup_crit(crit, pred, y1, y2, lam):
    return (lam*crit(pred,y1)+(1-lam)*crit(pred,y2)).mean()

# ===============================================================
# 8. TTA
# ===============================================================
def tta_predict(model, img):
    H,W = int(img.shape[2]), int(img.shape[3])
    views = [img, img.flip(2), img.flip(3)]
    crop = int(min(H,W)*.9); y0,x0 = (H-crop)//2, (W-crop)//2
    crop_v = nn.interpolate(img[:,:,y0:y0+crop, x0:x0+crop],
                            size=(384,384), mode='bilinear')
    views.append(crop_v)
    return jt.mean(jt.stack([model(v) for v in views]),0)

# ===============================================================
# 9. Optimizer & LR
# ===============================================================
def make_optimizer(model, base_lr, wd):
    backbone, head = [], []
    for n,p in model.named_parameters():
        (head if "head" in n else backbone).append(p)
    return jt.optim.AdamW(
        [{"params":backbone,"lr":base_lr},
         {"params":head,    "lr":base_lr*4}],
        lr=base_lr, weight_decay=wd)

def adjust_lr(opt, ep, total_ep, base_lr):
    warm=15
    if ep < warm: ratio = ep/warm
    elif ep < 25: ratio = .3*.5*(1+math.cos(math.pi*(ep-warm)/10))
    else:         ratio = .5*(1+math.cos(math.pi*(ep-25)/(total_ep-25)))
    opt.param_groups[0]['lr'] = base_lr*ratio
    opt.param_groups[1]['lr'] = base_lr*4*ratio
    return ratio

# ===============================================================
# 10. Train / Eval
# ===============================================================
def train_one(ep, model, loader, opt, crit, max_ep, log, accum):
    model.train(); tot=lsum=steps=0; opt.zero_grad()
    grad_clip = .5 if ep>=15 else 1.0
    for i,(x,y) in enumerate(tqdm(loader, desc=f"Train[{ep}]")):
        xm,y1,y2,lam = mixup_data(x,y,ep,max_ep)
        loss = mixup_crit(crit, model(xm), y1,y2,lam) / accum
        opt.backward(loss)
        if (i+1)%accum==0:
            clip_grad_norm_(model.parameters(),opt,grad_clip)
            opt.step(); steps+=1
        tot += x.shape[0]; lsum += loss.item()*accum*x.shape[0]
    if (i+1)%accum!=0:
        clip_grad_norm_(model.parameters(),opt,grad_clip)
        opt.step(); steps+=1
    log.info(f"Ep{ep} TrainLoss {lsum/tot:.4f} steps={steps}")

def evaluate(model, loader, crit, tta=False):
    model.eval(); tot=lsum=corr=0
    with jt.no_grad():
        for x,y in loader:
            p = (jt.concat([tta_predict(model,x[i:i+1]) for i in range(x.shape[0])])
                 if tta else model(x))
            loss = crit(p,y)
            tot+=x.shape[0]; lsum+=loss.item()*x.shape[0]
            corr += (p.numpy().argmax(1)==np.array(y)).sum()
    return lsum/tot, corr/tot

def fit(model, tr_ld, vl_ld, cfg, log, outd):
    crit = BI_RADSLoss() if cfg.medical_loss else LSCE()
    opt  = make_optimizer(model, cfg.lr, wd=0.05)
    stopper, best = EarlyStop(), 0.
    for ep in range(1, cfg.epochs+1):
        model.freeze_layers(ep)
        if isinstance(crit, BI_RADSLoss): crit.epoch = ep
        train_one(ep, model, tr_ld, opt, crit, cfg.epochs, log, cfg.grad_accum)

        tta_flag = (ep%5==0 or ep==cfg.epochs)
        vl_loss, vl_acc = evaluate(model, vl_ld, crit, tta_flag)
        log.info(f"Ep{ep} ValLoss {vl_loss:.4f} Acc {vl_acc:.3%}"
                 f"{' (TTA)' if tta_flag else ''}")

        if vl_acc > best:
            best = vl_acc; model.save(str(outd/"best.pkl")); log.info("  ↳ new best")

        adjust_lr(opt, ep, cfg.epochs, cfg.lr)
        if stopper.step(vl_loss, ep):
            log.info(f"EarlyStop @Ep{ep} (best@Ep{stopper.best_epoch})"); break
        jt.gc(); gc.collect()
    log.info(f"Finished! best_val_acc={best:.3%}")

# ===============================================================
# 11. main
# ===============================================================
def main(cfg):
    set_seed(cfg.seed)
    jt.flags.use_cuda = int(cfg.cuda)
    if cfg.amp: jt.flags.auto_mixed_precision_level = 2

    outd = OUTPUT_ROOT
    log  = logger_to(outd)
    log.info(cfg)

    train_tf = Compose([MedicalAug(), ToTensor(),
                        ImageNormalize([.485,.456,.406],[.229,.224,.225])])
    val_tf   = Compose([Resize(384), CenterCrop(384), ToTensor(),
                        ImageNormalize([.485,.456,.406],[.229,.224,.225])])

    # 统计训练和验证集图片数量
    n_train = sum(1 for _ in open(LABEL_TRAIN))   # 行数 = 训练图片数
    n_val   = sum(1 for _ in open(LABEL_VAL))     # 行数 = 验证图片数
    log.info(f"Images  Train/Val  = {n_train}/{n_val}   batch_size={cfg.bs}")

    tr_ds = ImageFolder(IMG_TRAIN, LABEL_TRAIN, train_tf)
    vl_ds = ImageFolder(IMG_TRAIN, LABEL_VAL,   val_tf)
    tr_ld = tr_ds.set_attrs(batch_size=cfg.bs, shuffle=True , num_workers=4)
    vl_ld = vl_ds.set_attrs(batch_size=cfg.bs, shuffle=False, num_workers=2)
    log.info(f"Train/Val = {len(tr_ds)}/{len(vl_ds)}  bs={cfg.bs}")

    model = NetLarge(cfg.num_classes)
    if cfg.test:
        model.load(str(outd/"best.pkl"))
        print(evaluate(model, vl_ld, LSCE(), tta=True))
    else:
        fit(model, tr_ld, vl_ld, cfg, log, outd)

# ===============================================================
# 12. CLI
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=120)
    parser.add_argument("--bs",          type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--num_classes", type=int,   default=6)
    parser.add_argument("--grad_accum",  type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--cuda",        action="store_true")
    parser.add_argument("--amp",         action="store_true")
    parser.add_argument("--medical_loss",action="store_true")
    parser.add_argument("--test",        action="store_true")
    main(parser.parse_args())