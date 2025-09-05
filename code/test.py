#python test.py
import os, argparse, numpy as np
from pathlib import Path
from PIL import Image
import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, ToTensor, ImageNormalize
from tqdm import tqdm

# ---------- 常量路径 ----------
DATA_DIR   = Path("/home/ly/max/jittor/TestSetA")
WEIGHT_PKL = Path("/home/ly/max/jittor/output/test_5_train_1/best.pkl")
OUT_TXT    = Path("/home/ly/max/jittor/output/test_5_train_1/result.txt")
IMG_SIZE   = 512
NUM_CLASS  = 6
BATCH      = 8            # 显存够可调大

# ---------- jimm patch ----------
import jimm
from jimm.models import registry as _reg
jimm.create_model = lambda n, **kw: _reg.model_entrypoint(n)(**kw)
# ---------------------------------

# =========== Dataset ============
class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root); self.tf = transform
        self.files = sorted(os.listdir(root))        # 0.jpg 1.jpg …
        self.total_len = len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        img = Image.open(self.root/fn).convert("RGB")
        if self.tf: img = self.tf(img)
        return jt.array(img), fn

# =========== Model ==============
class Net(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        # *** 必须与训练时模型一致 —> S 版 ***
        self.backbone = jimm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=NUM_CLASS
        )
        self.load(str(ckpt))  # str 保持兼容

    def execute(self, x):
        y = self.backbone(x)
        return y[0] if isinstance(y, (tuple, list)) else y

# =========== Inference ==========
def infer(model, loader, outfile: Path):
    model.eval(); preds=[]; names=[]
    print("Running inference …")
    for img, fn in tqdm(loader):
        logit = model(img); logit.sync()
        preds.append(logit.numpy().argmax(1)); names += list(fn)

    preds = np.concatenate(preds)
    pairs = sorted(zip(names, preds),
                   key=lambda x: int(os.path.splitext(x[0])[0]))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        for n, p in pairs:
            f.write(f"{n} {p}\n")
    print("Done! Result saved →", outfile)

# =========== Main ===============
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--cuda", action="store_true",
                      help="加 --cuda 用 GPU 推理")
    argp.add_argument("--bs", type=int, default=BATCH)
    cfg = argp.parse_args()

    jt.flags.use_cuda = int(cfg.cuda)

    tf = Compose([
        Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        ImageNormalize([.485,.456,.406],
                       [.229,.224,.225])
    ])
    ds = ImageFolder(DATA_DIR, tf)
    ld = ds.set_attrs(batch_size=cfg.bs,
                      shuffle=False, num_workers=4)

    model = Net(WEIGHT_PKL)
    infer(model, ld, OUT_TXT)
