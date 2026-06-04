import os, glob, re, argparse
from torchreid.reid.data import ImageDataManager, register_image_dataset
from torchreid.reid.engine import ImageTripletEngine
from torchreid.reid import models, optim
from torchreid.reid.utils import load_pretrained_weights
try:
    from torchreid.reid.data import ImageDataset
except Exception:
    from torchreid.reid.data.datasets import ImageDataset

DATA_ROOT = "/home/murim/EYE-D-RE/data/market1501-v1"
MARKET_W = os.path.expanduser("~/.cache/torch/checkpoints/osnet_ain_x1_0_market.pth")
_PAT = re.compile(r'([-\d]+)_c(\d+)')

class EyeD(ImageDataset):
    def __init__(self, root='', **kwargs):
        def proc(sub, relabel):
            paths = sorted(glob.glob(os.path.join(DATA_ROOT, sub, '*.jpg')))
            pids = sorted({int(_PAT.search(os.path.basename(p)).group(1)) for p in paths})
            lab = {p:i for i,p in enumerate(pids)}
            return [(p, lab[int(_PAT.search(os.path.basename(p)).group(1))] if relabel
                     else int(_PAT.search(os.path.basename(p)).group(1)),
                     int(_PAT.search(os.path.basename(p)).group(2))-1) for p in paths]
        super().__init__(proc('bounding_box_train', True), proc('query', False),
                         proc('bounding_box_test', False), **kwargs)

register_image_dataset('eyed', EyeD)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--save-dir', type=str, default='log/ft11_ain_lastblock')
    a = ap.parse_args()
    dm = ImageDataManager(root='', sources='eyed', targets='eyed', height=256, width=128,
        batch_size_train=a.batch, batch_size_test=64,
        transforms=['random_flip','random_crop','random_erase'],
        train_sampler='RandomIdentitySampler', num_instances=4)
    print(f"[INFO] train IDs={dm.num_train_pids}")
    model = models.build_model('osnet_ain_x1_0', dm.num_train_pids, loss='triplet', pretrained=False)
    load_pretrained_weights(model, MARKET_W)
    n=0
    for name,p in model.named_parameters():
        p.requires_grad = name.startswith(('conv5','fc','classifier'))
        if p.requires_grad: n+=p.numel()
    print(f"[INFO] 학습 파라미터(마지막 블록): {n:,}")
    model = model.cuda()
    opt = optim.build_optimizer(model, optim='adam', lr=a.lr)
    sch = optim.build_lr_scheduler(opt, lr_scheduler='cosine', max_epoch=a.epochs)
    eng = ImageTripletEngine(dm, model, optimizer=opt, scheduler=sch,
        margin=0.3, weight_t=1.0, weight_x=1.0, label_smooth=True)
    eng.run(save_dir=a.save_dir, max_epoch=a.epochs, eval_freq=a.epochs, print_freq=20, test_only=False)
    print(f"[DONE] {a.save_dir}/model/model.pth.tar-{a.epochs}")

if __name__ == '__main__':
    main()
