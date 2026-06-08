import os, glob, re, argparse
import torch
from tqdm import tqdm
from torchreid.reid.data import ImageDataManager, register_image_dataset
from torchreid.reid.engine import ImageTripletEngine
from torchreid.reid import models, optim
from torchreid.reid.utils import load_pretrained_weights
try:
    from torchreid.reid.data import ImageDataset
except Exception:
    from torchreid.reid.data.datasets import ImageDataset


class ProgressEngine(ImageTripletEngine):
    """에포크 + 배치 단위 tqdm 진행바를 추가한 엔진."""

    def run(self, save_dir, max_epoch=0, **kwargs):
        self._epoch_bar = tqdm(
            total=max_epoch,
            desc="Epochs",
            unit="ep",
            position=0,
            dynamic_ncols=True,
        )
        try:
            super().run(save_dir, max_epoch=max_epoch, **kwargs)
        finally:
            self._epoch_bar.close()

    def train(self, **kwargs):
        # 배치 단위 inner bar: 에포크가 끝나면 사라짐 (leave=False)
        original_loader = self.train_loader
        self.train_loader = tqdm(
            original_loader,
            desc=f"  ep {self.epoch + 1:>3}/{self.max_epoch}",
            unit="batch",
            leave=False,
            position=1,
            dynamic_ncols=True,
        )
        try:
            super().train(**kwargs)
        finally:
            self.train_loader = original_loader

        postfix = {}
        try:
            postfix["lr"] = f"{self.get_current_lr():.2e}"
        except Exception:
            pass
        self._epoch_bar.set_postfix(postfix)
        self._epoch_bar.update(1)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "market1501-v1")
_PAT = re.compile(r'([-\d]+)_c(\d+)')

SUPPORTED_MODELS = ['osnet_x1_0', 'osnet_ain_x1_0']

class EyeD(ImageDataset):
    def __init__(self, root='', **kwargs):
        def proc(sub, relabel):
            paths = sorted(glob.glob(os.path.join(DATA_ROOT, sub, '*.jpg')))
            pids = sorted({int(_PAT.search(os.path.basename(p)).group(1)) for p in paths})
            lab = {p: i for i, p in enumerate(pids)}
            return [(p, lab[int(_PAT.search(os.path.basename(p)).group(1))] if relabel
                     else int(_PAT.search(os.path.basename(p)).group(1)),
                     int(_PAT.search(os.path.basename(p)).group(2)) - 1) for p in paths]
        super().__init__(proc('bounding_box_train', True), proc('query', False),
                         proc('bounding_box_test', False), **kwargs)

register_image_dataset('eyed', EyeD)

def main():
    ap = argparse.ArgumentParser(description="EYE-D OSNet fine-tune (last block)")
    ap.add_argument('--model', type=str, default='osnet_x1_0', choices=SUPPORTED_MODELS,
                    help=f"모델 아키텍처 (기본: osnet_x1_0). 선택: {SUPPORTED_MODELS}")
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.0005)
    ap.add_argument('--save-dir', type=str, default=None,
                    help="체크포인트 저장 경로 (기본: log/ft_<model>_lastblock)")
    ap.add_argument('--weights', type=str, default=None,
                    help="사전학습 가중치 경로 (기본: ~/.cache/torch/checkpoints/<model>_market.pth)")
    a = ap.parse_args()

    if a.weights:
        weights_path = a.weights
    else:
        p1 = os.path.expanduser(f"~/.cache/torch/checkpoints/{a.model}_market1501.pth")
        p2 = os.path.expanduser(f"~/.cache/torch/checkpoints/{a.model}_market.pth")
        if os.path.exists(p1):
            weights_path = p1
        elif os.path.exists(p2):
            weights_path = p2
        else:
            weights_path = p1
    save_dir = a.save_dir or f"log/ft_{a.model}_lastblock"

    use_gpu = torch.cuda.is_available()
    print(f"[INFO] model    = {a.model}")
    print(f"[INFO] weights  = {weights_path}")
    print(f"[INFO] save_dir = {save_dir}")
    print(f"[INFO] use_gpu  = {use_gpu}")

    dm = ImageDataManager(root='', sources='eyed', targets='eyed', height=256, width=128,
        batch_size_train=a.batch, batch_size_test=64,
        transforms=['random_flip', 'random_crop', 'random_erase'],
        train_sampler='RandomIdentitySampler', num_instances=4, use_gpu=use_gpu)
    print(f"[INFO] train IDs={dm.num_train_pids}")

    model = models.build_model(a.model, dm.num_train_pids, loss='triplet', pretrained=False, use_gpu=use_gpu)
    load_pretrained_weights(model, weights_path)

    n = 0
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(('conv5', 'fc', 'classifier'))
        if p.requires_grad:
            n += p.numel()
    print(f"[INFO] 학습 파라미터(마지막 블록): {n:,}")

    if use_gpu:
        model = model.cuda()

    opt = optim.build_optimizer(model, optim='adam', lr=a.lr)
    sch = optim.build_lr_scheduler(opt, lr_scheduler='cosine', max_epoch=a.epochs)
    eng = ProgressEngine(dm, model, optimizer=opt, scheduler=sch,
        margin=0.3, weight_t=1.0, weight_x=1.0, label_smooth=True, use_gpu=use_gpu)
    eng.run(save_dir=save_dir, max_epoch=a.epochs, eval_freq=a.epochs, print_freq=20, test_only=False)
    print(f"[DONE] {save_dir}/model/model.pth.tar-{a.epochs}")

if __name__ == '__main__':
    main()
