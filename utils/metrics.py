"""utils/metrics.py"""
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def compute_triplet_accuracy(emb_a: torch.Tensor,
                              emb_p: torch.Tensor,
                              emb_n: torch.Tensor) -> float:
    with torch.no_grad():
        d_ap = torch.nn.functional.pairwise_distance(emb_a, emb_p, p=2)
        d_an = torch.nn.functional.pairwise_distance(emb_a, emb_n, p=2)
        return (d_ap < d_an).float().mean().item()
