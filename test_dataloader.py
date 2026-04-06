"""
test_dataloader.py — Verify all sampling contracts before training.

Run:
    python3 test_dataloader.py
"""
import sys, random
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import DATASET_ROOTS, DATASET_WEIGHTS, D6_A_OVERSAMPLE
from data.dataloader import (
    UniformTripletDataset, SharedPools,
    Dataset1Sampler, Dataset2Sampler, Dataset3Sampler,
    Dataset4Sampler, Dataset5Sampler, Dataset6Sampler,
    _scan, make_transform,
)
from models.model import build_model
from configs.config import EMBED_DIM, IMAGE_SIZE

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(name, cond, detail=""):
    if cond:
        print(f"[{PASS}] {name}" + (f"  — {detail}" if detail else ""))
    else:
        print(f"[{FAIL}] {name}" + (f"  — {detail}" if detail else ""))
        errors.append(name)

print("\n── 1. Dataset paths ────────────────────────────────────────────")
expected = {
    1: None,   # just check root exists
    2: None,
    3: None,
    4: {"A": 710,  "B": 363,  "C": 120,  "D": 570},
    5: {"A": 2894, "B": 573,  "C": 120,  "D": 29, "X": 1938},
    6: {"A": 1384, "X": 695},
}
for ds_id, classes in expected.items():
    root = DATASET_ROOTS[ds_id]
    check(f"D{ds_id} root exists", root.exists(), str(root))
    if classes:
        for cls, count in classes.items():
            folder = root / cls
            check(f"D{ds_id}/{cls} exists", folder.exists())
            if folder.exists():
                actual = len(_scan(folder))
                ok = abs(actual - count) / max(count, 1) < 0.05
                check(f"D{ds_id}/{cls} count ~{count}", ok, f"actual={actual}")

print("\n── 2. Shared pools ─────────────────────────────────────────────")
try:
    pools = SharedPools(DATASET_ROOTS)
    check("SharedPools created", True)
    check("D5 xclass pool non-empty",  len(pools.d5_xclass) > 0, f"len={len(pools.d5_xclass)}")
    check("D4 neg pool non-empty",     len(pools.d4_neg) > 0,    f"len={len(pools.d4_neg)}")
    d4_paths = set()
    for cls in ["A","B","C","D"]:
        d4_paths.update(_scan(DATASET_ROOTS[4] / cls))
    d4_in_neg = [p for p in pools.d4_neg if p in d4_paths]
    check("D4 neg pool has no intra-D4 images", len(d4_in_neg) == 0,
          f"found {len(d4_in_neg)}")
except Exception as e:
    check("SharedPools created", False, str(e))
    pools = None

print("\n── 3. Dataset_1 sampler ────────────────────────────────────────")
try:
    s1 = Dataset1Sampler(DATASET_ROOTS[1], train=False)
    check("D1 sampler created", True)
    check("D1 eligible keys > 0", len(s1.eligible_keys) > 0,
          f"keys={len(s1.eligible_keys)}")
    none_count = 0
    for _ in range(50):
        r = s1.sample()
        if r is None:
            none_count += 1
        elif not all(p.exists() for p in r):
            check("D1 triplet paths exist", False, str(r))
            break
    check("D1 samples mostly non-None", none_count < 10,
          f"{none_count}/50 returned None")
except Exception as e:
    check("D1 sampler created", False, str(e))

print("\n── 4. Dataset_6 sampler ────────────────────────────────────────")
if pools:
    try:
        s6 = Dataset6Sampler(DATASET_ROOTS[6], pools, train=False)
        check("D6 sampler created", True)
        raw_a = len(_scan(DATASET_ROOTS[6] / "A"))
        check("D6/A oversampled", len(s6.anchor_pool) == raw_a * D6_A_OVERSAMPLE,
              f"expected={raw_a*D6_A_OVERSAMPLE}  actual={len(s6.anchor_pool)}")
        d6_a = set(_scan(DATASET_ROOTS[6] / "A"))
        neg_violations = sum(1 for _ in range(100) if s6.sample()[2] in d6_a)
        check("D6 neg never from D6/A", neg_violations == 0,
              f"found {neg_violations}/100")
    except Exception as e:
        check("D6 sampler created", False, str(e))

print("\n── 5. Tensor shapes ────────────────────────────────────────────")
try:
    dataset = UniformTripletDataset(length=20, train=True)
    check("Dataset created", True)
    H, W = IMAGE_SIZE
    shape_ok = True
    for i in range(5):
        a, p, n = dataset[i]
        if a.shape != torch.Size([3, H, W]):
            shape_ok = False
            break
    check(f"Tensor shapes [3,{H},{W}]", shape_ok)
    finite_ok = all(
        torch.isfinite(t).all()
        for _ in range(5)
        for t in dataset[0]
    )
    check("No NaN/Inf in tensors", finite_ok)
except Exception as e:
    check("Dataset created", False, str(e))

print("\n── 6. Sampling weight distribution ────────────────────────────")
try:
    ds2 = UniformTripletDataset(length=500, train=False)
    counts = defaultdict(int)
    for _ in range(500):
        s = random.choices(ds2.samplers, weights=ds2.weights, k=1)[0]
        counts[s.stats()["name"]] += 1
    total = sum(counts.values())
    for name, cnt in sorted(counts.items()):
        print(f"  {name}: {cnt}/500 ({cnt/total*100:.1f}%)")
    d1_pct = counts.get("Dataset_1", 0) / total
    d6_pct = counts.get("Dataset_6", 0) / total
    check("D1 weight ~38% (±8%)", 0.30 <= d1_pct <= 0.46, f"actual={d1_pct:.2f}")
    check("D6 weight ~22% (±5%)", 0.17 <= d6_pct <= 0.27, f"actual={d6_pct:.2f}")
except Exception as e:
    check("Weight distribution", False, str(e))

print("\n── 7. Model forward pass ───────────────────────────────────────")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(device=device, freeze=False)
    dummy  = torch.randn(4, 3, *IMAGE_SIZE).to(device)
    with torch.no_grad():
        emb = model(dummy)
    check("Model forward pass", emb.shape == torch.Size([4, EMBED_DIM]),
          f"shape={emb.shape}")
    norms = emb.norm(dim=1)
    check("L2-normalised",
          torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
          f"norms min={norms.min():.4f} max={norms.max():.4f}")
    model.freeze_backbone()
    frozen = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
    check("Backbone freeze works", frozen > 0, f"frozen={frozen/1e6:.2f}M")
    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    check("Backbone unfreeze works", trainable > 0, f"trainable={trainable/1e6:.2f}M")
except Exception as e:
    check("Model forward pass", False, str(e))

print("\n" + "=" * 60)
if errors:
    print(f"  {len(errors)} FAILED: {errors}")
    sys.exit(1)
else:
    print("  All tests PASSED — ready to train.")
    print("  Run: python3 train.py")
print("=" * 60)
