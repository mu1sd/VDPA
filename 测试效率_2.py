# bench_models_base.py
import os, math, time, argparse, copy
from typing import List, Tuple, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn

# ------------------------ 可选依赖 ------------------------
def try_import_timm():
    try:
        import timm
        return timm
    except Exception as e:
        print(f"[WARN] timm 未安装：{e}")
        return None

def try_import_fvcore():
    try:
        from fvcore.nn import FlopCountAnalysis
        return FlopCountAnalysis
    except Exception as e:
        print(f"[WARN] fvcore 未安装或不可用：{e}")
        return None

def try_import_ptflops():
    try:
        from ptflops import get_model_complexity_info
        return get_model_complexity_info
    except Exception as e:
        print(f"[WARN] ptflops 未安装：{e}")
        return None

# ------------------------ VDPA（按需） ------------------------
def try_build_vdpa(num_classes=1000, img_size=512):
    """
    如果你的 VDPA 类在别的文件，请按需修改 import。
    """
    VDPA = None
    try:
        from final_model import VDPA as VDPA_CLS
        VDPA = VDPA_CLS
    except Exception:
        try:
            from vdpa_clean import VDPA as VDPA_CLS
            VDPA = VDPA_CLS
        except Exception as e2:
            print(f"[WARN] 未找到 VDPA 类，跳过 VDPA：{e2}")
            return None

    model = VDPA(
        img_size=img_size, patch_size=16, in_chans=3, num_classes=num_classes,
        embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4.0, attn_drop=0.0, drop=0.0,
        k_neighbors=9, alp_temperature=1.0, stk_temperature=0.5
    )
    return model

# ------------------------ 模型构建（Base 体量映射） ------------------------
# 第二列布尔：是否需要把 img_size 传给 timm（ViT/Swin 需要）
TIMM_MAP = {
    # ResNet（本身已是标准体量）
    "resnet50": ("resnet50", False),
    "resnet101": ("resnet101", False),

    # SE-ResNet
    "se-resnet50": ("seresnet50", False),
    "seresnet50": ("seresnet50", False),

    # DenseNet（121 是常用基准；如需更大可用 densenet201）
    "densenet121": ("densenet121", False),

    # MobileNet（使用 large_100）
    "mobilenetv2": ("mobilenetv2_100", False),
    "mobilenet v2": ("mobilenetv2_100", False),
    "mobilenetv3": ("mobilenetv3_large_100", False),
    "mobilenet v3": ("mobilenetv3_large_100", False),

    # VGG（vgg16 体量已大）
    "vgg16": ("vgg16", False),

    # EfficientNet（使用 B4，而非 B0）
    "efficientnet-b4": ("tf_efficientnet_b4", False),
    "efficientnet_b4": ("tf_efficientnet_b4", False),

    # ConvNeXt（base）
    "convnext": ("convnext_base", False),
    "convnext-base": ("convnext_base", False),
    "convnext_base": ("convnext_base", False),

    # ViT（base）
    "vit": ("vit_base_patch16_224", True),
    "vit-base": ("vit_base_patch16_224", True),
    "vit_base": ("vit_base_patch16_224", True),

    # Swin Transformer（base）
    "swin-transformer": ("swin_base_patch4_window7_224", True),
    "swin_transformer": ("swin_base_patch4_window7_224", True),
    "swin-base": ("swin_base_patch4_window7_224", True),
    "swin_base": ("swin_base_patch4_window7_224", True),
}

def build_model(name: str, num_classes: int, img_size: int):
    timm = try_import_timm()
    name = name.strip().lower()

    # 你的 VDPA
    if name in ["vdpa", "vdpa_full", "vdpa(ours)", "vdpa-ours", "vdpa_ours"]:
        m = try_build_vdpa(num_classes=num_classes, img_size=img_size)
        if m is None:
            raise RuntimeError("未能构建 VDPA，请检查 import 路径（见 try_build_vdpa）。")
        return m

    if timm is None:
        raise ImportError("需要安装 timm：pip install timm")

    if name not in TIMM_MAP:
        raise ValueError(f"未知模型名：{name}")

    timm_name, need_img = TIMM_MAP[name]
    kwargs = dict(pretrained=False, num_classes=num_classes)
    if need_img:
        # ViT/Swin 传入 img_size 以适配 pos_embed 与窗口
        kwargs["img_size"] = img_size
    return timm.create_model(timm_name, **kwargs)

# ------------------------ 计时（只在这里用 inference/AMP） ------------------------
def measure_latency(model: nn.Module,
                    device: str,
                    shape: Tuple[int, int, int, int],
                    amp: bool = False,
                    warmup: int = 50,
                    iters: int = 200) -> Tuple[float, float, float]:
    """
    返回：mean, median, std（毫秒）
    仅在此函数内部用 inference_mode + autocast，不影响后续 FLOPs 统计。
    """
    import numpy as np

    model.eval().to(device)
    x = torch.randn(shape, device=device)

    use_amp = bool(amp and device.startswith("cuda"))
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    with torch.inference_mode():
        # warmup
        for _ in range(max(1, warmup)):
            with amp_ctx:
                _ = model(x)
        sync()

        times = []
        if device.startswith("cuda"):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(max(1, iters)):
                starter.record()
                with amp_ctx:
                    _ = model(x)
                ender.record()
                sync()
                times.append(starter.elapsed_time(ender))
        else:
            t = []
            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                with amp_ctx:
                    _ = model(x)
                sync()
                t1 = time.perf_counter()
                t.append((t1 - t0) * 1000.0)
            times = t

    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.std())

# ------------------------ 参数量 & FLOPs ------------------------
def compute_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def _flops_fvcore(model: nn.Module, device: str, img: int, bs: int) -> Optional[float]:
    FlopCountAnalysis = try_import_fvcore()
    if FlopCountAnalysis is None:
        return None

    m = copy.deepcopy(model).eval().to(device)
    x = torch.randn((bs, 3, img, img), device=device).clone()

    try:
        # 注意：不要用 inference_mode，会让某些 op 的保存行为受限
        with torch.no_grad():
            flops = FlopCountAnalysis(m, x).total()
        return flops / 1e9
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and device.startswith("cuda"):
            print("[FLOPs] CUDA OOM，回退到 CPU 重试 ...")
            m_cpu = copy.deepcopy(model).eval().cpu()
            x_cpu = torch.randn((bs, 3, img, img))
            with torch.no_grad():
                flops = FlopCountAnalysis(m_cpu, x_cpu).total()
            return flops / 1e9
        raise
    finally:
        del m, x
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

def _flops_ptflops(model: nn.Module, img: int) -> Optional[float]:
    get_model_complexity_info = try_import_ptflops()
    if get_model_complexity_info is None:
        return None
    m = copy.deepcopy(model).eval().cpu()
    try:
        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                m, (3, img, img),
                as_strings=False, print_per_layer_stat=False, verbose=False
            )
        # ptflops 返回 MACs；通常 1 MAC ≈ 2 FLOPs
        return (2.0 * macs) / 1e9
    finally:
        del m
        torch.cuda.empty_cache()

def compute_flops_g(model: nn.Module, device: str, img: int, bs: int) -> Optional[float]:
    # 先尝试 fvcore
    try:
        gflops = _flops_fvcore(model, device, img, bs)
        if gflops is not None and math.isfinite(gflops):
            return gflops
    except Exception as e:
        print(f"[FLOPs] fvcore 统计失败：{e}")

    # 回退 ptflops（CPU）
    try:
        gflops = _flops_ptflops(model, img)
        if gflops is not None and math.isfinite(gflops):
            return gflops
    except Exception as e:
        print(f"[FLOPs] ptflops 统计失败：{e}")

    return None

# ------------------------ 主程序 ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        required=False,
        # 默认全部使用“base/大杯”命名：convnext/vit/swin-transformer 映射到 base
        default="resnet50,resnet101,seresnet50,densenet121,"
                "mobilenetv2,mobilenetv3,vgg16,efficientnet-b4,"
                "convnext,vit,swin-transformer,vdpa",
        help="以逗号分隔的模型名（见映射表，convnext/vit/swin-transformer 自动走 base 体量）"
    )
    parser.add_argument("--img", type=int, default=512)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--fp16", action="store_true", help="同时测 AMP（仅 CUDA 生效）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    # 建议开启 TF32（仅 CUDA）
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    names: List[str] = [n.strip() for n in args.models.split(",") if n.strip()]
    print(f"=== Bench: {names} | img={args.img} | bs={args.bs} | device={args.device} ===")

    for name in names:
        print(f"\n[{name}] building...")
        try:
            model = build_model(name, num_classes=args.num_classes, img_size=args.img)
        except Exception as e:
            print(f"[{name}] 构建失败：{e}")
            continue

        # 参数量
        params_m = compute_params_m(model)
        print(f"Params (M):           {params_m:6.2f}")

        # 延迟 FP32
        try:
            mean32, med32, std32 = measure_latency(
                model, device=args.device,
                shape=(args.bs, 3, args.img, args.img),
                amp=False, warmup=args.warmup, iters=args.iters
            )
            print(f"Latency FP32 (ms):    mean {mean32:.2f} | median {med32:.2f} | std {std32:.2f}")
        except Exception as e:
            print(f"Latency FP32 失败：{e}")

        # 延迟 AMP
        if args.fp16 and args.device == "cuda":
            try:
                mean16, med16, std16 = measure_latency(
                    model, device=args.device,
                    shape=(args.bs, 3, args.img, args.img),
                    amp=True, warmup=args.warmup, iters=args.iters
                )
                print(f"Latency AMP  (ms):    mean {mean16:.2f} | median {med16:.2f} | std {std16:.2f}")
            except Exception as e:
                print(f"Latency AMP 失败：{e}")

        # FLOPs（fvcore → ptflops 回退；CUDA OOM 则转 CPU）
        gflops = compute_flops_g(model, device=args.device, img=args.img, bs=args.bs)
        if gflops is None or not math.isfinite(gflops):
            print(f"GFLOPs@{args.img}^2:        N/A")
        else:
            print(f"GFLOPs@{args.img}^2:    {gflops:8.2f}")

        print("-" * 46)

if __name__ == "__main__":
    main()
