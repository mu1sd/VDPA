# /home/measure_models.py  或 /home/AIM/测试效率.py
import os, sys, csv, argparse, numpy as np
import torch
import torch.nn as nn

# ====== 你自己的 VDPA（保持和你论文一致）======
try:
    # 如果你的类文件叫 final_model.py，这里就写 from final_model import VDPA
    from final_model import VDPA
except Exception:
    try:
        from vdpa_clean import VDPA
    except Exception:
        VDPA = None

# ====== 可选：timm（ConvNeXtV2 / MaxViT 用） ======
def try_import_timm():
    try:
        import timm
        return timm
    except Exception:
        return None

# ====== ptflops / thop ======
def try_import_ptflops():
    try:
        from ptflops import get_model_complexity_info
        return get_model_complexity_info
    except Exception:
        return None

def try_import_thop():
    try:
        from thop import profile
        return profile
    except Exception:
        return None

# ====== 统一计时 ======
@torch.inference_mode()
def measure_latency(model, device, shape=(1,3,512,512), amp=False, warmup=50, iters=200):
    x = torch.randn(shape, device=device)
    model.eval().to(device)

    if amp:
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        class Dummy:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        autocast = Dummy()

    # warmup
    for _ in range(warmup):
        with autocast:
            _ = model(x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        starter.record()
        with autocast:
            _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    times = np.array(times, dtype=np.float64)
    return float(times.mean()), float(np.median(times)), float(times.std())

# ====== 统一 FLOPs（优先 ptflops，失败用 thop） ======
def measure_flops(model, img=512, device="cuda"):
    get_mci = try_import_ptflops()
    if get_mci is not None:
        try:
            macs, _ = get_mci(model, (3, img, img),
                              as_strings=False, print_per_layer_stat=False, verbose=False)
            return macs / 1e9   # 将 MACs 视为 FLOPs 报告（论文里统一声明口径）
        except Exception:
            pass
    # fallback to thop
    profile = try_import_thop()
    if profile is not None:
        model = model.to(device).eval()
        x = torch.randn(1,3,img,img).to(device)
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9
    return float("nan")

# ====== 模型构造：根据名字创建（各分支独立！） ======
def build_model(name, args):
    name = name.lower()
    timm = try_import_timm()

    # ---- 1) 你的 VDPA ----
    if name in ["vdpa", "vdpa_full", "vdpa(ours)"]:
        assert VDPA is not None, "未找到 VDPA 类，请确认 from final_model import VDPA / vdpa_clean 导入路径"
        return VDPA(
            img_size=args.img, patch_size=16, in_chans=3, num_classes=2,
            embed_dim=args.embed, depth=args.depth, num_heads=args.heads,
            mlp_ratio=4.0, attn_drop=0.0, drop=0.0,
            k_neighbors=9, alp_temperature=1.0, stk_temperature=0.5
        )

    # ---- 2) ConvNeXt V2 Tiny（timm）----
    if name in ["convnextv2_tiny", "convnextv2-tiny", "convnextv2"]:
        assert timm is not None, "需要安装 timm：pip install timm"
        return timm.create_model('convnextv2_tiny', pretrained=False, num_classes=2)

    # ---- 3) MaxViT Tiny（timm，创建后把窗口强制改为 8 支持 512×512）----
        # ---- 3) MaxViT Tiny（timm，保持默认窗口，timm 会自动 pad）----
    if name in ["maxvit_tiny", "maxvit-tiny", "maxvit"]:
        assert timm is not None, "需要安装 timm：pip install timm"
        # 尝试更大输入默认变体；若不可用则回退到 224 版本
        for variant in ["maxvit_tiny_tf_512", "maxvit_tiny_tf_384",
                        "maxvit_tiny_rw_256", "maxvit_tiny_tf_224"]:
            try:
                return timm.create_model(variant, pretrained=False, num_classes=2)
            except Exception:
                continue
        raise RuntimeError("无法创建 MaxViT Tiny，检查 timm 版本或变体名称。")


    # ---- 4) VMamba Tiny（本地仓库）----
    if name in ["vmamba_tiny", "vmamba-tiny", "vmamba"]:
        import importlib.util
        cand = []
        if args.vmamba_dir:
            cand.append(args.vmamba_dir)
            cand.append(os.path.join(args.vmamba_dir, "classification"))
            cand.append(os.path.join(args.vmamba_dir, "image_classification"))
        else:
            cand += [
                "/home/AIM/VMamba-main/classification",
                "/home/AIM/VMamba-main",
                "/home/VMamba-main/classification",
                "/home/VMamba-main",
            ]

        tried_msgs, backbone, feat_dim = [], None, 768
        for d in cand:
            if not d or not os.path.isdir(d):
                continue
            if d not in sys.path:
                sys.path.insert(0, d)

            # 入口 1：classification/models/vmamba.py: vmamba_tiny
            try:
                if importlib.util.find_spec("models.vmamba") is not None:
                    from models.vmamba import vmamba_tiny
                    backbone = vmamba_tiny(pretrained=False)
                    feat_dim = getattr(backbone, "num_features", feat_dim)
                    break
            except Exception as e:
                tried_msgs.append(f"{d}: models.vmamba import error: {e}")

            # 入口 2：vmamba/vmamba.py: VMambaTiny
            try:
                if importlib.util.find_spec("vmamba.vmamba") is not None:
                    from vmamba.vmamba import VMambaTiny
                    backbone = VMambaTiny(pretrained=False)
                    feat_dim = getattr(backbone, "num_features", feat_dim)
                    break
            except Exception as e:
                tried_msgs.append(f"{d}: vmamba.vmamba import error: {e}")

        if backbone is None:
            msg = "未能导入 VMamba Tiny。\n已尝试:\n  - " + "\n  - ".join(tried_msgs) + \
                  "\n请将 --vmamba_dir 指向**包含 models/ 的目录**（通常是仓库的 classification 子目录），" \
                  "或把入口替换为你仓库的真实构造函数，并设置 feat_dim（最后一层通道数）。"
            raise ImportError(msg)

        class VMambaTinyWrapper(nn.Module):
            def __init__(self, backbone, feat_dim, num_classes=2):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(feat_dim, num_classes)
            def forward(self, x):
                feats = self.backbone(x)
                if feats.dim() == 4:      # [B,C,H,W]
                    g = feats.mean(dim=[2,3])
                elif feats.dim() == 3:    # [B,N,C]
                    g = feats.mean(dim=1)
                else:
                    raise RuntimeError(f"Unexpected feature shape: {feats.shape}")
                return self.head(g)

        return VMambaTinyWrapper(backbone, feat_dim, num_classes=2)

    raise ValueError(f"未知模型名：{name}")


# ====== 主流程：批量评测并写 CSV ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str,
                    default="vdpa,convnextv2_tiny,maxvit,vmamba_tiny",
                    help="逗号分隔的模型名列表")
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--embed", type=int, default=512)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", help="同时测 AMP 延迟")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--vmamba_dir", type=str, default=None, help="VMamba 仓库目录（指到包含 models/ 的那层，如 .../classification）")
    ap.add_argument("--csv", type=str, default="/home/efficiency_results.csv")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    print("\n=========== Efficiency Benchmark ===========")
    print(f"Input: 3x{args.img}x{args.img}, BS={args.bs}, FP32{' + AMP' if args.fp16 else ''}")
    print("============================================\n")

    for name in models:
        print(f"[{name}] building...")
        model = build_model(name, args)

        # Params
        params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        # Latency（先测，防止 FLOPs 出错中断）
        mean_ms, med_ms, std_ms = measure_latency(model, device,
                                                  shape=(args.bs,3,args.img,args.img), amp=False)
        if args.fp16:
            mean16, med16, std16 = measure_latency(model, device,
                                                   shape=(args.bs,3,args.img,args.img), amp=True)
        else:
            mean16 = med16 = std16 = None

        # FLOPs（最后测）
        gflops = measure_flops(model, img=args.img, device=device)

        print(f"Params (M):           {params_m:.2f}")
        print(f"Latency FP32 (ms):    mean {mean_ms:.2f} | median {med_ms:.2f} | std {std_ms:.2f}")
        if mean16 is not None:
            print(f"Latency AMP  (ms):    mean {mean16:.2f} | median {med16:.2f} | std {std16:.2f}")
        print(f"GFLOPs@{args.img}^2:  {gflops:.2f}" if np.isfinite(gflops) else "GFLOPs:  NaN")
        print("-"*46)

        rows.append({
            "Model": name,
            "Params(M)": f"{params_m:.2f}",
            f"Latency_FP32_mean(ms)": f"{mean_ms:.2f}",
            f"Latency_FP32_median(ms)": f"{med_ms:.2f}",
            f"Latency_FP32_std(ms)": f"{std_ms:.2f}",
            f"Latency_AMP_median(ms)": f"{med16:.2f}" if med16 is not None else "",
            f"GFLOPs@{args.img}^2": f"{gflops:.2f}" if np.isfinite(gflops) else "NaN",
        })

        del model
        torch.cuda.empty_cache()

    # 写 CSV
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved CSV to: {args.csv}\nDone.")

if __name__ == "__main__":
    main()
