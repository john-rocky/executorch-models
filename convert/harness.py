"""Common torch.export -> XNNPACK .pte harness: parity gate vs torch fp32 + CPU timing.

Usage from a per-model script:
    from harness import convert_and_gate
    convert_and_gate("dinov2_vits14", model, (x,))                     # fp32
    convert_and_gate("dinov2_vits14", model, (x,), precision="fp16")   # fp16 compute, fp32 I/O
    convert_and_gate("dinov2_vits14", model, (x,), precision="int8",
                     calibrate=lambda m: [m(b) for b in batches])      # PT2E static int8

Outputs <repo>/pte/<name>_xnnpack_<precision>.pte and
<repo>/results/<name>.json (fp32, legacy name) / <name>_<precision>.json.
Every result carries a delegation-coverage report: ops that stayed on portable
are the re-authoring candidate list.
"""
import copy
import json
import operator
import os
import statistics
import sys
import time
from collections import Counter

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTE_DIR = os.path.join(REPO, "pte")
RESULTS_DIR = os.path.join(REPO, "results")

# Print a WARN when worst-output corr vs fp32 eager drops below these.
CORR_GATE = {"fp32": 0.999, "fp16": 0.995, "int8": 0.95}

# Every model ships on both backends. XNNPACK is CPU-only and portable (the same
# file runs on Android); Core ML reaches the Neural Engine and measured 3.5-13.9x
# faster on device (median 12x, see KNOWLEDGE). A convert_and_gate call for fp32
# therefore emits both files unless CONVERT_BACKEND narrows it to one, which is
# what the A/B scripts do. Core ML computes in fp16, so it answers to the fp16
# correlation gate rather than fp32's.
BACKEND = os.environ.get("CONVERT_BACKEND", "xnnpack")
ALSO_COREML = os.environ.get("CONVERT_BACKEND") is None and sys.platform == "darwin"
COREML_UNIT = os.environ.get("CONVERT_COREML_UNIT", "all")
# fp16 is what reaches the Neural Engine, so it is the default. Models whose
# decoders iterate (RT-DETRv2, D-FINE) lose too much in fp16 — their XNNPACK fp16
# builds were already withdrawn for it — and can be built fp32 instead, which
# stays on GPU/CPU but may still beat XNNPACK.
COREML_PRECISION = os.environ.get("CONVERT_COREML_PRECISION", "fp16")


# Ops coremltools will only accept in floating point. A constant that began life
# as a Python int stays integer through torch.export, and these are where it
# stops being harmless.
_FLOAT_ONLY_OPS = (
    torch.ops.aten.sqrt.default, torch.ops.aten.rsqrt.default,
    torch.ops.aten.exp.default, torch.ops.aten.log.default,
    torch.ops.aten.sin.default, torch.ops.aten.cos.default,
    torch.ops.aten.reciprocal.default, torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
)


def _fix_dtypes_for_coreml(ep):
    """Two dtype rules Core ML has and PyTorch does not, both from Python ints.

    float64 is not narrowed by coremltools so much as mistyped: the operand comes
    out the other side as an int, and the failure surfaces far away as
    `matmul ... got x as int32 and y as fp32`. RT-DETRv2 and D-FINE hit that
    through `torch.outer` in their anchor tables.

    Integer input to a float-only op is the other half — RAFT-small computes a
    normalisation constant from a Python int and feeds it to `sqrt`, which Core ML
    refuses outright.

    Cast at the operand rather than the producer: the same value may be consumed
    elsewhere by an op that is happy with it. Uses `_to_copy` because the EXIR
    frontend does not accept `to.dtype`.
    """
    gm = ep.graph_module
    cast = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        float_only = node.target in _FLOAT_ONLY_OPS
        args = []
        for a in node.args:
            v = getattr(a, "meta", {}).get("val", None) if hasattr(a, "meta") else None
            dt = getattr(v, "dtype", None)
            bad = dt == torch.float64 or (
                float_only and dt is not None and not dt.is_floating_point)
            if bad:
                with gm.graph.inserting_before(node):
                    a = gm.graph.call_function(torch.ops.aten._to_copy.default,
                                               (a,), {"dtype": torch.float32})
                cast += 1
            args.append(a)
        node.args = tuple(args)
    if cast:
        # torch.export also records `_assert_tensor_metadata` nodes stating the
        # dtype it saw. Retyping an operand makes those assertions false, and the
        # graph fails to re-trace with "Tensor dtype mismatch! Expected float64".
        # They carry no output, so erasing them is safe.
        for node in reversed(list(gm.graph.nodes)):
            if (node.op == "call_function"
                    and node.target is torch.ops.aten._assert_tensor_metadata.default):
                gm.graph.erase_node(node)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        print(f"  cast {cast} operand(s) to float32 for Core ML")
    return ep


def _vulkan_partitioner():
    """Android's GPU path. Unlike Core ML there is no host runtime to check
    against — a Vulkan .pte cannot execute on the Mac — so its parity is measured
    on the device instead and the host result records that."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    return VulkanPartitioner()


def _coreml_partitioner():
    """compute_unit is baked into the .pte at compile time, not chosen at load."""
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner

    units = {"all": ct.ComputeUnit.ALL,
             "ne": ct.ComputeUnit.CPU_AND_NE,
             "gpu": ct.ComputeUnit.CPU_AND_GPU,
             "cpu": ct.ComputeUnit.CPU_ONLY}
    specs = CoreMLBackend.generate_compile_specs(
        compute_precision=(ct.precision.FLOAT32 if COREML_PRECISION == "fp32"
                           else ct.precision.FLOAT16),
        compute_unit=units[COREML_UNIT],
        minimum_deployment_target=ct.target.iOS17,
    )
    return CoreMLPartitioner(compile_specs=specs)


def _require_contiguous(inputs, what):
    """The ExecuTorch runtime reads an input tensor as if it were contiguous and
    ignores its strides — no error, just wrong numbers. PyTorch honours strides,
    so a non-contiguous input makes .pte and eager disagree and looks exactly
    like a conversion bug. Anything built with `np.transpose(...)` without an
    `ascontiguousarray` lands here."""
    for i, t in enumerate(inputs):
        if isinstance(t, torch.Tensor) and not t.is_contiguous():
            raise ValueError(
                f"{what}[{i}] is not contiguous (shape {tuple(t.shape)}, "
                f"stride {t.stride()}). ExecuTorch ignores strides and would "
                f"silently return garbage — call .contiguous() first.")


def _flatten_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        flat = []
        for o in out:
            flat.extend(_flatten_outputs(o))
        return flat
    raise TypeError(f"unsupported output type {type(out)} — wrap the model to return tensors")


def _map_tensors(out, fn):
    if isinstance(out, torch.Tensor):
        return fn(out)
    if isinstance(out, tuple):
        return tuple(_map_tensors(o, fn) for o in out)
    if isinstance(out, list):
        return [_map_tensors(o, fn) for o in out]
    return out


class _HalfIO(torch.nn.Module):
    """fp16 weights/compute behind an fp32 I/O boundary, so apps can swap
    fp32/fp16/int8 .pte files without touching their tensor dtypes. The two
    boundary casts stay outside the delegate (visible as portable fallback)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *xs):
        xs = [x.half() if x.is_floating_point() else x for x in xs]
        out = self.m(*xs)
        return _map_tensors(out, lambda t: t.float() if t.is_floating_point() else t)


# Norms that compute statistics from live data: fp16 variance over large spatial
# extents overflows to inf -> NaN output (seen: MODNet's IBNorm InstanceNorm side).
# Eval-mode BatchNorm with running stats is a pure affine op and stays fp16.
_STAT_NORMS = (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
               torch.nn.GroupNorm, torch.nn.LayerNorm)


def _selective_half(model, scopes):
    """Partial fp16: halve only the named submodules, fp32 everywhere else, with
    cast hooks at each island's boundary. For graphs where one region is
    numerically fragile in fp16 — DETR decoders redo boxes as
    sigmoid(inverse_sigmoid(ref) + delta), which fp16 destroys — but the conv
    backbone halves safely."""
    named = dict(model.named_modules())
    for path in scopes:
        mod = named.get(path)
        assert mod is not None, f"fp16 scope {path!r} not in model; have e.g. " \
                                f"{list(named)[:8]}"
        mod.half()
        mod.register_forward_pre_hook(lambda m, a: tuple(
            x.half() if torch.is_tensor(x) and x.is_floating_point() else x for x in a))
        mod.register_forward_hook(lambda m, a, out: _map_tensors(
            out, lambda t: t.float() if t.is_floating_point() else t))
        _fp32_norm_islands(mod)
    return model


def _fp32_norm_islands(model):
    """After .half(): return data-stat norm layers to fp32 with cast boundaries.
    These ops don't delegate to XNNPACK anyway, so this costs only the casts."""
    def pre(mod, args):
        return tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a
                     for a in args)

    def post(mod, args, out):
        return _map_tensors(out, lambda t: t.half() if t.is_floating_point() else t)

    n = 0
    for mod in model.modules():
        is_bn = isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
        if isinstance(mod, _STAT_NORMS) or (is_bn and not mod.track_running_stats):
            mod.float()
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
            n += 1
    if n:
        print(f"  fp16: kept {n} data-stat norm layer(s) in fp32")
    return model


def _quantize_pt2e(model, example_inputs, calibrate, dynamic=False, per_channel=True,
                   op_types=None):
    """PT2E int8 via XNNPACKQuantizer.
    static (default): symmetric per-channel weights + calibrated activations —
      the CNN recipe. calibrate(prepared_module) should push representative
      inputs through; falls back to example inputs (weak — pass real data).
    dynamic: linear-only, runtime activation scales — the ViT recipe (static
      int8 wrecks ViT outputs; DA2 corr 0.49 static vs 0.99998 dynamic). A
      global dynamic config would also annotate convs, which XNNPACK does not
      support (ChannelsLastTaggedReshapePass: 'required rank 4 tensor')."""
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer, get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    gm = torch.export.export(model, tuple(example_inputs)).module()
    q = XNNPACKQuantizer()
    if dynamic:
        q.set_operator_type(
            torch.ops.aten.linear.default,
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
    elif op_types:
        # Quantize only these op types, leaving the rest in fp32. Use when a
        # global annotation either can't calibrate (integer tensors reach the
        # histogram observer: "histogram_cpu not implemented for 'Long'") or
        # wrecks accuracy through an activation the int8 grid can't represent.
        cfg = get_symmetric_quantization_config(is_per_channel=per_channel)
        for t in op_types:
            q.set_operator_type(t, cfg)
    else:
        q.set_global(get_symmetric_quantization_config(is_per_channel=per_channel))
    prepared = prepare_pt2e(gm, q)
    with torch.no_grad():
        if calibrate is None or dynamic:  # dynamic needs no data-range calibration
            prepared(*example_inputs)
        else:
            calibrate(prepared)
    qm = convert_pt2e(prepared)
    if dynamic:
        from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
            DuplicateDynamicQuantChainPass)
        DuplicateDynamicQuantChainPass()(qm)
    return qm


def _time_fn(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _strip_asserts(ep):
    """Erase runtime-assert chains (_assert_async <- _is_all_true etc.) that
    torch.export records but Edge's core-ATen verifier rejects (seen: RT-DETRv2's
    anchor validation). Assert nodes are side-effecting, so erase them explicitly,
    then let DCE drop their now-dead producers."""
    gm = ep.graph_module
    targets = {
        torch.ops.aten._assert_async.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
    }
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in targets:
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return ep


def _op_name(target):
    s = str(target)
    if "EdgeOpOverload: " in s:  # str(EdgeOpOverload) appends the full schema
        s = s.split("EdgeOpOverload: ", 1)[1].split(">", 1)[0]
    return s


def _delegation_report(gm):
    """Walk the lowered edge graph: ops inside lowered_module_N subgraphs ran on
    the delegate; call_function nodes left at top level fall back to portable."""
    delegated, portable = Counter(), Counter()
    subgraphs = 0
    for node in gm.graph.nodes:
        if node.op == "get_attr" and "lowered_module" in str(node.target):
            subgraphs += 1
            lm = getattr(gm, node.target)
            for n in lm.original_module.graph_module.graph.nodes:
                if n.op == "call_function" and n.target is not operator.getitem:
                    delegated[_op_name(n.target)] += 1
        elif node.op == "call_function":
            if node.target is operator.getitem or \
                    "executorch_call_delegate" in str(node.target):
                continue
            portable[_op_name(node.target)] += 1
    d, p = sum(delegated.values()), sum(portable.values())
    return {
        "subgraphs": subgraphs,
        "delegated_ops": d,
        "portable_ops": p,
        "coverage_pct": round(100.0 * d / max(1, d + p), 1),
        "portable_fallback": dict(sorted(portable.items(), key=lambda kv: -kv[1])),
    }


def _convert_one(name, model, example_inputs, runs=10, extra_meta=None, partitioner=None,
                     skip_dim_order=False, strip_asserts=False,
                     precision="fp32", calibrate=None, gate_inputs=None,
                     exclude_configs=None, int8_dynamic=False, int8_per_channel=True,
                     int8_op_types=None, fp16_scope=None):
    """gate_inputs: optional real-image inputs for the parity check (falls back to
    example_inputs). int8 parity on randn is meaningless — calibrated models clip
    out-of-distribution activations — so pass a calib image for quantized runs.
    exclude_configs: XNNPACK partitioner config class names to drop (e.g.
    ("SliceCopyConfig",) for quantized slice, ("PreluConfig",) for the 1.4.0
    PReLU segfault). Only applies when no explicit partitioner is passed."""
    os.makedirs(PTE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = model.eval()

    if gate_inputs is None:
        gate_inputs = example_inputs
    _require_contiguous(example_inputs, "example_inputs")
    _require_contiguous(gate_inputs, "gate_inputs")
    # Reference is always the fp32 eager model, whatever we export below.
    with torch.no_grad():
        ref = _flatten_outputs(model(*gate_inputs))

    if precision == "fp32":
        export_model = model
    elif precision == "fp16":
        if fp16_scope:
            export_model = _selective_half(copy.deepcopy(model), fp16_scope).eval()
        else:
            export_model = _HalfIO(_fp32_norm_islands(copy.deepcopy(model).half())).eval()
    elif precision == "int8":
        export_model = _quantize_pt2e(copy.deepcopy(model), example_inputs, calibrate,
                                      dynamic=int8_dynamic, per_channel=int8_per_channel,
                                      op_types=int8_op_types)
    else:
        raise ValueError(f"unknown precision {precision!r}")

    ep = torch.export.export(export_model, tuple(example_inputs))
    if strip_asserts:
        ep = _strip_asserts(ep)
    if BACKEND == "coreml":
        ep = _fix_dtypes_for_coreml(ep)

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    # partitioner=False -> no delegation (portable ops), for delegate-bug A/B.
    if partitioner is None:
        if BACKEND == "coreml":
            partitioner = _coreml_partitioner()
        elif BACKEND == "vulkan":
            partitioner = _vulkan_partitioner()
        elif exclude_configs:
            from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
            cfgs = [c for c in ALL_PARTITIONER_CONFIGS if c.__name__ not in exclude_configs]
            partitioner = XnnpackPartitioner(configs=cfgs)
        else:
            partitioner = XnnpackPartitioner()
    parts = [] if partitioner is False else [partitioner]
    # skip_dim_order: keep every tensor contiguous. Needed when a graph mixes
    # channels_last into ops the portable kernels/delegate shape-prop mishandle
    # (seen: SAM2.1 mask decoder -> "different dim orders" / bad static resize).
    kwargs = {}
    if skip_dim_order:
        from executorch.exir import EdgeCompileConfig
        kwargs["compile_config"] = EdgeCompileConfig(_skip_dim_order=True)
    edge = to_edge_transform_and_lower(ep, partitioner=parts, **kwargs)
    delegation = _delegation_report(edge.exported_program().graph_module)
    et = edge.to_executorch()
    if BACKEND == "coreml":
        suffix = f"coreml_{COREML_UNIT}" + ("_fp32" if COREML_PRECISION == "fp32" else "")
    elif BACKEND == "vulkan":
        suffix = "vulkan"
    else:
        suffix = f"xnnpack_{precision}"
    pte_path = os.path.join(PTE_DIR, f"{name}_{suffix}.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    size_mb = os.path.getsize(pte_path) / 1e6

    if BACKEND == "vulkan":
        # There is no Vulkan driver on the host, so nothing here can execute the
        # file. The delegation report is still meaningful; parity and timing come
        # from the device runner, and the result says so rather than reporting a
        # correlation of 1.0 that was never measured.
        parity = [{"output": i, "shape": list(r.shape), "max_abs_diff": None,
                   "corr": None, "rel_l2": None} for i, r in enumerate(ref)]
        et_ms = torch_ms = 0.0
    else:
        from executorch.runtime import Runtime

        rt = Runtime.get()
        prog = rt.load_program(pte_path)
        method = prog.load_method("forward")
        out = _flatten_outputs(method.execute(list(gate_inputs)))

        assert len(out) == len(ref), f"output arity mismatch: et={len(out)} torch={len(ref)}"
        parity = []
        for i, (o, r) in enumerate(zip(out, ref)):
            o, r = o.float(), r.float()
            diff = (o - r).abs().max().item()
            corr = torch.corrcoef(torch.stack([o.flatten(), r.flatten()]))[0, 1].item()
            rel = ((o - r).norm() / r.norm().clamp_min(1e-12)).item()
            parity.append({"output": i, "shape": list(r.shape), "max_abs_diff": diff,
                           "corr": corr, "rel_l2": rel})

        et_ms = _time_fn(lambda: method.execute(list(example_inputs)), runs=runs)
        with torch.no_grad():
            torch_ms = _time_fn(lambda: model(*example_inputs), runs=runs)

    result = {
        "name": name,
        "precision": precision,
        "pte": os.path.basename(pte_path),
        "size_mb": round(size_mb, 1),
        "inputs": [list(t.shape) for t in example_inputs],
        "gate_input": "random" if gate_inputs is example_inputs else "real image",
        "parity": parity,
        **({"int8_mode": "dynamic (linear-only)" if int8_dynamic else "static"}
           if precision == "int8" else {}),
        **({"backend": f"coreml ({COREML_UNIT}, fp16 compute)"} if BACKEND == "coreml" else {}),
        **({"backend": "vulkan (android gpu)", "host_verified": False}
           if BACKEND == "vulkan" else {}),
        "delegation": delegation,
        "et_ms_median": round(et_ms, 1),
        "torch_eager_ms_median": round(torch_ms, 1),
        "torch": torch.__version__,
        "executorch": "1.4.0",
    }
    if extra_meta:
        result.update(extra_meta)
    if BACKEND == "vulkan":
        rsuffix = "_vulkan"
    elif BACKEND == "coreml":
        rsuffix = f"_coreml_{COREML_UNIT}"
        if COREML_PRECISION == "fp32":
            rsuffix += "_fp32"
    else:
        rsuffix = "" if precision == "fp32" else f"_{precision}"
    with open(os.path.join(RESULTS_DIR, f"{name}{rsuffix}.json"), "w") as f:
        json.dump(result, f, indent=2)

    if BACKEND == "vulkan":
        print(f"[{name}:vulkan] pte={size_mb:.1f}MB — host cannot run Vulkan; "
              f"parity is measured on device")
        total = delegation["delegated_ops"] + delegation["portable_ops"]
        print(f"  delegation: {delegation['coverage_pct']}% "
              f"({delegation['delegated_ops']}/{total} ops, {delegation['subgraphs']} subgraph(s))")
        if delegation["portable_fallback"]:
            top = ", ".join(f"{k} x{v}" for k, v in
                            list(delegation["portable_fallback"].items())[:8])
            print(f"  portable fallback: {top}")
        return result

    worst_corr = min(p["corr"] for p in parity)
    print(f"[{name}:{BACKEND}/{precision}] pte={size_mb:.1f}MB et={et_ms:.1f}ms "
          f"eager(fp32)={torch_ms:.1f}ms worst_corr={worst_corr:.6f}")
    total = delegation["delegated_ops"] + delegation["portable_ops"]
    print(f"  delegation: {delegation['coverage_pct']}% "
          f"({delegation['delegated_ops']}/{total} ops, {delegation['subgraphs']} subgraph(s))")
    if delegation["portable_fallback"]:
        top = ", ".join(f"{k} x{v}" for k, v in
                        list(delegation["portable_fallback"].items())[:8])
        print(f"  portable fallback: {top}")
    gate_key = "fp16" if BACKEND == "coreml" else precision
    if worst_corr < CORR_GATE.get(gate_key, 0.999):
        print(f"  WARN: worst corr {worst_corr:.4f} below {gate_key} gate "
              f"{CORR_GATE[gate_key]} — inspect before shipping")
    for p in parity:
        print(f"  out{p['output']} {p['shape']} max_abs_diff={p['max_abs_diff']:.3e} "
              f"corr={p['corr']:.6f} rel_l2={p['rel_l2']:.3e}")
    return result


def convert_and_gate(*args, **kwargs):
    """Convert on every backend this model should ship on.

    XNNPACK is what runs on Android and is the portable baseline; Core ML is the
    iOS accelerator path and is 3.5-13.9x faster on device. Shipping only the
    first is what left the shelf running on the CPU for a day, so a plain fp32
    call now produces both rather than making Core ML something you remember to
    ask for. Reduced precisions stay XNNPACK-only: Core ML picks its own compute
    precision from the compile spec.

    Set CONVERT_BACKEND to pin a single backend (the delegate A/B scripts do).
    """
    global BACKEND
    result = _convert_one(*args, **kwargs)
    precision = kwargs.get("precision", "fp32")
    explicit_partitioner = kwargs.get("partitioner") is not None
    if not (ALSO_COREML and precision == "fp32" and not explicit_partitioner):
        return result
    was = BACKEND
    BACKEND = "coreml"
    try:
        return _convert_one(*args, **kwargs)
    except Exception as e:
        # A model that will not lower to Core ML still ships on XNNPACK; say so
        # and keep going rather than failing the whole export.
        print(f"  Core ML build skipped: {type(e).__name__}: {str(e)[:160]}")
        return result
    finally:
        BACKEND = was
