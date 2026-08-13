"""Common torch.export -> XNNPACK .pte harness: parity gate vs torch fp32 + CPU timing.

Usage from a per-model script:
    from harness import convert_and_gate
    convert_and_gate("dinov2_vits14", model, (x,))

Outputs <repo>/pte/<name>_xnnpack_fp32.pte and <repo>/results/<name>.json.
"""
import json
import os
import statistics
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTE_DIR = os.path.join(REPO, "pte")
RESULTS_DIR = os.path.join(REPO, "results")


def _flatten_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        flat = []
        for o in out:
            flat.extend(_flatten_outputs(o))
        return flat
    raise TypeError(f"unsupported output type {type(out)} — wrap the model to return tensors")


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
    import torch as _torch
    gm = ep.graph_module
    targets = {
        _torch.ops.aten._assert_async.default,
        _torch.ops.aten._assert_async.msg,
        _torch.ops.aten._assert_scalar.default,
    }
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in targets:
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return ep


def convert_and_gate(name, model, example_inputs, runs=10, extra_meta=None, partitioner=None,
                     skip_dim_order=False, strip_asserts=False):
    os.makedirs(PTE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = model.eval()

    with torch.no_grad():
        ref = _flatten_outputs(model(*example_inputs))

    ep = torch.export.export(model, tuple(example_inputs))
    if strip_asserts:
        ep = _strip_asserts(ep)

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    # partitioner=False -> no delegation (portable ops), for delegate-bug A/B.
    if partitioner is None:
        partitioner = XnnpackPartitioner()
    parts = [] if partitioner is False else [partitioner]
    # skip_dim_order: keep every tensor contiguous. Needed when a graph mixes
    # channels_last into ops the portable kernels/delegate shape-prop mishandle
    # (seen: SAM2.1 mask decoder -> "different dim orders" / bad static resize).
    kwargs = {}
    if skip_dim_order:
        from executorch.exir import EdgeCompileConfig
        kwargs["compile_config"] = EdgeCompileConfig(_skip_dim_order=True)
    et = to_edge_transform_and_lower(ep, partitioner=parts, **kwargs).to_executorch()
    pte_path = os.path.join(PTE_DIR, f"{name}_xnnpack_fp32.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    size_mb = os.path.getsize(pte_path) / 1e6

    from executorch.runtime import Runtime

    rt = Runtime.get()
    prog = rt.load_program(pte_path)
    method = prog.load_method("forward")
    out = _flatten_outputs(method.execute(list(example_inputs)))

    assert len(out) == len(ref), f"output arity mismatch: et={len(out)} torch={len(ref)}"
    parity = []
    for i, (o, r) in enumerate(zip(out, ref)):
        diff = (o - r).abs().max().item()
        corr = torch.corrcoef(torch.stack([o.flatten().float(), r.flatten().float()]))[0, 1].item()
        parity.append({"output": i, "shape": list(r.shape), "max_abs_diff": diff, "corr": corr})

    et_ms = _time_fn(lambda: method.execute(list(example_inputs)), runs=runs)
    with torch.no_grad():
        torch_ms = _time_fn(lambda: model(*example_inputs), runs=runs)

    result = {
        "name": name,
        "pte": os.path.basename(pte_path),
        "size_mb": round(size_mb, 1),
        "inputs": [list(t.shape) for t in example_inputs],
        "parity": parity,
        "et_ms_median": round(et_ms, 1),
        "torch_eager_ms_median": round(torch_ms, 1),
        "torch": torch.__version__,
        "executorch": "1.4.0",
    }
    if extra_meta:
        result.update(extra_meta)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)

    worst_corr = min(p["corr"] for p in parity)
    print(f"[{name}] pte={size_mb:.1f}MB et={et_ms:.1f}ms eager={torch_ms:.1f}ms worst_corr={worst_corr:.6f}")
    for p in parity:
        print(f"  out{p['output']} {p['shape']} max_abs_diff={p['max_abs_diff']:.3e} corr={p['corr']:.6f}")
    return result
