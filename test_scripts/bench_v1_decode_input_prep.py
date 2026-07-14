#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_decode_input_prep import V1SingleTokenDecodeInputPreparer


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", type=int, default=24576)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-reqs", type=int, default=128)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--block-sizes", default="16,16")
    parser.add_argument("--mrope-delta", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("v1_decode_input_prep.csv"))
    args = parser.parse_args()

    device = torch.device("cuda")
    pin_memory = is_pin_memory_available()
    block_sizes = [int(value) for value in args.block_sizes.split(",")]
    if args.position >= args.max_model_len:
        raise ValueError("position must be below max-model-len")

    block_table = MultiGroupBlockTable(
        max_num_reqs=args.max_num_reqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_tokens,
        pin_memory=pin_memory,
        device=device,
        block_sizes=block_sizes,
        kernel_block_sizes=block_sizes,
    )
    for table in block_table.block_tables:
        num_blocks = cdiv(args.max_model_len, table.block_size)
        table.add_row(list(range(num_blocks)), 0)
    block_table.commit_block_table(1)

    input_ids = CpuGpuBuffer(
        args.max_num_tokens,
        dtype=torch.int32,
        device=device,
        pin_memory=pin_memory,
    )
    positions = CpuGpuBuffer(
        args.max_num_tokens,
        dtype=torch.int64,
        device=device,
        pin_memory=pin_memory,
    )
    mrope_positions = CpuGpuBuffer(
        3,
        args.max_num_tokens + 1,
        dtype=torch.int64,
        device=device,
        pin_memory=pin_memory,
    )
    query_start_loc = CpuGpuBuffer(
        args.max_num_reqs + 1,
        dtype=torch.int32,
        device=device,
        pin_memory=pin_memory,
    )
    seq_lens = CpuGpuBuffer(
        args.max_num_reqs,
        dtype=torch.int32,
        device=device,
        pin_memory=pin_memory,
    )
    discard_request_mask = CpuGpuBuffer(
        args.max_num_reqs,
        dtype=torch.bool,
        device=device,
        pin_memory=pin_memory,
    )
    is_token_ids = CpuGpuBuffer(
        args.max_num_tokens,
        dtype=torch.bool,
        device=device,
        pin_memory=pin_memory,
    )

    token_id = 12345
    sampled_token_ids = torch.tensor([[token_id]], dtype=torch.int32, device=device)
    req_indices = np.zeros(1, dtype=np.int64)
    position_np = np.array([args.position], dtype=np.int64)

    def prepare_cpu_mirrors() -> None:
        input_ids.np[0] = token_id
        positions.np[0] = args.position
        mrope_positions.np[:, 0] = args.position + args.mrope_delta
        query_start_loc.np[0] = 0
        query_start_loc.np[1:].fill(1)
        seq_lens.np[0] = args.position + 1
        seq_lens.np[1:].fill(0)
        discard_request_mask.np.fill(False)
        is_token_ids.np[0] = True

    def baseline() -> None:
        prepare_cpu_mirrors()
        for table in block_table.block_tables:
            table.block_table.copy_to_gpu(1)
        block_table.compute_slot_mapping(req_indices, position_np)
        block_table.commit_slot_mapping(1)
        input_ids.gpu[:1].copy_(sampled_token_ids[:, 0], non_blocking=True)
        mrope_positions.gpu[:, :1].copy_(mrope_positions.cpu[:, :1], non_blocking=True)
        query_start_loc.copy_to_gpu()
        seq_lens.copy_to_gpu()
        discard_request_mask.copy_to_gpu(1)

    preparer = V1SingleTokenDecodeInputPreparer(
        block_tables=block_table.block_tables,
        max_num_reqs=args.max_num_reqs,
        device=device,
    )

    def candidate() -> None:
        prepare_cpu_mirrors()
        block_table.commit_block_table(1)
        preparer.prepare(
            sampled_token_ids=sampled_token_ids,
            input_ids=input_ids.gpu,
            positions=positions.gpu,
            mrope_positions=mrope_positions.gpu,
            query_start_loc=query_start_loc.gpu,
            seq_lens=seq_lens.gpu,
            discard_request_mask=discard_request_mask.gpu,
            is_token_ids=is_token_ids.gpu,
            position=args.position,
            mrope_delta=args.mrope_delta,
            write_is_token_ids=False,
        )

    baseline()
    torch.cuda.synchronize()
    candidate()
    torch.cuda.synchronize()

    assert input_ids.gpu[0].item() == token_id
    assert torch.equal(
        mrope_positions.gpu[:, 0].cpu(),
        torch.full((3,), args.position + args.mrope_delta, dtype=torch.int64),
    )
    assert torch.equal(
        query_start_loc.gpu.cpu(),
        torch.tensor([0] + [1] * args.max_num_reqs, dtype=torch.int32),
    )
    assert seq_lens.gpu[0].item() == args.position + 1
    assert torch.count_nonzero(seq_lens.gpu[1:]).item() == 0
    assert torch.count_nonzero(discard_request_mask.gpu).item() == 0
    for table in block_table.block_tables:
        expected_slot = (
            args.position // table.block_size * table.block_size
            + args.position % table.block_size
        )
        assert table.slot_mapping.gpu[0].item() == expected_slot

    baseline_before = time_ms(baseline, args.warmup, args.iters)
    candidate_ms = time_ms(candidate, args.warmup, args.iters)
    baseline_after = time_ms(baseline, args.warmup, args.iters)
    baseline_ms = min(baseline_before, baseline_after)
    speedup = baseline_ms / candidate_ms

    print(f"device={torch.cuda.get_device_name()} block_sizes={block_sizes}")
    print(
        f"baseline_before/candidate/baseline_after="
        f"{baseline_before:.6f}/{candidate_ms:.6f}/{baseline_after:.6f} ms "
        f"speedup={speedup:.4f}x saved={baseline_ms - candidate_ms:.6f} ms"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "variant",
                "position",
                "max_num_reqs",
                "block_sizes",
                "average_ms",
                "speedup",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "variant": "baseline",
                "position": args.position,
                "max_num_reqs": args.max_num_reqs,
                "block_sizes": args.block_sizes,
                "average_ms": f"{baseline_ms:.9f}",
                "speedup": "1.0",
            }
        )
        writer.writerow(
            {
                "variant": "gpu_fused",
                "position": args.position,
                "max_num_reqs": args.max_num_reqs,
                "block_sizes": args.block_sizes,
                "average_ms": f"{candidate_ms:.9f}",
                "speedup": f"{speedup:.9f}",
            }
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
