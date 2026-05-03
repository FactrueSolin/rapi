"""Checkpoint loading helpers for OPF model weights."""

import math
import os
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# Map the names assumed in this implementation to the checkpoint names.
PARAM_NAME_MAP = (
    {f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.swiglu.bias" for n in range(36)}
    | {
        f"block.{n}.mlp.mlp1_weight": (
            f"block.{n}.mlp.swiglu.weight.blocks",
            f"block.{n}.mlp.swiglu.weight.scales",
        )
        for n in range(36)
    }
    | {f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.out.bias" for n in range(36)}
    | {
        f"block.{n}.mlp.mlp2_weight": (
            f"block.{n}.mlp.out.weight.blocks",
            f"block.{n}.mlp.out.weight.scales",
        )
        for n in range(36)
    }
)


def _checkpoint_tensor_name(name: str) -> str:
    """Return the on-disk tensor name used for one model parameter."""
    mapped = PARAM_NAME_MAP.get(name, name)
    if isinstance(mapped, tuple):
        # Prefer the bf16 fused tensor name when writing checkpoints.
        blocks_name, _scales_name = mapped
        return blocks_name.rsplit(".blocks", 1)[0]
    return mapped


def _collect_checkpoint_tensors(
    named_tensors: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Convert named model tensors into checkpoint-serialized tensor names."""
    serialized: dict[str, torch.Tensor] = {}
    for name, tensor in named_tensors.items():
        checkpoint_name = _checkpoint_tensor_name(name)
        materialized = tensor.detach()
        if dtype is not None:
            materialized = materialized.to(dtype=dtype)
        materialized = materialized.contiguous().cpu()
        prior = serialized.get(checkpoint_name)
        if prior is not None and not torch.equal(prior, materialized):
            raise ValueError(
                "Conflicting tensors mapped to the same checkpoint key "
                f"{checkpoint_name!r}"
            )
        serialized[checkpoint_name] = materialized
    return serialized


def save_named_tensors(
    path: str | os.PathLike[str],
    named_tensors: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> None:
    """Save named model tensors to a safetensors checkpoint file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(_collect_checkpoint_tensors(named_tensors, dtype=dtype), str(output_path))


class Checkpoint:
    """Load tensors from a checkpoint directory backed by `.safetensors` files."""

    def __init__(self, path: str, device: torch.device):
        """Index all tensors available in one checkpoint directory."""
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str

        # Read from all files ending with .safetensors in the checkpoint directory
        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        # Build a mapping from tensor name to (file, key)
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    prior_file = tensor_name_to_file.get(key)
                    if prior_file is not None:
                        raise ValueError(
                            "Duplicate tensor name in checkpoint shards: "
                            f"{key!r} appears in {prior_file!r} and {safetensor_file!r}"
                        )
                    tensor_name_to_file[key] = safetensor_file

        self.tensor_name_to_file = tensor_name_to_file

    def get(self, name: str) -> torch.Tensor:
        """Return a tensor by logical model parameter name."""
        mapped = PARAM_NAME_MAP.get(name, name)
        if isinstance(mapped, tuple):
            blocks_name, scales_name = mapped
            bf16_name = blocks_name.rsplit(".blocks", 1)[0]
            if bf16_name in self.tensor_name_to_file:
                return self._get_tensor(bf16_name)
            return self._get_mxfp4_tensor(
                blocks_name, scales_name, dtype=torch.bfloat16
            )
        return self._get_tensor(mapped)

    def has(self, name: str) -> bool:
        """Report whether a logical model parameter is present in the checkpoint."""
        mapped = PARAM_NAME_MAP.get(name, name)
        if isinstance(mapped, tuple):
            blocks_name, scales_name = mapped
            bf16_name = blocks_name.rsplit(".blocks", 1)[0]
            return bf16_name in self.tensor_name_to_file or (
                blocks_name in self.tensor_name_to_file
                and scales_name in self.tensor_name_to_file
            )
        return mapped in self.tensor_name_to_file

    def _get_tensor(self, name: str) -> torch.Tensor:
        """Load one raw tensor by checkpoint name."""
        assert name in self.tensor_name_to_file, (
            f"Tensor {name} not found in checkpoint."
        )
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def _get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        """Decode one MXFP4-encoded tensor pair into a dense tensor."""
        assert blocks_name in self.tensor_name_to_file, (
            f"Blocks tensor {blocks_name} not found in checkpoint."
        )
        assert scales_name in self.tensor_name_to_file, (
            f"Scales tensor {scales_name} not found in checkpoint."
        )

        blocks = self._get_tensor(blocks_name)
        scales = self._get_tensor(scales_name).to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
