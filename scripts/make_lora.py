#!/usr/bin/env python3
"""A LoRA adapter for a GGUF model, written directly: the shapes come from the
base model, the values are PEFT's initialisation (A random, B zero), so the
adapter changes nothing the model says while costing exactly what a trained
one costs to apply. For latency experiments that is the adapter that matters;
a trained one drops in through the same file.

    python3 make_lora.py <base.gguf> <out-lora.gguf> [--rank 16] [--alpha 32]
                         [--targets attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down]

Needs numpy and llama.cpp's gguf-py (found beside this script's checkout);
no torch, no transformers.
"""
import argparse
import os
import sys
import types

# gguf-py pulls in sentencepiece and yaml for its vocab and metadata helpers,
# which this never uses.
sys.modules.setdefault("sentencepiece", types.ModuleType("sentencepiece"))
sys.modules["sentencepiece"].SentencePieceProcessor = object
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "miniosv", "app", "llama.cpp", "gguf-py"))

import numpy as np  # noqa: E402
import gguf         # noqa: E402

DEFAULT_TARGETS = "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.base)
    field = reader.fields["general.architecture"]
    arch = bytes(field.parts[field.data[0]]).decode()
    targets = set(a.targets.split(","))
    rng = np.random.default_rng(a.seed)

    w = gguf.GGUFWriter(a.out, arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", a.alpha)

    n = 0
    for t in reader.tensors:
        parts = t.name.split(".")
        # blk.<n>.<module>.weight
        if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight" or parts[2] not in targets:
            continue
        # numpy shape is (out, in); ggml's ne is the reverse. The loader wants
        # lora_a with ne = [in, r] and lora_b with ne = [r, out].
        out_dim, in_dim = int(t.shape[1]), int(t.shape[0])
        la = (rng.standard_normal((a.rank, in_dim)) / np.sqrt(in_dim)).astype(np.float16)
        lb = np.zeros((out_dim, a.rank), dtype=np.float16)
        w.add_tensor(f"{t.name}.lora_a", la)
        w.add_tensor(f"{t.name}.lora_b", lb)
        n += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {arch}, rank {a.rank}, alpha {a.alpha}, {n} tensors adapted, "
          f"{os.path.getsize(a.out) / 1048576:.1f} MiB")


if __name__ == "__main__":
    main()
