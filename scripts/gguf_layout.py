#!/usr/bin/env python3
"""Print the on-disk tensor layout of GGUF files.

  python3 gguf_layout.py model.gguf [...] [--pin-quotas 80,50,10] [--all]

For each file: size, tensor count, the tensors before the first block
(embeddings, output matrix), the blocks as one span, and the tensors after
the last block. Tensor sizes are taken from the gap to the next tensor's
offset, so they are exact for any type. --all lists every tensor.

--pin-quotas marks, for each memory quota (percent of the model size),
where miniosv's pinned prefix ends with the default --pin-percent 75: the
first 0.75 * quota of the file is held resident, so the tensor at that
boundary and everything before it never leave the cache.
"""
import argparse
import os
import struct

GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16"}
KV_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def gstr(f):
    n, = read(f, "<Q")
    return f.read(n).decode(errors="replace")


def kv_value(f, t):
    if t == 8:
        return gstr(f)
    if t == 9:
        et, = read(f, "<I")
        n, = read(f, "<Q")
        return [kv_value(f, et) for _ in range(n)]
    raw = f.read(KV_SIZE[t])
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.unpack(fmt, raw)[0]


def layout(path):
    with open(path, "rb") as f:
        magic, ver, nt, nkv = read(f, "<4sIQQ")
        assert magic == b"GGUF", f"{path}: not a GGUF file"
        meta = {}
        for _ in range(nkv):
            k = gstr(f)
            t, = read(f, "<I")
            v = kv_value(f, t)
            if k in ("general.alignment", "general.architecture", "general.name"):
                meta[k] = v
        tens = []
        for _ in range(nt):
            name = gstr(f)
            nd, = read(f, "<I")
            dims = read(f, f"<{nd}Q")
            ty, = read(f, "<I")
            off, = read(f, "<Q")
            tens.append([off, name, dims, ty])
        align = meta.get("general.alignment", 32)
        data_start = (f.tell() + align - 1) // align * align
    size = os.path.getsize(path)
    tens.sort(key=lambda t: t[0])
    for i, t in enumerate(tens):
        nxt = tens[i + 1][0] if i + 1 < len(tens) else size - data_start
        t.append(nxt - t[0])            # exact byte size
    return meta, data_start, size, tens


def mib(b):
    return f"{b / 2**20:8.1f} MiB"


def describe(path, quotas, show_all):
    meta, data_start, size, tens = layout(path)
    print(f"=== {os.path.basename(path)}: {mib(size).strip()}, {len(tens)} tensors, "
          f"arch {meta.get('general.architecture')}, data at {data_start} B")
    blk = [t for t in tens if t[1].startswith("blk.")]
    first_blk = tens.index(blk[0]) if blk else len(tens)
    last_blk = tens.index(blk[-1]) if blk else -1
    types = sorted({GGUF_TYPES.get(t[3], str(t[3])) for t in tens})
    print(f"    types: {', '.join(types)}")

    def line(t):
        off, name, dims, ty, sz = t
        print(f"    {mib(off)}  {mib(sz)}  {GGUF_TYPES.get(ty, ty):5s} {name:28s} {list(dims)}")

    if show_all:
        for t in tens:
            line(t)
    else:
        for t in tens[:first_blk]:
            line(t)
        if blk:
            nblk = len({t[1].split('.')[1] for t in blk})
            span = blk[-1][0] + blk[-1][4] - blk[0][0]
            print(f"    {mib(blk[0][0])}  {mib(span)}        blk.0 .. blk.{nblk - 1} ({len(blk)} tensors)")
        for t in tens[last_blk + 1:]:
            line(t)
    if quotas:
        print("    pinned prefix ends (--pin-percent 75):")
        for q in quotas:
            end = size * q / 100 * 0.75
            at = next((t for t in tens if t[0] <= end < t[0] + t[4]), None)
            where = f"inside {at[1]}" if at else "beyond the file"
            print(f"      {q:4d}% quota: {mib(end)}  {where}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--pin-quotas", default="", help="comma-separated quota percents to mark")
    p.add_argument("--all", action="store_true", help="list every tensor")
    a = p.parse_args()
    quotas = [int(q) for q in a.pin_quotas.split(",") if q]
    for path in a.files:
        describe(path, quotas, a.all)
