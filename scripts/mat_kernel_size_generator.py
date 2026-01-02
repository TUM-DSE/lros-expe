#!/usr/bin/env python3

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=1)

    args = parser.parse_args()

    output = [
        {
            "src0": {"row": args.batch_size, "col": 2048},
            "src1": {"row": 2048, "col": 128256},
            "name": "result_output"
        }
    ]

    ffn_out = [
        {
            "src0": {"row": args.batch_size, "col": 8192},
            "src1": {"row": 8192, "col": 2048},
            "name": f"ffn_out-{i}"
        } for i in range(16)
    ]

    ffn_up = [
        {
            "src0": {"row": args.batch_size, "col": 2048},
            "src1": {"row": 2048, "col": 8192},
            "name": f"ffn_up-{i}"
        } for i in range(16)
    ]

    ffn_gate = [
        {
            "src0": {"row": args.batch_size, "col": 2048},
            "src1": {"row": 2048, "col": 8192},
            "name": f"ffn_gate-{i}"
        } for i in range(16)
    ]

    with open(args.outfile, 'w') as f:
        json.dump({"pairs": output + ffn_out + ffn_up + ffn_gate}, f, indent=4)
