#!/usr/bin/env python3
"""The legend shared by the motivation panels: llama.cpp on Linux and LROS.

    python3 mot-legend.py <out.pdf>
"""
import sys

import mot_fonts  # noqa: F401
from common import *


def plot(out_path):
    fig = plt.figure(figsize=(3.0, 0.16))
    keys = ['llamacpp', 'lros']
    handles, labels = legend_handles(keys)
    fig.legend(handles, ['llama.cpp on Linux', 'LROS'], loc='center', ncol=len(keys),
               frameon=False, fontsize=7, handlelength=1.6,
               columnspacing=1.6, borderpad=0)
    save_path(fig, out_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip())
    plot(sys.argv[1])
