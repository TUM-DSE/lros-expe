"""Font sizes for the motivation panels, which the paper sets at 0.92 of their
size: printed, they match the evaluation figures (set at 0.87). Import before
`from common import *`, which copies the sizes."""
import common

common.FONTSIZE = 8
common.FONTSIZE_AXIS_LABEL = 8
common.FONTSIZE_TICK_LABEL = 7.5
common.FONTSIZE_LEGEND = 7.5
common.FONTSIZE_TITLE = 7.5
common.FONTSIZE_ANNOTATION = 6.5
common.mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
                            'ytick.labelsize': 7.5, 'legend.fontsize': 7.5})
