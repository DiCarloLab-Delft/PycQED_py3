"""
[2020-02-03] Modified version of the original qcodes.plots.colors
Mofied by Victor Negirneac for Measurement Control

It modules makes available all the colors maps from the qcodes, context menu of
the color bar from pyqtgraph, the circular colormap created by me (Victo),
and the reversed version of all of them.

Feel free to add new colors
See "make_qcodes_anglemap" and "make_anglemap45_colorlist" below to get you
started.
"""
from pycqed.analysis.tools.plotting import make_anglemap45_colorlist

# default colors and colorscales, taken from plotly
color_cycle = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


colorscales_raw = {
    "Greys": [[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]],
    "YlGnBu": [
        [0, "rgb(8, 29, 88)"],
        [0.125, "rgb(37, 52, 148)"],
        [0.25, "rgb(34, 94, 168)"],
        [0.375, "rgb(29, 145, 192)"],
        [0.5, "rgb(65, 182, 196)"],
        [0.625, "rgb(127, 205, 187)"],
        [0.75, "rgb(199, 233, 180)"],
        [0.875, "rgb(237, 248, 217)"],
        [1, "rgb(255, 255, 217)"],
    ],
    "Greens": [
        [0, "rgb(0, 68, 27)"],
        [0.125, "rgb(0, 109, 44)"],
        [0.25, "rgb(35, 139, 69)"],
        [0.375, "rgb(65, 171, 93)"],
        [0.5, "rgb(116, 196, 118)"],
        [0.625, "rgb(161, 217, 155)"],
        [0.75, "rgb(199, 233, 192)"],
        [0.875, "rgb(229, 245, 224)"],
        [1, "rgb(247, 252, 245)"],
    ],
    "YlOrRd": [
        [0, "rgb(128, 0, 38)"],
        [0.125, "rgb(189, 0, 38)"],
        [0.25, "rgb(227, 26, 28)"],
        [0.375, "rgb(252, 78, 42)"],
        [0.5, "rgb(253, 141, 60)"],
        [0.625, "rgb(254, 178, 76)"],
        [0.75, "rgb(254, 217, 118)"],
        [0.875, "rgb(255, 237, 160)"],
        [1, "rgb(255, 255, 204)"],
    ],
    "bluered": [[0, "rgb(0,0,255)"], [1, "rgb(255,0,0)"]],
    # modified RdBu based on
    # www.sandia.gov/~kmorel/documents/ColorMaps/ColorMapsExpanded.pdf
    "RdBu": [
        [0, "rgb(5, 10, 172)"],
        [0.35, "rgb(106, 137, 247)"],
        [0.5, "rgb(190,190,190)"],
        [0.6, "rgb(220, 170, 132)"],
        [0.7, "rgb(230, 145, 90)"],
        [1, "rgb(178, 10, 28)"],
    ],
    # Scale for non-negative numeric values
    "Reds": [
        [0, "rgb(220, 220, 220)"],
        [0.2, "rgb(245, 195, 157)"],
        [0.4, "rgb(245, 160, 105)"],
        [1, "rgb(178, 10, 28)"],
    ],
    # Scale for non-positive numeric values
    "Blues": [
        [0, "rgb(5, 10, 172)"],
        [0.35, "rgb(40, 60, 190)"],
        [0.5, "rgb(70, 100, 245)"],
        [0.6, "rgb(90, 120, 245)"],
        [0.7, "rgb(106, 137, 247)"],
        [1, "rgb(220, 220, 220)"],
    ],
    "picnic": [
        [0, "rgb(0,0,255)"],
        [0.1, "rgb(51,153,255)"],
        [0.2, "rgb(102,204,255)"],
        [0.3, "rgb(153,204,255)"],
        [0.4, "rgb(204,204,255)"],
        [0.5, "rgb(255,255,255)"],
        [0.6, "rgb(255,204,255)"],
        [0.7, "rgb(255,153,255)"],
        [0.8, "rgb(255,102,204)"],
        [0.9, "rgb(255,102,102)"],
        [1, "rgb(255,0,0)"],
    ],
    "rainbow": [
        [0, "rgb(150,0,90)"],
        [0.125, "rgb(0, 0, 200)"],
        [0.25, "rgb(0, 25, 255)"],
        [0.375, "rgb(0, 152, 255)"],
        [0.5, "rgb(44, 255, 150)"],
        [0.625, "rgb(151, 255, 0)"],
        [0.75, "rgb(255, 234, 0)"],
        [0.875, "rgb(255, 111, 0)"],
        [1, "rgb(255, 0, 0)"],
    ],
    "portland": [
        [0, "rgb(12,51,131)"],
        [0.25, "rgb(10,136,186)"],
        [0.5, "rgb(242,211,56)"],
        [0.75, "rgb(242,143,56)"],
        [1, "rgb(217,30,30)"],
    ],
    "jet": [
        [0, "rgb(0,0,131)"],
        [0.125, "rgb(0,60,170)"],
        [0.375, "rgb(5,255,255)"],
        [0.625, "rgb(255,255,0)"],
        [0.875, "rgb(250,0,0)"],
        [1, "rgb(128,0,0)"],
    ],
    "hot": [
        [0, "rgb(0,0,0)"],
        [0.3, "rgb(230,0,0)"],
        [0.6, "rgb(255,210,0)"],
        [1, "rgb(255,255,255)"],
    ],
    "blackbody": [
        [0, "rgb(0,0,0)"],
        [0.2, "rgb(230,0,0)"],
        [0.4, "rgb(230,210,0)"],
        [0.7, "rgb(255,255,255)"],
        [1, "rgb(160,200,255)"],
    ],
    "earth": [
        [0, "rgb(0,0,130)"],
        [0.1, "rgb(0,180,180)"],
        [0.2, "rgb(40,210,40)"],
        [0.4, "rgb(230,230,50)"],
        [0.6, "rgb(120,70,20)"],
        [1, "rgb(255,255,255)"],
    ],
    "electric": [
        [0, "rgb(0,0,0)"],
        [0.15, "rgb(30,0,100)"],
        [0.4, "rgb(120,0,100)"],
        [0.6, "rgb(160,90,0)"],
        [0.8, "rgb(230,200,0)"],
        [1, "rgb(255,250,220)"],
    ],
    "viridis": [
        [0, "#440154"],
        [0.06274509803921569, "#48186a"],
        [0.12549019607843137, "#472d7b"],
        [0.18823529411764706, "#424086"],
        [0.25098039215686274, "#3b528b"],
        [0.3137254901960784, "#33638d"],
        [0.3764705882352941, "#2c728e"],
        [0.4392156862745098, "#26828e"],
        [0.5019607843137255, "#21918c"],
        [0.5647058823529412, "#1fa088"],
        [0.6274509803921569, "#28ae80"],
        [0.6901960784313725, "#3fbc73"],
        [0.7529411764705882, "#5ec962"],
        [0.8156862745098039, "#84d44b"],
        [0.8784313725490196, "#addc30"],
        [0.9411764705882353, "#d8e219"],
        [1, "#fde725"],
    ],
}

# Extracted https://github.com/pyqtgraph/pyqtgraph/blob/develop/pyqtgraph/graphicsItems/GradientEditorItem.py
Gradients = {
    "thermal": [
        (0.3333, (185, 0, 0, 255)),
        (0.6666, (255, 220, 0, 255)),
        (1, (255, 255, 255, 255)),
        (0, (0, 0, 0, 255)),
    ],
    "flame": [
        (0.2, (7, 0, 220, 255)),
        (0.5, (236, 0, 134, 255)),
        (0.8, (246, 246, 0, 255)),
        (1.0, (255, 255, 255, 255)),
        (0.0, (0, 0, 0, 255)),
    ],
    "yellowy": [
        (0.0, (0, 0, 0, 255)),
        (0.2328863796753704, (32, 0, 129, 255)),
        (0.8362738179251941, (255, 255, 0, 255)),
        (0.5257586450247, (115, 15, 255, 255)),
        (1.0, (255, 255, 255, 255)),
    ],
    "bipolar": [
        (0.0, (0, 255, 255, 255)),
        (1.0, (255, 255, 0, 255)),
        (0.5, (0, 0, 0, 255)),
        (0.25, (0, 0, 255, 255)),
        (0.75, (255, 0, 0, 255)),
    ],
    "spectrum": [
        (1.0, (255, 0, 255, 255)),
        (0.0, (255, 0, 0, 255)),
    ],  # this is a hsv, didn't patch qcodes to allow the specification of that part...
    "cyclic": [
        (0.0, (255, 0, 4, 255)),
        (1.0, (255, 0, 0, 255)),
    ],  # this is a hsv, didn't patch qcodes to allow the specification of that part...
    # "greyclip": [
    #     (0.0, (0, 0, 0, 255)),
    #     (0.99, (255, 255, 255, 255)),
    #     (1.0, (255, 0, 0, 255)),
    # ],
    "grey": [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))],
    # Perceptually uniform sequential colormaps from Matplotlib 2.0
    "viridis": [
        (0.0, (68, 1, 84, 255)),
        (0.25, (58, 82, 139, 255)),
        (0.5, (32, 144, 140, 255)),
        (0.75, (94, 201, 97, 255)),
        (1.0, (253, 231, 36, 255)),
    ],
    "inferno": [
        (0.0, (0, 0, 3, 255)),
        (0.25, (87, 15, 109, 255)),
        (0.5, (187, 55, 84, 255)),
        (0.75, (249, 142, 8, 255)),
        (1.0, (252, 254, 164, 255)),
    ],
    "plasma": [
        (0.0, (12, 7, 134, 255)),
        (0.25, (126, 3, 167, 255)),
        (0.5, (203, 71, 119, 255)),
        (0.75, (248, 149, 64, 255)),
        (1.0, (239, 248, 33, 255)),
    ],
    "magma": [
        (0.0, (0, 0, 3, 255)),
        (0.25, (80, 18, 123, 255)),
        (0.5, (182, 54, 121, 255)),
        (0.75, (251, 136, 97, 255)),
        (1.0, (251, 252, 191, 255)),
    ],
}


def make_qcodes_anglemap45():
    anglemap_colorlist = make_anglemap45_colorlist(N=9, use_hpl=False)
    len_colorlist = len(anglemap_colorlist)
    color_scale = [
        [i / (len_colorlist - 1), "rgb" + repr(tuple((int(x * 255) for x in col)))]
        for i, col in enumerate(anglemap_colorlist)
    ]
    return color_scale


qcodes_anglemap45 = make_qcodes_anglemap45()

colorscales_raw["anglemap45"] = qcodes_anglemap45


def make_rgba(colorscale):
    return [(v, one_rgba(c)) for v, c in colorscale]


def one_rgba(c):
    """
    convert a single color value to (r, g, b, a)
    input can be an rgb string 'rgb(r,g,b)', '#rrggbb'
    if we decide we want more we can make more, but for now this is just
    to convert plotly colorscales to pyqtgraph tuples
    """
    if c[0] == "#" and len(c) == 7:
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)
    if c[:4] == "rgb(":
        return tuple(map(int, c[4:-1].split(","))) + (255,)
    raise ValueError("one_rgba only supports rgb(r,g,b) and #rrggbb colors")


colorscales = {}
for scale_name, scale in colorscales_raw.items():
    colorscales[scale_name] = make_rgba(scale)

for scale_name, scale in Gradients.items():
    colorscales[scale_name] = scale


for name, scale in list(colorscales.items()):
    last_idx = len(scale) - 1
    reversed_scale = [
        (scale[last_idx - i][0], color[1]) for i, color in enumerate(scale)
    ]
    colorscales[name + "_reversed"] = reversed_scale

# Generate also all scales with cliping at green
for name, scale in list(colorscales.items()):
    clip_percent = 0.03
    clip_color = (0, 255, 0, 255)

    scale_low = list(scale)
    scale_low.insert(1, scale[0])
    scale_low[0] = (0.0, clip_color)

    if scale[1][0] < clip_percent:
        scale_low[1] = ((scale[1][0] + scale[0][0]) / 2, scale_low[1][1])
    else:
        scale_low[1] = (clip_percent, scale_low[1][1])
    colorscales[name + "_clip_low"] = scale_low

    scale_high = list(scale)
    scale_high.insert(-1, scale[-1])
    scale_high[-1] = (1.0, clip_color)

    if scale[-2][0] > 1 - clip_percent:
        scale_high[-2] = ((scale[-1][0] + scale[-2][0]) / 2, scale_high[-2][1])
    else:
        scale_high[-2] = (1 - clip_percent, scale_high[-2][1])
    colorscales[name + "_clip_high"] = scale_high
