from matplotlib import pyplot
from matplotlib import cm
import matplotlib
from PIL import ImageFont
import os 

def get_value_from_cm(color, cmap=cm.get_cmap('RdBu'), colrange=[0.0, 1.0]):
    norm = matplotlib.colors.Normalize(
        vmin=colrange[0], vmax=colrange[1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return mapper.to_rgba(color, bytes=True)


def draw_reference(context, center, size=100, color=[0, 255, 0, 255], opacity_level=127):
    color[3] = opacity_level
    context.rectangle(((center[0]-size/2, center[1]-size/2),
                       (center[0]+size/2, center[1]+size/2)), fill=tuple(color))


def draw_rectangle(context, value, center, size=100, opacity_level=127, colrange=[0.0, 1.0], ismin=False, ismax=False, isfirst=False, isref=False, outline="black"):

    args = {"xy": ((center[0]-size/2, center[1]-size/2),
                   (center[0] + size/2, center[1]+size/2)), "outline" : outline}

    if ismax:
        args["fill"] = (0, 255, 0, opacity_level)
    elif ismin:
        args["fill"] = (255, 255, 0, opacity_level)
    elif isfirst:
        args["fill"] = (255, 0, 255, opacity_level)
    else:
        color = list(get_value_from_cm(value, colrange=colrange))
        color[3] = opacity_level
        args["fill"] = tuple(color)

    if isref:
        args["width"] = 15

    context.rectangle(**args)


def draw_rectangle_with_value(context, value, center, size=100, colrange=[0.0, 1.0], ismin=False, ismax=False, isfirst=False, isref=False):
    assert (not (ismin and ismax)) and (not (ismax and isfirst)) and (not (ismin and isfirst))
    draw_rectangle(context, value, center, size=size,
                   colrange=colrange, ismin=ismin, ismax=ismax, isfirst=isfirst, isref=isref)
    if os.path.exists("/Library/Fonts/Arial Bold.ttf"):
        font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 30)
        context.text((center[0], center[1]), str(value), fill=(0, 0, 0, 128), font=font)
    else:
        context.text((center[0], center[1]), str(value), fill=(0, 0, 0, 128))
