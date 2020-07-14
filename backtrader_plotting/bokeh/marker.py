"""marker definition used to generate markers in bokeh using matplotlib notation"""
_mrk_fncs = {
    # "."	m00	point
    '.': ('dot', ["color"], {"size": 1}, {}),
    # ","	m01	pixel
    ',': ('dot', ["color"], {"size": 2}, {}),
    # "o"	m02	circle
    'o': ('circle', ["color", "size"], {}, {}),
    # "v"	m03	triangle_down
    'v': ('triangle', ["color", "size"],
          {"angle": 180, "angle_units": "deg"}, {}),
    # "^"	m04	triangle_up
    '^': ('triangle', ["color", "size"], {}, {}),
    # "<"	m05	triangle_left
    '<': ('triangle', ["color", "size"],
          {"angle": -90, "angle_units": "deg"}, {}),
    # ">"	m06	triangle_right
    '>': ('triangle', ["color", "size"],
          {"angle": 90, "angle_units": "deg"}, {}),
    # "1"	m07	tri_down
    '1': ('y', ["color", "size"], {}, {}),
    # "2"	m08	tri_up
    '2': ('y', ["color", "size"],
          {"angle": 180, "angle_units": "deg"}, {}),
    # "3"	m09	tri_left
    '3': ('y', ["color", "size"],
          {"angle": -90, "angle_units": "deg"}, {}),
    # "4"	m10	tri_right
    '4': ('y', ["color", "size"],
          {"angle": 90, "angle_units": "deg"}, {}),
    # "8"	m11	octagon
    '8': ('octagon', ["color", "size"], {}, {}),
    # "s"	m12	square
    's': ('square', ["color", "size"], {}, {}),
    # "p"	m13	pentagon
    'p': ('pentagon', ["color", "size"], {}, {}),
    # "P"	m23	plus(filled)
    'P': ('plus', ["color", "size"], {}, {"size": -2}),
    # "*"	m14	star
    '*': ('asterisk', ["color", "size"], {}, {}),
    # "h"	m15	hexagon1
    'h': ('hex', ["color", "size"], {}, {}),
    # "H"	m16	hexagon2
    'H': ('hex', ["color", "size"],
          {"angle": 45, "angle_units": "deg"}, {}),
    # "+"	m17	plus
    '+': ('plus', ["color", "size"], {}, {}),
    # "x"	m18	x
    'x': ('x', ["color", "size"], {}, {}),
    # "X"	m24	x(filled)
    'X': ('x', ["color", "size"], {}, {"size": -2}),
    # "D"	m19	diamond
    'D': ('diamond_cross', ["color", "size"], {}, {}),
    # "d"	m20	thin_diamond
    'd': ('diamond', ["color", "size"], {}, {}),
    # "|"	m21	vline
    '|': ('vbar', ["color"], {}, {}),
    # "_"	m22	hline
    '_': ('hbar', ["color"], {}, {}),
    # 0 (TICKLEFT)	m25	tickleft
    0: ('triangle', ["color", "size"],
        {"angle": -90, "angle_units": "deg"}, {"size": -2}),
    # 1 (TICKRIGHT)	m26	tickright
    1: ('triangle', ["color", "size"],
        {"angle": 90, "angle_units": "deg"}, {"size": -2}),
    # 2 (TICKUP)	m27	tickup
    2: ('triangle', ["color", "size"], {}, {"size": -2}),
    # 3 (TICKDOWN)	m28	tickdown
    3: ('triangle', ["color", "size"],
        {"angle": 180, "angle_units": "deg"}, {"size": -2}),
    # 4 (CARETLEFT)	m29	caretleft
    4: ('triangle', ["fill_color", "color", "size"],
        {"angle": -90, "angle_units": "deg"}, {}),
    # 5 (CARETRIGHT)	m30	caretright
    5: ('triangle', ["fill_color", "color", "size"],
        {"angle": 90, "angle_units": "deg"}, {}),
    # 6 (CARETUP)	m31	caretup
    6: ('triangle', ["fill_color", "color", "size"], {}, {}),
    # 7 (CARETDOWN)	m32	caretdown
    7: ('triangle', ["fill_color", "color", "size"],
        {"angle": 180, "angle_units": "deg"}, {}),
    # 8 (CARETLEFTBASE)	m33	caretleft(centered at base)
    8: ('triangle', ["fill_color", "color", "size"],
        {"angle": -90, "angle_units": "deg"}, {"x": 0.25}),
    # 9 (CARETRIGHTBASE)	m34	caretright(centered at base)
    9: ('triangle', ["fill_color", "color", "size"],
        {"angle": 90, "angle_units": "deg"}, {"x": -0.25}),
    # 10 (CARETUPBASE)	m35	caretup(centered at base)
    10: ('triangle', ["fill_color", "color", "size"],
         {}, {"y": -0.25}),
    # 11 (CARETDOWNBASE)	m36	caretdown(centered at base)
    11: ('triangle', ["fill_color", "color", "size"],
         {"angle": 180, "angle_units": "deg"}, {"y": 0.25}),
    # "None", " " or ""	 	nothing
    '': ('text', ["text_color", "text_font_size", "text"],
         {}, {}),
    ' ': ('text', ["text_color", "text_font_size", "text"],
          {}, {}),
    # '$...$' text
    '$': ('text', ["text_color", "text_font_size", "text"],
          {}, {}),
}


def get_marker_info(marker):
    fnc_name, attrs, vals, updates = None, list(), dict(), dict()
    if isinstance(marker, (int, float)):
        fnc_name, attrs, vals, updates = _mrk_fncs[int(marker)]
    elif isinstance(marker, str):
        fnc_name, attrs, vals, updates = _mrk_fncs[str(marker)[0]]
    else:
        raise Exception(
            f"unsupported marker type {type(marker)} for {marker}")
    return fnc_name, attrs, vals, updates
