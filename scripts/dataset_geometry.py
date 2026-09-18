"""Explicit pixel-to-DocTags geometry conversion; not dataset eligibility approval.

Only location tokens change. A caller must establish the source coordinate frame;
this module never infers it, fixes labels, removes content, or loads a model.
"""
import re

LOCATION = re.compile(r'<loc_([0-9]{1,5})>')
BOX = re.compile(r'(?:<loc_[0-9]{1,5}>){4}')


def convert_pixel_locations(raw, width, height, *, convention):
    if convention != 'top-left-pixels':
        raise ValueError('An explicit verified top-left-pixels convention is required')
    if any(type(d) is not int or not 1 <= d <= 20000 for d in (width, height)):
        raise ValueError('Image dimensions must be integers in 1..20000')
    if not isinstance(raw, str) or not raw or len(raw.encode('utf-8')) > 2_000_000:
        raise ValueError('Missing, nontext or oversized target')
    # Full groups only, no silently ignored malformed or orphan location markers.
    covered = BOX.sub('', raw)
    if '<loc_' in covered:
        raise ValueError('Malformed or orphan location tokens')
    for run in re.findall(r'(?:<loc_[0-9]{1,5}>)+', raw):
        if len(LOCATION.findall(run)) != 4:
            raise ValueError('Expected exactly four consecutive location tokens')

    def convert(match):
        left, top, right, bottom = map(int, LOCATION.findall(match[0]))
        if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
            raise ValueError('Inverted or out-of-image box; no repair permitted')
        # Match docling-core 2.97.0 DocumentToken.get_location's float ratios,
        # Python ties-to-even rounding and the 499 endpoint. Unlike that helper,
        # reject inverted/out-of-frame inputs instead of sorting/clipping them.
        values = (left, top, right, bottom)
        sizes = (width, height, width, height)
        return ''.join(f'<loc_{min(499, round(500 * (v / size)))}>'
                       for v, size in zip(values, sizes))

    return BOX.sub(convert, raw)
