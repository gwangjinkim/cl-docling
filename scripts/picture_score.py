"""Frozen supplemental picture/text scoring helpers."""
from book_score import normalized
from coverage_score import distance


def box_iou(left, right):
    if (len(left) != 4 or len(right) != 4 or
            any(type(v) is not int for v in [*left, *right])):
        raise ValueError('Expected integer LTRB boxes')
    if any(box[0] >= box[2] or box[1] >= box[3] for box in (left, right)):
        raise ValueError('Expected nonempty LTRB boxes')
    def area(box):
        return (box[2]-box[0]) * (box[3]-box[1])
    intersection = max(0, min(left[2], right[2])-max(left[0], right[0])) * max(0, min(left[3], right[3])-max(left[1], right[1]))
    union = area(left) + area(right) - intersection
    if union <= 0:
        raise ValueError('Empty picture union')
    return intersection / union


def score_picture(reference, result, threshold):
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError('Picture IoU threshold must be within [0,1]')
    expected = normalized(reference['markdown'])
    if not expected or not isinstance(reference.get('signature'), list) or not reference['signature']:
        raise ValueError('Empty or incomplete reviewed picture reference')
    strict = (result is not None and result.get('stop_reason') == 'eos' and not result.get('error') and
              not result.get('diagnostics') and isinstance(result.get('markdown'), str) and
              result.get('asset_pixels_verified') is True)
    report = dict(strict_export=bool(strict), markdown_exact=False, structure_exact=False,
                  picture_count_exact=False, picture_iou=None, picture_iou_passed=False,
                  source_asset_verified=bool(strict), classification=None, accepted=False,
                  character_edits=None, reference_characters=len(expected), word_edits=None,
                  reference_words=len(expected.split()), markdown_cer=None, markdown_wer=None)
    actual = normalized(result['markdown']) if strict else ''
    character_edits = distance(expected, actual)
    word_edits = distance(expected.split(), actual.split())
    report.update(character_edits=character_edits, word_edits=word_edits,
                  markdown_cer=character_edits/len(expected),
                  markdown_wer=word_edits/len(expected.split()))
    if not strict:
        return report
    expected_boxes, actual_boxes = reference['picture_boxes'], result['picture_boxes']
    count = len(expected_boxes) == len(actual_boxes) == 1
    iou = box_iou(expected_boxes[0], actual_boxes[0]) if count else None
    report.update(markdown_exact=expected == actual,
                  structure_exact=reference['signature'] == result['signature'], picture_count_exact=count,
                  picture_iou=iou, picture_iou_passed=iou is not None and iou >= threshold,
                  classification=result.get('classification'))
    report['accepted'] = all(report[k] for k in ('strict_export', 'markdown_exact', 'structure_exact',
                                                  'picture_count_exact', 'picture_iou_passed', 'source_asset_verified'))
    return report
