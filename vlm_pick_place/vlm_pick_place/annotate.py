"""動画用の可視化: Gemini が選んだ「掴む点」「置く点」と理由を画像に描く."""

import cv2

from .geometry import normalized_yx_to_pixel

PICK_COLOR = (0, 220, 0)      # BGR
PLACE_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)


def _put_text(img, text, org, scale=0.5):
    # 黒縁を付けて背景に関係なく読めるようにする（putText は日本語非対応なので英語のみ）
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_decision(bgr, decision, step_index, instruction_en=''):
    img = bgr.copy()
    h, w = img.shape[:2]
    _put_text(img, f'Step {step_index}', (10, 22), 0.6)
    if instruction_en:
        _put_text(img, instruction_en[:70], (10, 44))
    if decision.done:
        _put_text(img, 'DONE: ' + decision.reason[:60], (10, h - 14), 0.55)
        return img

    pu, pv = (int(round(c)) for c in normalized_yx_to_pixel(decision.pick.point_yx, w, h))
    qu, qv = (int(round(c)) for c in normalized_yx_to_pixel(decision.place.point_yx, w, h))
    cv2.arrowedLine(img, (pu, pv), (qu, qv), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.08)
    cv2.circle(img, (pu, pv), 9, PICK_COLOR, 2, cv2.LINE_AA)
    cv2.drawMarker(img, (qu, qv), PLACE_COLOR, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    _put_text(img, 'pick: ' + decision.pick.label, (pu + 12, pv - 8))
    _put_text(img, 'place: ' + decision.place.label, (qu + 12, qv + 18))
    _put_text(img, decision.reason[:80], (10, h - 14))
    return img


def draw_text_banner(bgr, text):
    img = bgr.copy()
    _put_text(img, text[:80], (10, img.shape[0] - 14), 0.55)
    return img
