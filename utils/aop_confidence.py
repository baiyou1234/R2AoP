from __future__ import annotations
import math
import numpy as np
import cv2


# -------------------------
# Geometry copied/compatible with your eval
# -------------------------
def _pubic_endpoints_from_ellipse(ellipse_pubic):
    """
    Replicate your drawline_AOD pubic axis endpoint computation.
    Returns (d11,d12,d21,d22) as floats.
    """
    (cx, cy), (w, h), ang = ellipse_pubic
    # match your eval convention
    element1 = ((cx, cy), (h, w), ang - 90)

    d11 = element1[0][0] - element1[1][0] / 2 * math.cos(element1[2] * 0.01745)
    d12 = element1[0][1] - element1[1][0] / 2 * math.sin(element1[2] * 0.01745)
    d21 = element1[0][0] + element1[1][0] / 2 * math.cos(element1[2] * 0.01745)
    d22 = element1[0][1] + element1[1][0] / 2 * math.sin(element1[2] * 0.01745)
    return d11, d12, d21, d22


def _tangent_point_on_head_ellipse(ellipse_head, d21, d22):
    """
    Compute the tangent point qie=(qie1,qie2) on head ellipse, following your eval math.
    """
    (cx, cy), (w, h), ang = ellipse_head
    element = ((cx, cy), (h, w), ang - 90)

    a = element[1][0] / 2.0
    b = element[1][1] / 2.0
    phi = 2 * math.pi * element[2] / 360.0

    dp21 = d21 - element[0][0]
    dp22 = d22 - element[0][1]
    dp2 = np.array([[dp21], [dp22]], dtype=np.float64)

    R1 = np.array([[math.cos(-phi), -math.sin(-phi)],
                   [math.sin(-phi),  math.cos(-phi)]], dtype=np.float64)
    R2 = np.array([[math.cos(phi), -math.sin(phi)],
                   [math.sin(phi),  math.cos(phi)]], dtype=np.float64)

    dpz2 = R1 @ dp2
    x0 = float(dpz2[0][0])
    y0 = float(dpz2[1][0])

    if x0**2 - a**2 == 0:
        x0 += 1.0

    disc = (b**2 * x0**2 + a**2 * y0**2 - a**2 * b**2)
    if disc < 0:
        # no valid tangent in this degenerate case
        return None

    k = (x0 * y0 - math.sqrt(disc)) / (x0**2 - a**2)
    bias = y0 - k * x0

    qx = (-2 * k * bias / b**2) / (2 * (1 / a**2 + k**2 / b**2))
    qy = qx * k + bias

    q_local = np.array([[qx], [qy]], dtype=np.float64)
    q_global = R2 @ q_local
    qie1 = float(q_global[0][0] + element[0][0])
    qie2 = float(q_global[1][0] + element[0][1])
    return qie1, qie2


def _aop_from_points(d11, d12, d21, d22, qie1, qie2):
    """
    AOP computed by cosine law, identical to your drawline_AOD end part.
    """
    ld1d3 = math.sqrt((d11 - d21) ** 2 + (d12 - d22) ** 2)
    ld3x4 = math.sqrt((d21 - qie1) ** 2 + (d22 - qie2) ** 2)
    ld1x4 = math.sqrt((d11 - qie1) ** 2 + (d12 - qie2) ** 2)
    # protect acos
    denom = max(2 * ld1d3 * ld3x4, 1e-8)
    cosv = (ld1d3 ** 2 + ld3x4 ** 2 - ld1x4 ** 2) / denom
    cosv = max(min(cosv, 1.0), -1.0)
    return math.acos(cosv) / math.pi * 180.0


def aop_from_ellipses(ellipse_head, ellipse_pubic):
    d11, d12, d21, d22 = _pubic_endpoints_from_ellipse(ellipse_pubic)
    q = _tangent_point_on_head_ellipse(ellipse_head, d21, d22)
    if q is None:
        return 0.0, None, (d11, d12, d21, d22)
    qie1, qie2 = q
    aop = _aop_from_points(d11, d12, d21, d22, qie1, qie2)
    return aop, (qie1, qie2), (d11, d12, d21, d22)


def _ellipse_local_param_t(ellipse_head, qie1, qie2):
    """
    Convert a point on ellipse into param t in local ellipse frame:
      x'=a cos t, y'=b sin t
    """
    (cx, cy), (w, h), ang = ellipse_head
    element = ((cx, cy), (h, w), ang - 90)
    a = element[1][0] / 2.0
    b = element[1][1] / 2.0
    phi = 2 * math.pi * element[2] / 360.0

    # global -> local
    dx = qie1 - element[0][0]
    dy = qie2 - element[0][1]
    R = np.array([[math.cos(-phi), -math.sin(-phi)],
                  [math.sin(-phi),  math.cos(-phi)]], dtype=np.float64)
    p = R @ np.array([[dx], [dy]], dtype=np.float64)
    x = float(p[0][0])
    y = float(p[1][0])

    # avoid div by zero
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    t = math.atan2(y / b, x / a)
    return t, a, b, phi, element[0][0], element[0][1]


def _point_on_ellipse_from_t(t, a, b, phi, cx, cy):
    x = a * math.cos(t)
    y = b * math.sin(t)
    R = np.array([[math.cos(phi), -math.sin(phi)],
                  [math.sin(phi),  math.cos(phi)]], dtype=np.float64)
    p = R @ np.array([[x], [y]], dtype=np.float64)
    return float(p[0][0] + cx), float(p[1][0] + cy)


# -------------------------
# Weighted ellipse fitting (A option): systematic resampling + cv2.fitEllipse
# -------------------------
def systematic_resample(weights: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Deterministic systematic resampling (particle filter style).
    weights: (N,) nonnegative, not necessarily normalized.
    returns indices (n_samples,)
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        # uniform
        w = np.ones_like(w) / max(len(w), 1)
    else:
        w = w / s

    N = len(w)
    n_samples = int(max(n_samples, 1))
    positions = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    cumsum = np.cumsum(w)
    idx = np.zeros(n_samples, dtype=np.int64)
    i = 0
    for j, p in enumerate(positions):
        while p > cumsum[i] and i < N - 1:
            i += 1
        idx[j] = i
    return idx


def fit_ellipse_weighted(contour_xy: np.ndarray, weights: np.ndarray | None, n_samples: int = 256):
    """
    contour_xy: (N,2) float/int, x,y order
    weights: (N,) in [0,1] or None
    Return cv2.fitEllipse ellipse tuple, or None.
    """
    if contour_xy is None or len(contour_xy) < 6:
        return None

    pts = contour_xy.astype(np.float32)

    if weights is None:
        sel = np.arange(len(pts), dtype=np.int64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(pts):
            return None
        sel = systematic_resample(w + 1e-6, n_samples=min(n_samples, len(pts) * 4))

    pts_sel = pts[sel]
    if len(pts_sel) < 6:
        return None

    # cv2.fitEllipse expects shape (N,1,2)
    pts_cv = pts_sel.reshape(-1, 1, 2)
    try:
        ellipse = cv2.fitEllipse(pts_cv)
    except Exception:
        ellipse = None
    return ellipse


def _largest_contour(binary_255: np.ndarray):
    contours, _ = cv2.findContours(cv2.medianBlur(binary_255, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours is None or len(contours) == 0:
        return None
    c = max(contours, key=lambda x: x.shape[0])
    if c.shape[0] < 6:
        return None
    return c


def aop_confidence_from_pred(
    pred_label_012: np.ndarray,
    conf_map_2hw: np.ndarray | None,
    deltas_deg=( -2.0, -1.0, 1.0, 2.0 ),
    sigma_deg: float = 2.0,
    alpha: float = 0.5,
):
    """
    pred_label_012: (H,W) uint8, values {0,1,2}
    conf_map_2hw: (2,H,W) float in [0,1] or None
      channel0 -> pubic(1), channel1 -> head(2)
    Returns dict:
      {
        'aop': aop0,
        'w_mean': boundary mean weight,
        'arc_rms': delta_rms,
        'c_aop': combined confidence
      }
    """
    H, W = pred_label_012.shape[:2]
    pubic = (pred_label_012 == 1).astype(np.uint8) * 255
    head  = (pred_label_012 == 2).astype(np.uint8) * 255

    c_p = _largest_contour(pubic)
    c_h = _largest_contour(head)
    if c_p is None or c_h is None:
        return {"aop": 0.0, "w_mean": 0.0, "arc_rms": 1e6, "c_aop": 0.0}

    # contour points: (N,1,2) -> (N,2) with x,y
    pts_p = c_p[:, 0, :].astype(np.int32)
    pts_h = c_h[:, 0, :].astype(np.int32)

    # weights sampled at boundary points
    w_p = None
    w_h = None
    if conf_map_2hw is not None:
        conf = conf_map_2hw
        # clip points inside
        xp = np.clip(pts_p[:, 0], 0, W - 1)
        yp = np.clip(pts_p[:, 1], 0, H - 1)
        xh = np.clip(pts_h[:, 0], 0, W - 1)
        yh = np.clip(pts_h[:, 1], 0, H - 1)

        w_p = conf[0, yp, xp].astype(np.float64)
        w_h = conf[1, yh, xh].astype(np.float64)

    # fit ellipses (A option)
    ellipse_p = fit_ellipse_weighted(pts_p, w_p, n_samples=256)
    ellipse_h = fit_ellipse_weighted(pts_h, w_h, n_samples=256)
    if ellipse_p is None or ellipse_h is None:
        return {"aop": 0.0, "w_mean": 0.0, "arc_rms": 1e6, "c_aop": 0.0}

    aop0, qie, pubic_pts = aop_from_ellipses(ellipse_h, ellipse_p)
    if qie is None:
        return {"aop": 0.0, "w_mean": 0.0, "arc_rms": 1e6, "c_aop": 0.0}

    qie1, qie2 = qie
    d11, d12, d21, d22 = pubic_pts

    # mean boundary confidence
    if w_p is None or w_h is None:
        w_mean = 1.0
    else:
        w_mean = 0.5 * (float(np.mean(w_p)) + float(np.mean(w_h)))

    # arc perturbation on head ellipse
    t0, a, b, phi, cx, cy = _ellipse_local_param_t(ellipse_h, qie1, qie2)
    aops = []
    for dd in deltas_deg:
        t = t0 + (dd * math.pi / 180.0)
        qx, qy = _point_on_ellipse_from_t(t, a, b, phi, cx, cy)
        aops.append(_aop_from_points(d11, d12, d21, d22, qx, qy))
    aops = np.array(aops, dtype=np.float64)
    arc_rms = float(np.sqrt(np.mean((aops - aop0) ** 2)))

    # combined confidence
    sigma = max(float(sigma_deg), 1e-6)
    s_arc = float(math.exp(-(arc_rms ** 2) / (sigma ** 2)))
    aa = float(np.clip(alpha, 0.0, 1.0))
    c_aop = float((w_mean ** aa) * (s_arc ** (1.0 - aa)))

    return {"aop": float(aop0), "w_mean": float(w_mean), "arc_rms": float(arc_rms), "c_aop": float(c_aop)}
