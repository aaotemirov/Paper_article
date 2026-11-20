"""Parametric axial CT phantom of the temporal bone (80 x 80 ROI, Hounsfield units).

Structures: tympanic cavity, attic (epitympanum), external auditory canal (EAC), ossicles,
scutum (lateral attic wall) and mastoid air cells, with randomised size, position and an
in-plane rotation of up to +/-8 degrees.

Class-specific findings (s = severity in [0.1, 1]):
  * ETD  - fluid (5-25 HU) with a horizontal air-fluid level in the cavity;
  * COM  - mucosal thickening, granulation tissue (60 % of cases in the attic), mastoid sclerosis;
  * chol - soft-tissue mass (25-50 HU) arising in the attic, radius 2 + 7.5 s pixels,
           ossicular erosion and blunting of the scutum.
Acquisition: Gaussian blur and additive Gaussian noise (110-180 HU; external domain:
160-240 HU, stronger blur, +/-30 HU calibration offset).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, rotate

AIR, BONE, DENSE_BONE = -1000.0, 1100.0, 1600.0
CLASSES = ("normal", "etd", "com", "chol")


def _ellipse(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _disc(yy, xx, cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def make_phantom(cls: str, severity: float, rng: np.random.Generator, size: int = 80,
                 external: bool = False) -> np.ndarray:
    """Return a (size, size) float32 HU image for one ear."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    k = size / 80.0
    img = np.full((size, size), BONE, np.float32) + rng.normal(0, 40, (size, size)).astype(np.float32)

    # --- anatomy (randomised geometry) -------------------------------------------------
    cy, cx = 46 * k + rng.uniform(-3, 3) * k, 42 * k + rng.uniform(-3, 3) * k     # tympanic cavity
    t_ry, t_rx = rng.uniform(9, 12) * k, rng.uniform(7, 10) * k
    ay, ax = cy - t_ry - rng.uniform(3, 6) * k, cx + rng.uniform(-2, 3) * k       # attic
    a_ry, a_rx = rng.uniform(5, 7) * k, rng.uniform(6, 9) * k
    cavity = _ellipse(yy, xx, cy, cx, t_ry, t_rx)
    attic = _ellipse(yy, xx, ay, ax, a_ry, a_rx)
    eac_y = cy + rng.uniform(-2, 2) * k
    eac = (np.abs(yy - eac_y) <= rng.uniform(3.5, 5) * k) & (xx <= cx - t_rx * 0.6)
    img[cavity | attic | eac] = AIR

    # scutum: bony spur between EAC roof and attic (lateral attic wall)
    sc_len = rng.uniform(5, 7) * k
    scutum = (np.abs(yy - (eac_y - 5 * k)) <= 1.6 * k) & (xx >= cx - t_rx - 2 * k) & (xx <= cx - t_rx - 2 * k + sc_len)
    # mastoid air cells (upper right)
    my, mx = 18 * k + rng.uniform(-3, 3) * k, 62 * k + rng.uniform(-3, 3) * k
    cells = np.zeros_like(cavity)
    for _ in range(rng.integers(9, 15)):
        r = rng.uniform(1.5, 3.0) * k
        cells |= _disc(yy, xx, my + rng.uniform(-9, 9) * k, mx + rng.uniform(-9, 9) * k, r)
    img[cells] = AIR
    # ossicles (malleus head / incus body in the attic, handle in the cavity)
    oss = [(ay + rng.uniform(-1, 1) * k, ax - 1.5 * k, 2.3 * k), (ay + 1 * k, ax + 2.5 * k, 2.0 * k),
           (cy - 2 * k, cx - 1 * k, 1.4 * k)]

    # --- pathology ------------------------------------------------------------------------
    s = float(severity)
    if cls == "etd":
        level = cy + t_ry - (0.3 + 0.6 * s) * 2 * t_ry          # horizontal air-fluid level
        fluid = cavity & (yy >= level)
        img[fluid] = rng.uniform(5, 25)
    elif cls == "com":
        thick = (1 + 3 * s) * k
        inner = _ellipse(yy, xx, cy, cx, max(t_ry - thick, 1), max(t_rx - thick, 1))
        img[cavity & ~inner] = rng.uniform(20, 60)
        n_gran = 1 + int(round(2 * s))
        for _ in range(n_gran):
            in_attic = rng.random() < 0.6
            gy = ay + rng.uniform(-2, 2) * k if in_attic else cy + rng.uniform(-4, 4) * k
            gx = ax + rng.uniform(-3, 3) * k if in_attic else cx + rng.uniform(-4, 4) * k
            g = _disc(yy, xx, gy, gx, (1.5 + 2.5 * s) * k) & (cavity | attic)
            img[g] = rng.uniform(30, 60)
        if rng.random() < s:                                     # sclerosis of mastoid cells
            img[cells & (rng.random((size, size)) < 0.5 + 0.5 * s)] = BONE
    elif cls == "chol":
        r = (2 + 7.5 * s) * k
        mass = _disc(yy, xx, ay + rng.uniform(-1, 1) * k, ax + rng.uniform(-1, 1) * k, r) & (cavity | attic)
        img[mass] = rng.uniform(25, 50)
        oss = [(oy, ox, orad * max(0.25, 1 - 0.75 * s)) for oy, ox, orad in oss]   # ossicular erosion
        sc_len *= max(0.15, 1 - 0.85 * s)                                       # scutum blunting
        scutum = (np.abs(yy - (eac_y - 5 * k)) <= 1.6 * k) & (xx >= cx - t_rx - 2 * k) & (xx <= cx - t_rx - 2 * k + sc_len)

    img[scutum] = DENSE_BONE
    for oy, ox, orad in oss:
        img[_disc(yy, xx, oy, ox, orad)] = DENSE_BONE if cls != "chol" else 900.0 + 700.0 * (1 - s)

    # --- geometry + acquisition -----------------------------------------------------------
    angle = rng.uniform(-8, 8)
    img = rotate(img, angle, reshape=False, order=1, mode="nearest")
    blur = rng.uniform(1.0, 1.5) if external else rng.uniform(0.6, 1.0)
    img = gaussian_filter(img, blur)
    sigma = rng.uniform(160, 240) if external else rng.uniform(110, 180)
    img = img + rng.normal(0, sigma, img.shape)
    if external:
        img = img + rng.uniform(-30, 30)
    return img.astype(np.float32)


def window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """I_{C,W}(i,j) = min(1, max(0, (HU - C + W/2) / W))."""
    return np.clip((hu - center + width / 2.0) / width, 0.0, 1.0)


def encode_ct(hu: np.ndarray) -> np.ndarray:
    """Two-channel encoding x_t = [I_{40,400}; I_{700,4000}] (soft-tissue, bone windows)."""
    return np.stack([window(hu, 40, 400), window(hu, 700, 4000)], axis=-3).astype(np.float32)


def dicom_to_hu(pixels: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """HU = RescaleSlope * v + RescaleIntercept (for real DICOM input)."""
    return pixels.astype(np.float32) * slope + intercept
