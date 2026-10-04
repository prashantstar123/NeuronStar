"""Shared support-aware drawing convention for every mass--radius figure."""

from __future__ import annotations

import numpy as np


BODY_SUPPORT = 0.05
BODY_ESS = 100.0
TAIL_ESS = 1.0


def support_from_curve_counts(band: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover posterior support diagnostics from a saved 4-row band (5/50/95 percentiles, curve count)."""

    band = np.asarray(band, dtype=np.float64)
    if band.ndim != 2 or band.shape[0] != 4:
        raise ValueError(f"Expected a (4, N) band, found {band.shape}")
    rows = np.asarray(band[3], dtype=np.float64)
    maximum = float(np.nanmax(rows))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Band has no positive curve count")
    support_weight = np.clip(rows / maximum, 0.0, 1.0)
    # The saved 4-row bands were constructed from equal-weight posterior
    # resamples, so the conditional ESS equals the number of supporting rows.
    support_ess = rows.copy()
    return support_weight, support_ess, rows


def certified_body_mask(
    band: np.ndarray,
    support_weight: np.ndarray,
    support_ess: np.ndarray,
) -> np.ndarray:
    """Return the central range supported by at least 5% of the posterior."""

    band = np.asarray(band, dtype=np.float64)
    support_weight = np.asarray(support_weight, dtype=np.float64)
    support_ess = np.asarray(support_ess, dtype=np.float64)
    return (
        np.isfinite(band).all(axis=0)
        & np.isfinite(support_weight)
        & np.isfinite(support_ess)
        & (support_weight >= BODY_SUPPORT)
        & (support_ess >= BODY_ESS)
    )


def common_certified_body(
    bands: list[np.ndarray],
    support_weights: list[np.ndarray],
    support_esses: list[np.ndarray],
) -> np.ndarray:
    """Return the common certified body used for a method comparison."""

    if not bands or not (len(bands) == len(support_weights) == len(support_esses)):
        raise ValueError("Band and support lists must have the same nonzero length")
    masks = [
        certified_body_mask(band, weight, ess)
        for band, weight, ess in zip(
            bands, support_weights, support_esses, strict=True
        )
    ]
    common = np.logical_and.reduce(masks)
    if not common.any():
        raise ValueError("The compared methods have no common certified mass range")
    return common


def draw_support_aware_band(
    axis,
    mass: np.ndarray,
    body_band: np.ndarray,
    support_weight: np.ndarray,
    support_ess: np.ndarray,
    support_rows: np.ndarray,
    body_mask: np.ndarray,
    *,
    color: str,
    linestyle,
    linewidth: float,
    label: str,
    shaded: bool = False,
    fill_alpha: float = 0.20,
    tail_band: np.ndarray | None = None,
    zorder: int = 5,
):
    """Draw a certified 90% band and its naturally fading high-mass tail.

    The two pointwise quantile edges are never joined by a synthetic horizontal
    segment.  For hyperonic figures, the interior remains shaded and the fill
    fades with posterior support alongside the edge lines.
    """

    mass = np.asarray(mass, dtype=np.float64)
    body_band = np.asarray(body_band, dtype=np.float64)
    tail_band = body_band if tail_band is None else np.asarray(tail_band, dtype=np.float64)
    support_weight = np.asarray(support_weight, dtype=np.float64)
    support_ess = np.asarray(support_ess, dtype=np.float64)
    support_rows = np.asarray(support_rows, dtype=np.float64)
    body_mask = np.asarray(body_mask, dtype=bool)
    expected = (3, len(mass))
    if body_band.shape != expected or tail_band.shape != expected:
        raise ValueError(
            f"Expected body/tail bands with shape {expected}, found "
            f"{body_band.shape} and {tail_band.shape}"
        )
    for name, value in (
        ("support_weight", support_weight),
        ("support_ess", support_ess),
        ("support_rows", support_rows),
        ("body_mask", body_mask),
    ):
        if value.shape != (len(mass),):
            raise ValueError(f"{name} has shape {value.shape}, expected {(len(mass),)}")
    if not body_mask.any():
        raise ValueError("Cannot draw an empty certified band body")

    body_index = np.flatnonzero(body_mask)
    if shaded:
        axis.fill_betweenx(
            mass[body_mask],
            body_band[0, body_mask],
            body_band[2, body_mask],
            color=color,
            alpha=fill_alpha,
            linewidth=0.0,
            zorder=zorder - 1,
            label=label,
        )
        edge_label = "_nolegend_"
    else:
        edge_label = label
    for edge in (0, 2):
        axis.plot(
            body_band[edge, body_mask],
            mass[body_mask],
            color=color,
            ls=linestyle,
            lw=linewidth,
            zorder=zorder,
            label=edge_label if edge == 0 else "_nolegend_",
        )

    # Anchor the tail to the last body point, then taper the conditional
    # interval about its median in direct proportion to the surviving posterior
    # weight.  This leaves the certified central band exactly unchanged while
    # preventing a broad, artificial-looking terminal cut at vanishing support.
    body_last = int(body_index[-1])
    tail = tail_band.copy()
    tail[:, body_last] = body_band[:, body_last]
    support_ratio = np.clip(support_weight / BODY_SUPPORT, 0.0, 1.0)
    display_tail = tail.copy()
    tail_columns = np.arange(len(mass)) > body_last
    for edge in (0, 2):
        display_tail[edge, tail_columns] = tail[1, tail_columns] + support_ratio[
            tail_columns
        ] * (tail[edge, tail_columns] - tail[1, tail_columns])
    valid_tail = (
        np.isfinite(display_tail).all(axis=0)
        & np.isfinite(support_weight)
        & np.isfinite(support_ess)
        & np.isfinite(support_rows)
        & (support_rows >= 1.0)
        & (support_ess >= TAIL_ESS)
    )
    for index in range(body_last, len(mass) - 1):
        if not (valid_tail[index] and valid_tail[index + 1]):
            continue
        relative = float(
            np.sqrt(
                np.clip(
                    np.mean(support_ratio[index : index + 2]),
                    0.0,
                    1.0,
                )
            )
        )
        if shaded and relative > 0.0:
            axis.fill_betweenx(
                mass[index : index + 2],
                display_tail[0, index : index + 2],
                display_tail[2, index : index + 2],
                color=color,
                alpha=fill_alpha * relative,
                linewidth=0.0,
                zorder=zorder - 1,
            )
        line_alpha = float(np.clip(relative, 0.035, 1.0))
        for edge in (0, 2):
            axis.plot(
                display_tail[edge, index : index + 2],
                mass[index : index + 2],
                color=color,
                ls=linestyle,
                lw=linewidth,
                alpha=line_alpha,
                zorder=zorder,
            )
