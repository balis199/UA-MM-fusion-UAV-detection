import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    ### 1. Imports & dataset setup
    Loads all required libraries (cv2, numpy, matplotlib, sklearn, marimo), resolves the
    `Anti-UAV-RGBT` dataset path (works from both `notebooks/` and repo root), and lists
    all test-split sequence names for the dropdown selector.
    """)
    return


@app.cell(hide_code=True)
def _():
    import functools
    import io
    import json
    import re
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression

    dataset_path = Path("../data/Anti-UAV-RGBT")
    if not dataset_path.exists():  # launched from the repo root instead of notebooks/
        dataset_path = Path("data/Anti-UAV-RGBT")
    seq_names = sorted(d.name for d in (dataset_path / "test").iterdir() if d.is_dir())
    return (
        LinearRegression,
        Path,
        cv2,
        dataset_path,
        functools,
        io,
        json,
        np,
        plt,
        re,
        seq_names,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### 2. Sequence selector widget
    Creates a `mo.ui.dropdown` listing all test-split sequence names. The selected value
    drives the reactive `seq` path in the next cell: changing it re-runs all downstream
    per-sequence cells.
    """)
    return


@app.cell
def _(seq_names):
    import marimo as mo

    # sequence selector; displayed together with the other controls in the last cell
    seq_select = mo.ui.dropdown(
        options=seq_names,
        value=seq_names[0],
        label="sequence (test split)",
    )
    return mo, seq_select


@app.cell
def _(mo):
    mo.md(r"""
    ### 3. Reactive sequence path
    Resolves the filesystem path for the currently selected test sequence. Because this
    cell depends on `seq_select.value`, marimo re-runs it (and every dependent cell)
    automatically whenever the dropdown changes.
    """)
    return


@app.cell
def _(dataset_path, seq_select):
    # reactive: re-runs whenever the sequence dropdown changes
    seq = dataset_path / "test" / str(seq_select.value)
    return (seq,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4. Cached frame access & ground-truth loading
    Scans the `processed/<split>/<seq>/` directory for preprocessed PNG frames (fast reads
    vs. slow mp4 seeks), builds a `get_frames(frame_id)` accessor backed by an in-memory
    `_cache` dict so each frame is decoded only once, and loads the `visible.json` /
    `infrared.json` ground-truth annotations for the current sequence.
    """)
    return


@app.cell(hide_code=True)
def _(cv2, dataset_path, json, mo, re, seq):
    # cached access to the preprocessed frames (processed/<split>/<seq>/<modality>/...png).
    # png reads are fast (unlike mp4 seeks), so scrolling stays responsive;
    # each frame is decoded once and then served from _cache.
    pdir = dataset_path / "processed" / seq.parent.name / seq.name
    if not pdir.is_dir():
        raise FileNotFoundError(f"no preprocessed frames for {seq.name} in {pdir.parent}")

    frame_re = re.compile(r"_frame_(\d+)\.png$")
    frame_ids = sorted(
        int(m.group(1)) for p in (pdir / "visible").glob("*.png") if (m := frame_re.search(p.name))
    )

    _cache: dict[int, tuple] = {}


    def get_frames(frame_id: int):
        if frame_id not in _cache:
            rgb = cv2.imread(str(pdir / "visible" / f"{seq.name}_visible_frame_{frame_id:06d}.png"))
            ir = cv2.imread(str(pdir / "infrared" / f"{seq.name}_infrared_frame_{frame_id:06d}.png"))
            if rgb is None or ir is None:
                raise ValueError(f"missing processed frame {frame_id} in {pdir}")
            _cache[frame_id] = (rgb, ir)
        return _cache[frame_id]


    gt_rgb = json.loads((seq / "visible.json").read_text())
    gt_ir = json.loads((seq / "infrared.json").read_text())

    mo.md(
        f"sequence **{seq.name}**: {len(frame_ids)} preprocessed frames "
        f"(ids {frame_ids[0]}..{frame_ids[-1]}, step {frame_ids[1] - frame_ids[0]}), served from a cache"
    )
    return frame_ids, get_frames, gt_ir, gt_rgb


@app.cell
def _(mo):
    mo.md(r"""
    ### 5. Affine regression: IR → RGB center mapping (train only)
    Fits two independent `LinearRegression` models (x and y) mapping IR GT box centers to
    RGB GT box centers, using **only the train split** to avoid data leakage. Also
    pre-computes subsampled scatter points (max 200/sequence) and per-sequence colors via
    `distinctipy`
    """)
    return


@app.cell(hide_code=True)
def _(LinearRegression, Path, dataset_path, json, mo, np):
    import distinctipy


    # regression from train split gt boxes only (no test labels -> no data leakage)
    def gt_centers(seq_dir: Path) -> "tuple[np.ndarray, np.ndarray]":
        out = []
        for mod in ("visible", "infrared"):
            gt = json.loads((seq_dir / f"{mod}.json").read_text())
            rects = [r for r, e in zip(gt["gt_rect"], gt["exist"]) if e == 1 and len(r) == 4]
            if rects:
                arr = np.array(rects, dtype=float)
                out.append(arr[:, :2] + arr[:, 2:] / 2)  # (x, y, w, h) -> center
            else:
                out.append(np.empty((0, 2)))
        n = min(len(out[0]), len(out[1]))
        return out[0][:n], out[1][:n]


    rgb_centers, ir_centers = [], []
    for seq_dir in sorted((dataset_path / "train").iterdir()):
        if not seq_dir.is_dir():
            continue
        c_rgb, c_ir = gt_centers(seq_dir)
        if len(c_rgb):
            rgb_centers.append(c_rgb)
            ir_centers.append(c_ir)

    centers_rgb = np.vstack(rgb_centers)
    centers_ir = np.vstack(ir_centers)

    reg_x = LinearRegression().fit(centers_ir[:, [0]], centers_rgb[:, 0])
    reg_y = LinearRegression().fit(centers_ir[:, [1]], centers_rgb[:, 1])
    scale_x, offset_x = float(reg_x.coef_[0]), float(reg_x.intercept_)
    scale_y, offset_y = float(reg_y.coef_[0]), float(reg_y.intercept_)


    # one-off costs paid here once (this cell does not re-run on widget changes):
    # distinctipy colors (seconds for 160 sequences) and a subsampled copy of the
    # centers for plotting (full data is 142k points, slow to scatter per re-render)
    def _subsample(arr: np.ndarray, max_n: int = 200) -> np.ndarray:
        if len(arr) <= max_n:
            return arr
        idx = np.linspace(0, len(arr) - 1, max_n).astype(int)
        return arr[idx]


    plot_ir = [_subsample(a) for a in ir_centers]
    plot_rgb = [_subsample(a) for a in rgb_centers]
    seq_colors = distinctipy.get_colors(len(rgb_centers), exclude_colors=[(1.0, 0.0, 0.0)])

    mo.md(
        f"regression from {len(centers_rgb)} train gt boxes: "
        f"**scale** = ({scale_x:.4f}, {scale_y:.4f}), "
        f"**offset** = ({offset_x:.2f}, {offset_y:.2f})"
    )
    return (
        ir_centers,
        offset_x,
        offset_y,
        plot_ir,
        plot_rgb,
        reg_x,
        reg_y,
        rgb_centers,
        scale_x,
        scale_y,
        seq_colors,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### 6. Per-frame regression error & quality bucketing
    Computes the pixel distance between the regression-warped IR GT box center and the RGB
    GT box center for every preprocessed frame (`frame_err`, `inf` for non-preprocessed or
    missing boxes). The `bucket_frames(p)` helper returns frames whose error falls in a
    ±5-percentile window around `p`, used by the "match quality" slider to jump to random
    good/bad alignment examples.
    """)
    return


@app.cell(hide_code=True)
def _(frame_ids, gt_ir, gt_rgb, np, offset_x, offset_y, scale_x, scale_y):
    # per-frame regression error at the preprocessed frame ids: distance between the
    # regression-warped ir gt box center and the rgb gt box center (inf elsewhere)
    _ids = set(frame_ids)
    frame_err = np.full(len(gt_rgb["gt_rect"]), np.inf)
    for _i, (_rv, _ri) in enumerate(zip(gt_rgb["gt_rect"], gt_ir["gt_rect"])):
        if (
            _i in _ids
            and gt_rgb["exist"][_i] == 1
            and len(_rv) == 4
            and gt_ir["exist"][_i] == 1
            and len(_ri) == 4
        ):
            cx_r, cy_r = _rv[0] + _rv[2] / 2, _rv[1] + _rv[3] / 2
            cx_i = scale_x * (_ri[0] + _ri[2] / 2) + offset_x
            cy_i = scale_y * (_ri[1] + _ri[3] / 2) + offset_y
            frame_err[_i] = float(np.hypot(cx_r - cx_i, cy_r - cy_i))

    valid_err = frame_err[np.isfinite(frame_err)]


    def bucket_frames(p: int) -> np.ndarray:
        # preprocessed frames whose regression error lies in the +-5 percentile window around p
        lo = np.quantile(valid_err, max(p - 5, 0) / 100)
        hi = np.quantile(valid_err, min(p + 5, 100) / 100)
        bucket = np.flatnonzero((frame_err >= lo) & (frame_err <= hi))
        if len(bucket) == 0:
            bucket = np.array([int(np.argmin(np.abs(frame_err - np.quantile(valid_err, p / 100))))])
        return bucket

    return bucket_frames, frame_err


@app.cell
def _(mo):
    mo.md(r"""
    ### 7. Radial correction models (train residuals only)
    Fits two radial multiplier models on top of the affine regression, using only train GT:
    - **radial**: `m(r) = 1 + a·r`: fits the gradient of the residual *ratio* vs. radius
      (excluding noisy near-center points where r_p < 20 px).
    - **fisheye**: `m(r) = 1 / (1 + k₁r² + k₂r⁴)`: a division model solved via least-squares
      on the rearranged equation `r_a·(k₁r_p² + k₂r_p⁴) = r_p − r_a`.
    Both operate on radii measured from the RGB image center (960, 540).
    """)
    return


@app.cell(hide_code=True)
def _(ir_centers, np, offset_x, offset_y, rgb_centers, scale_x, scale_y):
    # Radial corrections on top of the regression, fitted on the TRAIN gt residuals only
    # (no test labels). For each train gt center: pred = M_reg @ ir, residual radii about
    # the rgb image center: r_p = |pred - c|, r_a = |rgb - c|, residual = r_a - r_p.
    # Two multiplier models:
    #   "radial":  m(r) = 1 + a r     (a = gradient of the residuals vs radius)
    #   "fisheye": m(r) = 1 / (1 + k1 r^2 + k2 r^4)   (division model, fisheye-style)
    _M = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _ci = np.vstack(ir_centers)
    _cr = np.vstack(rgb_centers)
    _pred = (_M @ np.vstack([_ci.T, np.ones(len(_ci))]))[:2].T
    _img_c = np.array([1920.0, 1080.0]) / 2  # rgb canvas size (same for all sequences)
    _rp = np.linalg.norm(_pred - _img_c, axis=1)
    _ra = np.linalg.norm(_cr - _img_c, axis=1)

    # radial: m(r) = 1 + a*r with a = gradient of the residual RATIO vs radius.
    # (fitting the raw residual through the origin is unstable: near r_p ~ 0 the
    # residual is still large, which inflates the slope ~100x and multiplies the
    # wrong way - so fit ratio-1 vs r, excluding the noisy near-center points)
    _ratio = _ra / np.maximum(_rp, 1e-9)
    _ok = _rp > 20
    a = float(np.linalg.lstsq(_rp[_ok, None], _ratio[_ok] - 1, rcond=None)[0][0])
    # fisheye division: r_a = r_p / (1 + k1 r_p^2 + k2 r_p^4)
    #   -> r_a * (k1 r_p^2 + k2 r_p^4) = r_p - r_a
    _X = np.column_stack([_ra * _rp**2, _ra * _rp**4])
    _k12 = np.linalg.lstsq(_X, _rp - _ra, rcond=None)[0]
    k1, k2 = float(_k12[0]), float(_k12[1])
    return a, k1, k2


@app.cell
def _(mo):
    mo.md(r"""
    ### 8. Cached render functions (overlay / fit / vectors)
    Defines three `functools.lru_cache`-backed PNG renderers that live in this cell so their
    caches survive widget-driven re-renders:
    - `overlay_png(frame, bbox, alpha, variant)` -> blends RGB + aligned IR image with GT
      boxes overlaid (lime = RGB GT, red = aligned IR GT quad).
    - `fit_png(frame, variant)` -> per-sequence scatter of IR→RGB centers per axis with a
      regression line; per method, radial/fisheye, a line is fitted on corrected centers.
    - `vectors_png(frame, variant)` -> residual vector scatter (box-relative xy + residual
      length vs. target radius) for the current sequence.
    Also pre-computes forward-mapped predictions for all 142k train centers once per sequence.
    """)
    return


@app.cell(hide_code=True)
def _(
    a,
    cv2,
    frame_err,
    frame_ids,
    functools,
    get_frames,
    gt_ir,
    gt_rgb,
    io,
    ir_centers,
    k1,
    k2,
    np,
    offset_x,
    offset_y,
    plot_ir,
    plot_rgb,
    plt,
    reg_x,
    reg_y,
    rgb_centers,
    scale_x,
    scale_y,
    seq,
    seq_colors,
):
    # Render functions live in this cell (which runs once per sequence, not on widget
    # changes), so their lru_caches survive re-renders of the display cell below.
    # Each call renders a figure once to PNG bytes; repeat calls with the same
    # arguments are served from cache and cost ~nothing.
    _M_reg = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _rgb0, _ir0 = get_frames(frame_ids[0])
    _H, _W = _rgb0.shape[:2]
    _img_c0 = np.array([_W / 2, _H / 2])


    def _centers_of(gt):
        _out = np.full((len(gt["gt_rect"]), 2), np.nan)
        for _i, (_r, _e) in enumerate(zip(gt["gt_rect"], gt["exist"])):
            if _e == 1 and len(_r) == 4:
                _out[_i] = (_r[0] + _r[2] / 2, _r[1] + _r[3] / 2)
        return _out


    _cur_rgb = _centers_of(gt_rgb)
    _cur_ir = _centers_of(gt_ir)
    _all_ir = np.vstack(ir_centers)


    @functools.lru_cache(maxsize=8)
    def _radial_maps(variant: str):
        # dst->src remap for a forward radial multiplier m(r) about the image center,
        # inverted through a lookup table (m is monotonic for our small coefficients)
        ys, xs = np.mgrid[0:_H, 0:_W].astype(np.float64)
        rq = np.hypot(xs - _img_c0[0], ys - _img_c0[1])
        rp_lut = np.linspace(0, rq.max() * 1.5, 20000)
        if variant == "radial":
            m = 1.0 + a * rp_lut  # a = gradient of the residuals vs radius
        else:  # fisheye division model
            m = 1.0 / (1.0 + k1 * rp_lut**2 + k2 * rp_lut**4)
        rq_lut = rp_lut * m
        rp = np.interp(rq, rq_lut, rp_lut)
        scale = np.where(rq > 1e-9, rp / np.maximum(rq, 1e-9), 1.0)
        map_x = (_img_c0[0] + (xs - _img_c0[0]) * scale).astype(np.float32)
        map_y = (_img_c0[1] + (ys - _img_c0[1]) * scale).astype(np.float32)
        return map_x, map_y


    def _multiplier(r, variant):
        if variant == "radial":
            return 1.0 + a * r  # a = gradient of the residuals vs radius
        if variant == "fisheye":
            return 1.0 / (1.0 + k1 * r**2 + k2 * r**4)
        return np.ones_like(r)


    def _fwd(p, variant, img_c):
        # forward map ir coords -> corrected rgb coords (affine + radial multiplier)
        q = (_M_reg @ np.vstack([p.T, np.ones(len(p))]))[:2].T
        if variant == "reg":
            return q
        d = q - img_c
        r = np.linalg.norm(d, axis=1)
        return img_c + d * _multiplier(r, variant)[:, None]


    # variant predictions on all train ir centers, computed once per sequence
    # (142k points) and reused by every fit_png call; likewise for the
    # subsampled per-sequence scatter points and the current sequence
    _pred_all = {v: _fwd(_all_ir, v, _img_c0) for v in ("reg", "radial", "fisheye")}
    _all_rgb = np.vstack(rgb_centers)
    _plot_pred = {v: [_fwd(s, v, _img_c0) for s in plot_ir] for v in ("reg", "radial", "fisheye")}
    _cur_pred = {v: _fwd(_cur_ir, v, _img_c0) for v in ("reg", "radial", "fisheye")}


    @functools.lru_cache(maxsize=256)
    def overlay_png(frame: int, bbox: bool, alpha: float, variant: str) -> bytes:
        rgb, ir = get_frames(frame)
        img_c = np.array([rgb.shape[1] / 2, rgb.shape[0] / 2])
        ir_w = cv2.warpAffine(ir, _M_reg, (rgb.shape[1], rgb.shape[0]))
        if variant in ("radial", "fisheye"):
            mx, my = _radial_maps(variant)
            ir_aligned = cv2.remap(ir_w, mx, my, cv2.INTER_LINEAR)
        else:
            ir_aligned = ir_w

        fig, ax = plt.subplots(figsize=(10, 5.6), dpi=80)
        ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), alpha=alpha)
        ax.imshow(cv2.cvtColor(ir_aligned, cv2.COLOR_BGR2RGB), alpha=1 - alpha)

        err = np.inf
        if bbox and np.isfinite(frame_err[frame]):
            rv = gt_rgb["gt_rect"][frame]
            ax.add_patch(
                plt.Rectangle(
                    (rv[0], rv[1]), rv[2], rv[3], fill=False, edgecolor="lime", lw=2, label="rgb gt"
                )
            )
            ri = gt_ir["gt_rect"][frame]
            corners = np.array(
                [
                    [ri[0], ri[1]],
                    [ri[0] + ri[2], ri[1]],
                    [ri[0] + ri[2], ri[1] + ri[3]],
                    [ri[0], ri[1] + ri[3]],
                ],
                np.float64,
            )
            wc = _fwd(corners, variant, img_c)
            ax.add_patch(plt.Polygon(wc, fill=False, edgecolor="red", lw=2, label="ir gt (aligned)"))
            ax.legend(loc="upper right", fontsize=7)
            pc = _fwd(np.array([[ri[0] + ri[2] / 2, ri[1] + ri[3] / 2]]), variant, img_c)[0]
            a = np.array([rv[0] + rv[2] / 2, rv[1] + rv[3] / 2])
            err = float(np.hypot(pc[0] - a[0], pc[1] - a[1]))

        ax.set_title(
            f"[{variant}] frame {frame} | err {err:.1f}px | rgb a {alpha:.2f}, ir a {1 - alpha:.2f}",
            fontsize=9,
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    @functools.lru_cache(maxsize=128)
    def fit_png(frame: int, variant: str) -> bytes:
        # fit plot per alignment version, styled like the regression overview:
        # per-sequence scatter of (ir center -> rgb center) per axis with a red
        # regression line. For "reg" the raw centers and the original train
        # regression line; for "radial"/"fisheye" the centers AFTER the version's
        # correction is applied, with a NEW regression line fitted on the
        # corrected centers (does the correction leave a linear relation?)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
        for ax2, i, name, reg in zip(axes, [0, 1], ["x", "y"], [reg_x, reg_y]):
            if variant == "reg":
                s_ir_list, cur_x = plot_ir, _cur_ir
                x_all = _all_ir[:, i]
            else:
                s_ir_list, cur_x = _plot_pred[variant], _cur_pred[variant]
                x_all = _pred_all[variant][:, i]
            for s_ir, s_rgb, color in zip(s_ir_list, plot_rgb, seq_colors):
                ax2.scatter(s_ir[:, i], s_rgb[:, i], s=1, alpha=0.5, color=color)
            mask = np.isfinite(cur_x[:, i])
            ax2.scatter(
                cur_x[mask, i],
                _cur_rgb[mask, i],
                s=10,
                color="black",
                label=f"current sequence ({seq.name})",
            )
            if np.isfinite(cur_x[frame, i]):
                ax2.scatter(
                    [cur_x[frame, i]],
                    [_cur_rgb[frame, i]],
                    marker="*",
                    s=150,
                    color="magenta",
                    zorder=5,
                    label=f"frame {frame}",
                )
            x_axis = np.linspace(x_all.min(), x_all.max(), 100)
            if variant == "reg":
                ax2.plot(x_axis, reg.predict(x_axis[:, None]), color="red", label="regression line")
            else:
                A = np.column_stack([x_all, np.ones(len(x_all))])
                slope, intercept = np.linalg.lstsq(A, _all_rgb[:, i], rcond=None)[0]
                ax2.plot(
                    x_axis,
                    slope * x_axis + intercept,
                    color="red",
                    label="regression after correction",
                )
            ax2.set_xlabel(f"{'corrected ' if variant != 'reg' else ''}ir {name} center")
            ax2.set_ylabel(f"rgb {name} center")
            ax2.legend()
        fig.suptitle(
            f"fit [{variant}] of rgb on ir gt box centers per train sequence - highlighted: {seq.name}"
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    @functools.lru_cache(maxsize=128)
    def vectors_png(frame: int, variant: str) -> bytes:
        # per-frame vectors of the current sequence for one alignment variant:
        # the (variant-mapped) ir gt box center relative to the rgb gt box center,
        # and the target radius (rgb image center -> rgb gt box center)
        rgb0, _ir0b = get_frames(frame_ids[0])
        img_c = np.array([rgb0.shape[1] / 2, rgb0.shape[0] / 2])
        fids = [
            f
            for f in range(len(gt_rgb["gt_rect"]))
            if gt_rgb["exist"][f] == 1
            and len(gt_rgb["gt_rect"][f]) == 4
            and gt_ir["exist"][f] == 1
            and len(gt_ir["gt_rect"][f]) == 4
        ]
        v_ir, v_rgb = [], []
        for f in fids:
            ri = gt_ir["gt_rect"][f]
            ci = np.array([[ri[0] + ri[2] / 2, ri[1] + ri[3] / 2]])
            v_ir.append(_fwd(ci, variant, img_c)[0] - img_c)
            rv = gt_rgb["gt_rect"][f]
            v_rgb.append(np.array([rv[0] + rv[2] / 2, rv[1] + rv[3] / 2]) - img_c)
        v_ir = np.array(v_ir)
        v_rgb = np.array(v_rgb)
        res = v_ir - v_rgb  # origin = rgb box center
        l_res = np.linalg.norm(res, axis=1)
        l_radius = np.linalg.norm(v_rgb, axis=1)
        cur = fids.index(frame) if frame in fids else None

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=80)
        axes[0].scatter(res[:, 0], res[:, 1], s=4, alpha=0.5, label="ir center - rgb box center")
        if cur is not None:
            axes[0].scatter(
                [res[cur, 0]],
                [res[cur, 1]],
                marker="*",
                s=150,
                color="magenta",
                zorder=5,
                label=f"frame {frame}",
            )
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].axvline(0, color="k", lw=0.5)
        axes[0].set_xlabel("dx (px), origin = rgb box center")
        axes[0].set_ylabel("dy (px), origin = rgb box center")
        axes[0].set_aspect("equal")
        axes[0].legend(fontsize=8)
        axes[0].set_title("ir box center around rgb box center (xy)")

        axes[1].scatter(l_radius, l_res, s=4, alpha=0.5)
        if cur is not None:
            axes[1].scatter(
                [l_radius[cur]],
                [l_res[cur]],
                marker="*",
                s=150,
                color="magenta",
                zorder=5,
                label=f"frame {frame}",
            )
        axes[1].set_xlabel("|rgb image center -> rgb box center| (px)")
        axes[1].set_ylabel("|rgb box center -> ir center (reg-mapped)| (px)")
        axes[1].legend(fontsize=8)
        axes[1].set_title("residual length vs target radius")

        fig.suptitle(
            f"center vectors [{variant}] - {seq.name} - current frame {frame} "
            f"(median residual {np.median(l_res):.1f}px)"
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    return fit_png, overlay_png, vectors_png


@app.cell
def _(mo):
    mo.md(r"""
    ### 9. UI widgets (frame slider, match-quality slider, toggles)
    Defines the interactive controls in a dedicated cell (marimo cells cannot consume their
    own definitions): a frame slider stepping through preprocessed frame IDs, a "match
    quality" slider (0=good, 100=bad) that jumps to a random frame in the corresponding
    error-percentile bucket, a bbox toggle, and an opacity slider (debounced so it only
    re-renders on release). State is held in `mo.state` so all display cells stay in sync.
    """)
    return


@app.cell(hide_code=True)
def _(bucket_frames, frame_ids, mo, np):
    # widgets are defined here (marimo cells cannot consume their own definitions)
    # and displayed together with the plots in the last cell
    frame_state, set_frame = mo.state(frame_ids[0])
    _step = frame_ids[1] - frame_ids[0] if len(frame_ids) > 1 else 1


    def pick_random_duo(p):
        # match slider moved: jump to a random frame duo in that quality bucket
        rng = np.random.default_rng(int(p))
        set_frame(int(rng.choice(bucket_frames(int(p)))))


    frame_slider = mo.ui.slider(
        frame_ids[0],
        frame_ids[-1],
        step=_step,
        value=frame_ids[0],
        label=f"frame (forward/backward, step {_step} = preprocessed frames)",
        on_change=set_frame,
    )
    match_slider = mo.ui.slider(
        0,
        100,
        step=1,
        value=50,
        debounce=True,
        label="match quality (0 = very good, 100 = very bad) -> random duo in bucket",
        on_change=pick_random_duo,
    )
    bbox_box = mo.ui.checkbox(value=True, label="render gt bbox")
    opacity = mo.ui.slider(
        0.0,
        1.0,
        step=0.05,
        value=0.5,
        debounce=True,  # update only on release: opacity changes must not re-render per tick
        label="rgb opacity (ir opacity = 1 - rgb)",
    )
    return bbox_box, frame_slider, frame_state, match_slider, opacity


@app.cell
def _(mo):
    mo.md(r"""
    ### 10. Display: controls + video overlays
    Reads the current widget state (`frame_state`, `bbox_box`, `opacity`) and renders the
    control stack (`seq_select`, `frame_slider`, `match_slider`, `bbox_box`, `opacity`)
    followed by three overlay images stacked vertically — one per alignment variant
    (`reg`, `radial`, `fisheye`) — each served from the `overlay_png` LRU cache.
    """)
    return


@app.cell(hide_code=True)
def _(
    bbox_box,
    frame_slider,
    frame_state,
    match_slider,
    mo,
    opacity,
    overlay_png,
    seq_select,
):
    # cell 1: controls on top, then the three video overlays below each other
    _frame = int(frame_state())
    _bbox = bool(bbox_box.value)
    _alpha = round(float(opacity.value), 2)
    _controls = mo.vstack([seq_select, frame_slider, match_slider, bbox_box, opacity])
    mo.vstack(
        [
            _controls,
            mo.image(overlay_png(_frame, _bbox, _alpha, "reg"), width="100%"),
            mo.image(overlay_png(_frame, _bbox, _alpha, "radial"), width="100%"),
            mo.image(overlay_png(_frame, _bbox, _alpha, "fisheye"), width="100%"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 11. Display: fit plots per alignment variant
    Renders the three `fit_png` scatter plots (one per variant: `reg`, `radial`, `fisheye`)
    stacked vertically. Each shows per-sequence IR→RGB center scatter with the current
    sequence highlighted in black and the current frame as a magenta star.
    """)
    return


@app.cell(hide_code=True)
def _(fit_png, frame_state, mo):
    # cell 2: fit plots (per-sequence ir->rgb center scatter with the version's
    # predicted curve), one per alignment version, below each other
    _frame = int(frame_state())
    mo.vstack([mo.image(fit_png(_frame, v), width="100%") for v in ("reg", "radial", "fisheye")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 12. Display: residual vector plots per alignment variant
    Renders the three `vectors_png` plots stacked vertically. Each has two panels: (left)
    the mapped IR box center relative to the RGB box center in xy, and (right) residual
    length vs. distance from image center. With the current frame highlighted as a star.
    """)
    return


@app.cell(hide_code=True)
def _(frame_state, mo, vectors_png):
    # cell 3: residual plots (box-relative residual xy + residual length vs radius),
    # one per alignment version, below each other
    _frame = int(frame_state())
    mo.vstack([mo.image(vectors_png(_frame, v), width="100%") for v in ("reg", "radial", "fisheye")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 13. GT bounding-box size & position report
    Pools all RGB GT boxes from the train and test splits and renders a markdown table with
    per-split statistics: count, width/height/area (mean/median/std), center x/y (mean±std),
    and distance to image center (mean/median). Below the table, a cached 3-panel figure
    shows width/height histograms (train vs. current sequence) and a spatial scatter of box
    centers on the image canvas.
    """)
    return


@app.cell(hide_code=True)
def _(
    dataset_path,
    frame_ids,
    functools,
    get_frames,
    gt_rgb,
    io,
    json,
    mo,
    np,
    plt,
    seq,
):
    # report below the plots: size & position stats of the rgb gt boxes,
    # current test sequence vs all train sequences (rgb coordinate system)
    rgb0, _ir0 = get_frames(frame_ids[0])
    _H, _W = rgb0.shape[:2]
    _img_c = np.array([_W / 2, _H / 2])


    def _rects_of(gt):
        rs = [r for r, e in zip(gt["gt_rect"], gt["exist"]) if e == 1 and len(r) == 4]
        return np.array(rs, dtype=float).reshape(-1, 4)


    def _collect(split):
        out = []
        for seq_dir in sorted((dataset_path / split).iterdir()):
            if not seq_dir.is_dir():
                continue
            r = _rects_of(json.loads((seq_dir / "visible.json").read_text()))
            if len(r):
                out.append(r)
        return out


    _train_rects = np.vstack(_collect("train"))
    _test_rects = np.vstack(_collect("test"))  # whole test set pooled, for the table
    _cur_rects = _rects_of(gt_rgb)  # current sequence, for the figure overlay


    def _row(name, rects):
        w, h = rects[:, 2], rects[:, 3]
        cx, cy = rects[:, 0] + w / 2, rects[:, 1] + h / 2
        dist = np.hypot(cx - _img_c[0], cy - _img_c[1])

        def _mms(x):
            return f"{x.mean():.1f} / {np.median(x):.1f} / {x.std():.1f}"

        return (
            f"| {name} | {len(rects)} | {_mms(w)} | {_mms(h)} | {_mms(w * h)} "
            f"| {cx.mean():.0f}±{cx.std():.0f} | {cy.mean():.0f}±{cy.std():.0f} "
            f"| {dist.mean():.0f} / {np.median(dist):.0f} |"
        )


    _table = "\n".join(
        [
            f"rgb gt bbox report (image {_W}x{_H}, all values in rgb px; "
            "columns: width / height / area = mean / median / std, "
            "center = mean±std, last = mean / median distance to image center)",
            "",
            "| split | #boxes | width | height | area | center x | center y | dist to img center (mean/med) |",
            "|---|---|---|---|---|---|---|---|",
            _row("train (all)", _train_rects),
            _row("test (all)", _test_rects),
        ]
    )


    @functools.lru_cache(maxsize=16)
    def _report_png(seq_name: str) -> bytes:
        cur = _rects_of(gt_rgb)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=80)
        for ax2, col, name in zip(axes[:2], [2, 3], ["width", "height"]):
            ax2.hist(_train_rects[:, col], bins=60, color="lightgray", label="train (all)")
            ax2.hist(cur[:, col], bins=60, color="C0", alpha=0.7, label=seq_name)
            ax2.set_xlabel(f"rgb gt box {name} (px)")
            ax2.set_ylabel("count")
            ax2.legend(fontsize=8)
        ax3 = axes[2]
        ax3.add_patch(plt.Rectangle((0, 0), _W, _H, fill=False, edgecolor="k", lw=1))
        ax3.scatter(
            _train_rects[:, 0] + _train_rects[:, 2] / 2,
            _train_rects[:, 1] + _train_rects[:, 3] / 2,
            s=1,
            alpha=0.2,
            color="gray",
            label="train centers",
        )
        ax3.scatter(
            cur[:, 0] + cur[:, 2] / 2,
            cur[:, 1] + cur[:, 3] / 2,
            s=8,
            color="C0",
            label=f"{seq_name} centers",
        )
        ax3.plot(*_img_c, marker="+", color="red", ms=12, mew=2, label="image center")
        ax3.set_xlim(0, _W)
        ax3.set_ylim(_H, 0)
        ax3.set_aspect("equal")
        ax3.legend(fontsize=8)
        ax3.set_title("rgb gt box centers")
        fig.suptitle(f"rgb gt bbox size & position - {seq_name}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    mo.vstack([mo.md(_table), mo.image(_report_png(seq.name), width="100%")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 14. IoU computation: aligned IR boxes vs RGB GT (center-only variants)
    For every test sequence, maps the 4 IR GT box corners into RGB coordinates via each
    variant's forward transform (affine `reg`, `radial` multiplier, `fisheye` division),
    takes the axis-aligned bounding box of the mapped corners as the predicted box, and
    computes IoU against the RGB GT box. Test labels are used *for evaluation only*, the
    transforms were fitted on train labels. Results stored per-sequence and pooled.
    """)
    return


@app.cell(hide_code=True)
def _(a, dataset_path, json, k1, k2, np, offset_x, offset_y, scale_x, scale_y):
    # IoU between the converted (aligned) ir gt boxes and the rgb gt boxes, per test
    # sequence and per alignment version. The ir box corners are mapped into the rgb
    # coordinate system (affine + radial multiplier about the rgb image center) and the
    # axis-aligned bbox of the mapped corners is used as the predicted box (for the
    # diagonal affine map this is exact; for the radial maps the quad distortion across
    # one small box is negligible). Test labels are used for EVALUATION only - the
    # transforms were fitted on train labels only.
    _M = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _img_c = np.array([1920.0, 1080.0]) / 2  # rgb canvas size (same for all sequences)
    _variants = ("reg", "radial", "fisheye")


    def _fwd(p, variant):
        q = (_M @ np.vstack([p.T, np.ones(len(p))]))[:2].T
        if variant == "reg":
            return q
        d = q - _img_c
        r = np.linalg.norm(d, axis=1)
        if variant == "radial":
            m = 1.0 + a * r
        else:
            m = 1.0 / (1.0 + k1 * r**2 + k2 * r**4)
        return _img_c + d * m[:, None]


    def _seq_ious(_seq_dir):
        gt_r = json.loads((_seq_dir / "visible.json").read_text())
        gt_i = json.loads((_seq_dir / "infrared.json").read_text())
        pairs = [
            (rv, ri)
            for rv, ri, ev, ei in zip(gt_r["gt_rect"], gt_i["gt_rect"], gt_r["exist"], gt_i["exist"])
            if ev == 1 and ei == 1 and len(rv) == 4 and len(ri) == 4
        ]
        if not pairs:
            return {v: np.empty(0) for v in _variants}
        rv = np.array([p[0] for p in pairs], float)
        ri = np.array([p[1] for p in pairs], float)
        corners = np.stack(
            [
                np.stack([ri[:, 0], ri[:, 1]], 1),
                np.stack([ri[:, 0] + ri[:, 2], ri[:, 1]], 1),
                np.stack([ri[:, 0] + ri[:, 2], ri[:, 1] + ri[:, 3]], 1),
                np.stack([ri[:, 0], ri[:, 1] + ri[:, 3]], 1),
            ],
            axis=1,
        )  # (N, 4, 2)
        rx0, ry0 = rv[:, 0], rv[:, 1]
        rx1, ry1 = rx0 + rv[:, 2], ry0 + rv[:, 3]
        out = {}
        for v in _variants:
            q = _fwd(corners.reshape(-1, 2), v).reshape(-1, 4, 2)
            px0, py0 = q.min(axis=1).T
            px1, py1 = q.max(axis=1).T
            iw = np.clip(np.minimum(px1, rx1) - np.maximum(px0, rx0), 0, None)
            ih = np.clip(np.minimum(py1, ry1) - np.maximum(py0, ry0), 0, None)
            inter = iw * ih
            union = (px1 - px0) * (py1 - py0) + (rx1 - rx0) * (ry1 - ry0) - inter
            out[v] = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        return out


    iou_by_seq = {}
    for _seq_dir in sorted((dataset_path / "test").iterdir()):
        if _seq_dir.is_dir():
            iou_by_seq[_seq_dir.name] = _seq_ious(_seq_dir)
    iou_all = {v: np.concatenate([s[v] for s in iou_by_seq.values() if len(s[v])]) for v in _variants}
    return iou_all, iou_by_seq


@app.cell
def _(mo):
    mo.md(r"""
    ### 15. Display: IoU metrics table & plots (center-only variants)
    Builds a markdown table with mean/median IoU, AP@0.5, AP@0.75, and mAP@0.50:0.95 for
    `reg`, `radial`, `fisheye` over the **whole test set** (AP@t = fraction of frames with
    IoU ≥ t, since there is one box per frame). Below: a cached 2-panel figure with the IoU
    distribution histogram and the AP-vs-threshold curve per variant.
    """)
    return


@app.cell(hide_code=True)
def _(functools, io, iou_all, mo, np, plt, seq):
    # IoU metrics (AP@0.5, AP@0.75, mAP@0.5:0.95) between the converted ir boxes and the
    # rgb gt boxes. One box per modality per frame, so AP@t = fraction of frames with
    # IoU >= t. Table + plots use the whole test set.
    _thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    _variants = ("reg", "radial", "fisheye")


    def _ap(ious, t):
        return float(np.mean(ious >= t)) if len(ious) else float("nan")


    def _row(scope, variant, ious):
        return (
            f"| {scope} | {variant} | {np.mean(ious):.3f} | {np.median(ious):.3f} "
            f"| {_ap(ious, 0.5):.3f} | {_ap(ious, 0.75):.3f} "
            f"| {np.mean([_ap(ious, t) for t in _thresholds]):.3f} | {len(ious)} |"
        )


    _table = "\n".join(
        [
            "IoU of converted ir gt box vs rgb gt box over whole test set "
            "(AP@t = fraction of frames with IoU >= t; mAP = mean AP over t = 0.50:0.05:0.95)",
            "",
            "| scope | variant | mean IoU | median IoU | AP@0.5 | AP@0.75 | mAP@.5:.95 | #frames |",
            "|---|---|---|---|---|---|---|---|",
            *[_row("all test", v, iou_all[v]) for v in _variants],
        ]
    )


    @functools.lru_cache(maxsize=16)
    def _iou_png(seq_name: str) -> bytes:
        # plots use the WHOLE test set (all sequences pooled), not the current one
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
        for v, c in zip(_variants, ["C0", "C1", "C2"]):
            axes[0].hist(iou_all[v], bins=np.linspace(0, 1, 41), alpha=0.6, color=c, label=v)
        axes[0].set_xlabel("IoU (converted ir box vs rgb gt box)")
        axes[0].set_ylabel("frames")
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"IoU distribution - whole test set ({len(iou_all['reg'])} frames)")
        for v, c in zip(_variants, ["C0", "C1", "C2"]):
            axes[1].plot(_thresholds, [_ap(iou_all[v], t) for t in _thresholds], color=c, label=v)
        axes[1].set_xlabel("IoU threshold t")
        axes[1].set_ylabel("AP@t")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend(fontsize=8)
        axes[1].set_title("AP vs IoU threshold - whole test set")
        fig.suptitle("IoU metrics of aligned ir boxes vs rgb gt boxes - whole test set")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    mo.vstack([mo.md(_table), mo.image(_iou_png(seq.name), width="100%")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 16. Size regression + IoU (reg+wh / radial+wh variants)
    Fits a linear size regression on *train labels*: `rgb_w = s_w·ir_w + o_w` and
    `rgb_h = s_h·ir_h + o_h` via least-squares. Then computes IoU for two new variants that
    predict a full RGB-space box from an IR box `reg+wh` (affine center + regressed size)
    and `radial+wh` (radial-corrected center + regressed size) over all test sequences.
    """)
    return


@app.cell(hide_code=True)
def _(a, dataset_path, json, np, offset_x, offset_y, scale_x, scale_y):
    # size regression on top of the center mapping: rgb box w/h regressed on the ir box
    # w/h (train labels only). Variants "reg+wh" (affine center + size) and "radial+wh"
    # (radial center + size) predict a full rgb-space box from an ir box.
    _M = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _img_c = np.array([1920.0, 1080.0]) / 2  # rgb canvas size (same for all sequences)
    _variants = ("reg+wh", "radial+wh")

    _rw, _rh, _iw, _ih = [], [], [], []
    for _sd in sorted((dataset_path / "train").iterdir()):
        if not _sd.is_dir():
            continue
        _gr = json.loads((_sd / "visible.json").read_text())
        _gi = json.loads((_sd / "infrared.json").read_text())
        for _rv, _ri, _ev, _ei in zip(_gr["gt_rect"], _gi["gt_rect"], _gr["exist"], _gi["exist"]):
            if _ev == 1 and _ei == 1 and len(_rv) == 4 and len(_ri) == 4:
                _rw.append(_rv[2])
                _rh.append(_rv[3])
                _iw.append(_ri[2])
                _ih.append(_ri[3])
    _rw, _rh, _iw, _ih = (np.array(x, float) for x in (_rw, _rh, _iw, _ih))
    _sw, _ow = np.linalg.lstsq(np.column_stack([_iw, np.ones(len(_iw))]), _rw, rcond=None)[0]
    _sh, _oh = np.linalg.lstsq(np.column_stack([_ih, np.ones(len(_ih))]), _rh, rcond=None)[0]
    size_reg = ((float(_sw), float(_ow)), (float(_sh), float(_oh)))


    def _fwd_c(p, variant):
        q = (_M @ np.vstack([p.T, np.ones(len(p))]))[:2].T
        if variant == "reg+wh":
            return q
        d = q - _img_c
        r = np.linalg.norm(d, axis=1)
        return _img_c + d * (1.0 + a * r)[:, None]


    def _seq_ious_wh(_sd):
        _gr = json.loads((_sd / "visible.json").read_text())
        _gi = json.loads((_sd / "infrared.json").read_text())
        _pairs = [
            (_rv, _ri)
            for _rv, _ri, _ev, _ei in zip(_gr["gt_rect"], _gi["gt_rect"], _gr["exist"], _gi["exist"])
            if _ev == 1 and _ei == 1 and len(_rv) == 4 and len(_ri) == 4
        ]
        if not _pairs:
            return {v: np.empty(0) for v in _variants}
        _rv = np.array([p[0] for p in _pairs], float)
        _ri = np.array([p[1] for p in _pairs], float)
        _c = _ri[:, :2] + _ri[:, 2:] / 2
        _w = _sw * _ri[:, 2] + _ow
        _h = _sh * _ri[:, 3] + _oh
        _rx0, _ry0 = _rv[:, 0], _rv[:, 1]
        _rx1, _ry1 = _rx0 + _rv[:, 2], _ry0 + _rv[:, 3]
        _out = {}
        for v in _variants:
            _cc = _fwd_c(_c, v)
            _px0, _py0 = _cc[:, 0] - _w / 2, _cc[:, 1] - _h / 2
            _px1, _py1 = _cc[:, 0] + _w / 2, _cc[:, 1] + _h / 2
            _iw2 = np.clip(np.minimum(_px1, _rx1) - np.maximum(_px0, _rx0), 0, None)
            _ih2 = np.clip(np.minimum(_py1, _ry1) - np.maximum(_py0, _ry0), 0, None)
            _inter = _iw2 * _ih2
            _union = _w * _h + (_rx1 - _rx0) * (_ry1 - _ry0) - _inter
            _out[v] = np.where(_union > 0, _inter / np.maximum(_union, 1e-12), 0.0)
        return _out


    iou_wh_by_seq = {}
    for _sd in sorted((dataset_path / "test").iterdir()):
        if _sd.is_dir():
            iou_wh_by_seq[_sd.name] = _seq_ious_wh(_sd)
    iou_wh_all = {
        v: np.concatenate([s[v] for s in iou_wh_by_seq.values() if len(s[v])]) for v in _variants
    }
    return iou_wh_all, iou_wh_by_seq, size_reg


@app.cell
def _(mo):
    mo.md(r"""
    ### 17. Display: combined IoU metrics (all 5 variants)
    Extends the IoU report to include the size-regressed variants (`reg+wh`, `radial+wh`)
    alongside the center-only ones. Table shows mean/median IoU, AP@0.5, AP@0.75, mAP for
    all five variants over the whole test set. Plots: IoU distribution for the top variants
    + AP-vs-threshold curves for all five.
    """)
    return


@app.cell(hide_code=True)
def _(
    functools,
    io,
    iou_all,
    iou_by_seq,
    iou_wh_all,
    iou_wh_by_seq,
    mo,
    np,
    plt,
    seq,
    size_reg,
):
    # IoU metrics for the size-regressed variants ("reg+wh", "radial+wh") next to the
    # center-only variants, current sequence + all test. Same AP definition as above.
    _thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    _old = ("reg", "radial", "fisheye")
    _new = ("reg+wh", "radial+wh")
    _colors = {"reg": "C0", "radial": "C1", "fisheye": "C2", "reg+wh": "C3", "radial+wh": "C4"}


    def _ap(ious, t):
        return float(np.mean(ious >= t)) if len(ious) else float("nan")


    def _get(scope, v):
        if v in _new:
            return iou_wh_by_seq[scope][v] if scope != "all test" else iou_wh_all[v]
        return iou_by_seq[scope][v] if scope != "all test" else iou_all[v]


    def _row(scope, v):
        x = _get(scope, v)
        return (
            f"| {scope} | {v} | {np.mean(x):.3f} | {np.median(x):.3f} "
            f"| {_ap(x, 0.5):.3f} | {_ap(x, 0.75):.3f} "
            f"| {np.mean([_ap(x, t) for t in _thresholds]):.3f} | {len(x)} |"
        )


    (_sw, _ow), (_sh, _oh) = size_reg
    _table = "\n".join(
        [
            f"IoU with size regression (rgb w = {_sw:.3f}*ir_w + {_ow:.1f}, "
            f"rgb h = {_sh:.3f}*ir_h + {_oh:.1f}; train labels only) over whole test set. "
            "AP@t = fraction of frames with IoU >= t; mAP = mean AP over t = 0.50:0.05:0.95",
            "",
            "| scope | variant | mean IoU | median IoU | AP@0.5 | AP@0.75 | mAP@.5:.95 | #frames |",
            "|---|---|---|---|---|---|---|---|",
            *[_row("all test", v) for v in (*_old, *_new)],
        ]
    )


    @functools.lru_cache(maxsize=16)
    def _iou_wh_png(seq_name: str) -> bytes:
        # plots use the WHOLE test set (all sequences pooled), not the current one
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
        for v in ("radial", "reg+wh", "radial+wh"):
            axes[0].hist(
                _get("all test", v),
                bins=np.linspace(0, 1, 41),
                alpha=0.6,
                color=_colors[v],
                label=v,
            )
        axes[0].set_xlabel("IoU (converted ir box vs rgb gt box)")
        axes[0].set_ylabel("frames")
        axes[0].legend(fontsize=8)
        axes[0].set_title(f"IoU distribution - whole test set ({len(iou_all['reg'])} frames)")
        for v in (*_old, *_new):
            axes[1].plot(
                _thresholds,
                [_ap(_get("all test", v), t) for t in _thresholds],
                color=_colors[v],
                label=v,
            )
        axes[1].set_xlabel("IoU threshold t")
        axes[1].set_ylabel("AP@t")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend(fontsize=8)
        axes[1].set_title("AP vs IoU threshold - whole test set")
        fig.suptitle("IoU metrics with size regression - whole test set")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    mo.vstack([mo.md(_table), mo.image(_iou_wh_png(seq.name), width="100%")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 18. Per-frame diagnostics for near-zero-IoU frames (whole test set)
    Iterates over all test sequences and collects per-frame diagnostics to explain why some
    frames have near-zero IoU: IoU per variant, RGB box center distance from image center,
    RGB box width/height, and the center distance between the radial-mapped IR box and the
    RGB box. Results are concatenated into a single `diag` dict keyed by `seq`, `iou`,
    `r`, `wh`, `cdist`.
    """)
    return


@app.cell(hide_code=True)
def _(a, dataset_path, json, k1, k2, np, offset_x, offset_y, scale_x, scale_y):
    # per-frame diagnostics over the WHOLE test set, to explain the near-zero-IoU
    # frames: IoU per variant (AABB of the mapped ir box corners), rgb box center
    # distance from the image center, rgb box w/h, and the center distance between
    # the radial-mapped ir box and the rgb box
    _M = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _img_c = np.array([1920.0, 1080.0]) / 2
    _variants = ("reg", "radial", "fisheye")


    def _fwd(p, variant):
        q = (_M @ np.vstack([p.T, np.ones(len(p))]))[:2].T
        if variant == "reg":
            return q
        d = q - _img_c
        r = np.linalg.norm(d, axis=1)
        if variant == "radial":
            m = 1.0 + a * r
        else:
            m = 1.0 / (1.0 + k1 * r**2 + k2 * r**4)
        return _img_c + d * m[:, None]


    _seq, _iou, _r, _wh, _cdist = [], [], [], [], []
    for _sd in sorted((dataset_path / "test").iterdir()):
        if not _sd.is_dir():
            continue
        _gr = json.loads((_sd / "visible.json").read_text())
        _gi = json.loads((_sd / "infrared.json").read_text())
        _pairs = [
            (_rv, _ri)
            for _rv, _ri, _ev, _ei in zip(_gr["gt_rect"], _gi["gt_rect"], _gr["exist"], _gi["exist"])
            if _ev == 1 and _ei == 1 and len(_rv) == 4 and len(_ri) == 4
        ]
        if not _pairs:
            continue
        _rv = np.array([p[0] for p in _pairs], float)
        _ri = np.array([p[1] for p in _pairs], float)
        _c_ir = _ri[:, :2] + _ri[:, 2:] / 2
        _c_rgb = _rv[:, :2] + _rv[:, 2:] / 2
        _corners = np.stack(
            [
                np.stack([_ri[:, 0], _ri[:, 1]], 1),
                np.stack([_ri[:, 0] + _ri[:, 2], _ri[:, 1]], 1),
                np.stack([_ri[:, 0] + _ri[:, 2], _ri[:, 1] + _ri[:, 3]], 1),
                np.stack([_ri[:, 0], _ri[:, 1] + _ri[:, 3]], 1),
            ],
            axis=1,
        )
        _rx0, _ry0 = _rv[:, 0], _rv[:, 1]
        _rx1, _ry1 = _rx0 + _rv[:, 2], _ry0 + _rv[:, 3]
        _ious = {}
        for v in _variants:
            _q = _fwd(_corners.reshape(-1, 2), v).reshape(-1, 4, 2)
            _px0, _py0 = _q.min(axis=1).T
            _px1, _py1 = _q.max(axis=1).T
            _iw = np.clip(np.minimum(_px1, _rx1) - np.maximum(_px0, _rx0), 0, None)
            _ih = np.clip(np.minimum(_py1, _ry1) - np.maximum(_py0, _ry0), 0, None)
            _inter = _iw * _ih
            _union = (_px1 - _px0) * (_py1 - _py0) + (_rx1 - _rx0) * (_ry1 - _ry0) - _inter
            _ious[v] = np.where(_union > 0, _inter / np.maximum(_union, 1e-12), 0.0)
        _seq.append(np.full(len(_pairs), _sd.name))
        _iou.append(np.stack([_ious[v] for v in _variants], axis=1))
        _r.append(np.linalg.norm(_c_rgb - _img_c, axis=1))
        _wh.append(np.stack([_rv[:, 2], _rv[:, 3]], axis=1))
        _cdist.append(np.linalg.norm(_fwd(_c_ir, "radial") - _c_rgb, axis=1))
    diag = {
        "seq": np.concatenate(_seq),
        "iou": np.concatenate(_iou),  # (N, 3): reg, radial, fisheye
        "r": np.concatenate(_r),  # rgb box center distance from image center (px)
        "wh": np.concatenate(_wh),  # rgb box (w, h)
        "cdist": np.concatenate(_cdist),  # radial-mapped ir center vs rgb center (px)
    }
    return (diag,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 19. Display: near-zero-IoU diagnostics
    Analyses frames with radial IoU < 0.05 across the whole test set. Renders a summary
    (counts per variant, top contributing sequences, median center distance / box size /
    center offset for zero-IoU vs. all frames), a top-10 sequence table, and a cached 2×2
    figure: zero-IoU frames per sequence (barh), center-distance histogram, IoU vs. center
    distance scatter, and center-offset histogram.
    """)
    return


@app.cell(hide_code=True)
def _(diag, functools, io, mo, np, plt):
    # diagnostics for the near-zero-IoU frames (IoU < 0.05) on the whole test set,
    # radial variant as reference: which sequences they come from, where the target
    # sits relative to the image center, how big the boxes are, and how far apart
    # the mapped ir and rgb box centers are
    _variants = ("reg", "radial", "fisheye")
    iou_r = diag["iou"][:, 1]
    zero = iou_r < 0.05
    seqs = diag["seq"]
    uniq, counts = np.unique(seqs, return_counts=True)
    zc = np.array([int(zero[seqs == s].sum()) for s in uniq])
    order = np.argsort(-zc)

    _n = len(iou_r)
    _summary = "\n".join(
        [
            f"near-zero-IoU diagnostics (IoU < 0.05) over whole test set, {_n} frames with boxes in both modalities",
            f"- frames with IoU < 0.05: reg {int((diag['iou'][:, 0] < 0.05).sum())}, "
            f"radial {int(zero.sum())}, fisheye {int((diag['iou'][:, 2] < 0.05).sum())}",
            f"- radial zero-IoU frames from top-1 sequence: {zc[order[0]] / max(zero.sum(), 1):.1%} "
            f"({uniq[order[0]]}), from top-5: {zc[order[:5]].sum() / max(zero.sum(), 1):.1%}",
            f"- median center distance (rgb box center vs image center): zero-IoU {np.median(diag['r'][zero]):.0f}px "
            f"vs all {np.median(diag['r']):.0f}px",
            f"- median mapped-center offset (radial ir vs rgb): zero-IoU {np.median(diag['cdist'][zero]):.0f}px "
            f"vs all {np.median(diag['cdist']):.0f}px",
            f"- median rgb box size (w x h): zero-IoU {np.median(diag['wh'][zero, 0]):.0f}x{np.median(diag['wh'][zero, 1]):.0f}px "
            f"vs all {np.median(diag['wh'][:, 0]):.0f}x{np.median(diag['wh'][:, 1]):.0f}px",
        ]
    )

    _rows = []
    for i in order[:10]:
        _m = seqs == uniq[i]
        _zm = _m & zero
        _rows.append(
            f"| {uniq[i]} | {int(_m.sum())} | {int(_zm.sum())} | {_zm.sum() / max(_m.sum(), 1):.0%} "
            f"| {np.median(diag['r'][_zm]):.0f} | {np.median(diag['wh'][_zm, 0]):.0f}x{np.median(diag['wh'][_zm, 1]):.0f} "
            f"| {np.median(diag['cdist'][_zm]):.0f} |"
        )
    _table = "\n".join(
        [
            "top 10 sequences by #near-zero-IoU frames (radial, whole test set)",
            "",
            "| sequence | #frames | #IoU<0.05 | share of seq | med r center (zero) | med box wxh (zero) | med center offset (zero) |",
            "|---|---|---|---|---|---|---|",
            *_rows,
        ]
    )


    @functools.lru_cache(maxsize=4)
    def _diag_png(_key: str) -> bytes:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=80)
        top = order[:12]
        axes[0, 0].barh([uniq[i] for i in top][::-1], zc[top][::-1], color="C3")
        axes[0, 0].set_xlabel("#frames with IoU < 0.05 (radial)")
        axes[0, 0].set_title("zero-IoU frames per sequence (top 12)")
        axes[0, 1].hist(diag["r"], bins=50, color="gray", alpha=0.6, label="all frames", log=True)
        axes[0, 1].hist(diag["r"][zero], bins=50, color="C3", alpha=0.7, label="IoU < 0.05", log=True)
        axes[0, 1].set_xlabel("rgb box center distance from image center (px)")
        axes[0, 1].set_title("center distance: zero-IoU vs all")
        axes[0, 1].legend(fontsize=8)
        idx = np.linspace(0, _n - 1, 20000).astype(int)
        axes[1, 0].scatter(diag["r"][idx], iou_r[idx], s=2, alpha=0.3, color="gray", label="all frames")
        axes[1, 0].scatter(diag["r"][zero], iou_r[zero], s=2, alpha=0.3, color="C3", label="IoU < 0.05")
        axes[1, 0].set_xlabel("rgb box center distance from image center (px)")
        axes[1, 0].set_ylabel("IoU (radial)")
        axes[1, 0].set_title("IoU vs center distance")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].hist(
            diag["cdist"],
            bins=60,
            range=(0, 500),
            color="gray",
            alpha=0.6,
            label="all frames",
            log=True,
        )
        axes[1, 1].hist(
            diag["cdist"][zero],
            bins=60,
            range=(0, 500),
            color="C3",
            alpha=0.7,
            label="IoU < 0.05",
            log=True,
        )
        axes[1, 1].set_xlabel("mapped center offset, radial ir vs rgb (px)")
        axes[1, 1].set_title("center offset: zero-IoU vs all")
        axes[1, 1].legend(fontsize=8)
        fig.suptitle("diagnostics of near-zero-IoU frames - whole test set (radial)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    mo.vstack([mo.md(_summary), mo.md(_table), mo.image(_diag_png("all"), width="100%")])
    return


if __name__ == "__main__":
    app.run()
