import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    ### 1. Imports & dataset setup
    Imports all required libraries; `import marimo as mo` happens only in the code cell
    below and every other cell receives `mo` as a parameter. Resolves the `Anti-UAV-RGBT`
    dataset path (works from both `notebooks/` and the repo root) and lists all train-split
    sequence names for the dropdown. Per-sequence calibration is explored on the train split;
    applying it to test sequences without leakage is an open assumption.
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
    import marimo as mo
    import numpy as np
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    dataset_path = Path("../data/Anti-UAV-RGBT")
    if not dataset_path.exists():  # launched from the repo root instead of notebooks/
        dataset_path = Path("data/Anti-UAV-RGBT")
    # per-sequence calibration is explored on the TRAIN split for now; applying it to
    # test sequences without leakage is an open assumption (a test sequence has no
    # labels to calibrate on), to be figured out later
    seq_names = sorted(d.name for d in (dataset_path / "train").iterdir() if d.is_dir())
    return (
        LinearRegression,
        Path,
        RANSACRegressor,
        cv2,
        dataset_path,
        functools,
        io,
        json,
        mo,
        np,
        plt,
        re,
        seq_names,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### 2. Control widgets (sequence + outlier method)
    Two `mo.ui.dropdown`s: the train-split sequence selector and the outlier method for the
    per-sequence fit (`ransac` / `mad` / `none`). Both are displayed together with the other
    controls in the video-overlay cell (section 9).
    """)
    return


@app.cell(hide_code=True)
def _(mo, seq_names):
    # sequence selector (train split) + outlier method for the per-sequence fit;
    # displayed with the other controls in the videos cell
    seq_select = mo.ui.dropdown(
        options=seq_names,
        value=seq_names[0],
        label="sequence (train split)",
    )
    outlier_select = mo.ui.dropdown(
        options=["ransac", "mad", "none"],
        value="ransac",
        label="outlier method for the per-sequence fit (bin-weighted line, one fit per sequence)",
    )
    return outlier_select, seq_select


@app.cell
def _(mo):
    mo.md(r"""
    ### 3. Reactive sequence path
    Resolves the filesystem path of the currently selected train sequence. Depends on
    `seq_select.value`, so marimo re-runs this cell and all downstream per-sequence cells
    whenever the dropdown changes.
    """)
    return


@app.cell
def _(dataset_path, seq_select):
    # reactive: re-runs whenever the sequence dropdown changes
    seq = dataset_path / "train" / str(seq_select.value)
    return (seq,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 4. Cached frame access & ground-truth loading
    Scans `processed/train/<seq>/` for preprocessed PNG frames and builds a
    `get_frames(frame_id)` accessor backed by an in-memory `_cache` dict, so each visible +
    infrared frame pair is decoded by `cv2.imread` only once. Also loads the `visible.json` /
    `infrared.json` GT annotations and prints a frame-range summary.
    """)
    return


@app.cell
def _(cv2, dataset_path, json, mo, re, seq):
    # cached access to the preprocessed frames (processed/train/<seq>/<modality>/...png)
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
        f"sequence **{seq.name}** (train): {len(frame_ids)} preprocessed frames "
        f"(ids {frame_ids[0]}..{frame_ids[-1]}, step {frame_ids[1] - frame_ids[0]}), served from a cache"
    )
    return frame_ids, get_frames, gt_ir, gt_rgb


@app.cell
def _(mo):
    mo.md(r"""
    ### 5. Global reference fit (plain OLS over all train sequences)
    Pools GT box centers from every train sequence and fits a plain per-axis OLS mapping
    IR -> RGB centers (no binning, no clipping), plus a global radial gradient `a_global`.
    Used only as the dashed reference line in the fit plots and as the `global-*` baseline
    rows in the IoU table; the method under test is the per-sequence fit (section 6).
    """)
    return


@app.cell
def _(LinearRegression, Path, dataset_path, json, mo, np):
    # GLOBAL reference fit: plain per-axis OLS of rgb on ir gt box centers pooled over
    # ALL train sequences (no bins, no clipping). Only used as a dashed reference line in
    # the fit plots and as baseline rows in the IoU table - the method under test is the
    # PER-SEQUENCE fit (next cells).
    def gt_centers(seq_dir: Path) -> "tuple[np.ndarray, np.ndarray]":
        out = []
        for mod in ("visible", "infrared"):
            gt = json.loads((seq_dir / f"{mod}.json").read_text())
            rects = [r for r, e in zip(gt["gt_rect"], gt["exist"]) if e == 1 and len(r) == 4]
            if rects:
                arr = np.array(rects, dtype=float)
                out.append(arr[:, :2] + arr[:, 2:] / 2)
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

    _M = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _img_c = np.array([1920.0, 1080.0]) / 2
    _pred = (_M @ np.vstack([centers_ir.T, np.ones(len(centers_ir))]))[:2].T
    _rp = np.linalg.norm(_pred - _img_c, axis=1)
    _ra = np.linalg.norm(centers_rgb - _img_c, axis=1)
    _ratio = _ra / np.maximum(_rp, 1e-9)
    _ok = _rp > 20
    a_global = float(np.linalg.lstsq(_rp[_ok, None], _ratio[_ok] - 1, rcond=None)[0][0])

    mo.md(
        f"global reference (plain OLS, all {len(centers_rgb)} train gt boxes): "
        f"**scale** = ({scale_x:.4f}, {scale_y:.4f}), **offset** = ({offset_x:.2f}, {offset_y:.2f}), "
        f"**a_global** = {a_global:.6f}"
    )
    return a_global, offset_x, offset_y, reg_x, reg_y, scale_x, scale_y


@app.cell
def _(mo):
    mo.md(r"""
    ### 6. Per-sequence fit with selectable outlier handling
    Fits ONE bin-weighted line per axis on the current sequence's GT centers (20 bins,
    inverse bin-count weights so the dense center cluster cannot skew the line) under all
    three outlier methods: `none` (weighted OLS only), `mad` (3 iterations of 3-sigma MAD
    clipping), `ransac` (200 trials, threshold = 3*1.4826*MAD of the weighted-OLS residuals,
    final line = bin-weighted LSQ on the consensus set). Also derives the per-sequence radial
    gradient `a_seq`, the per-frame center error `frame_err`, and `bucket_frames(p)` (a +/-5-
    percentile error window) for the match-quality slider. The selected method drives the
    overlays / residual plots.
    """)
    return


@app.cell
def _(
    LinearRegression,
    RANSACRegressor,
    frame_ids,
    gt_ir,
    gt_rgb,
    mo,
    np,
    outlier_select,
    seq,
):
    # PER-SEQUENCE fit: ONE line per axis per sequence (no subfits), bin-weighted so the
    # dense center cluster cannot skew it, with a selectable outlier method:
    #   "none"  : bin-weighted OLS only
    #   "mad"   : bin-weighted OLS + iterative MAD clipping (3 iters, k=3)
    #   "ransac": RANSAC with uniform trial sampling and unweighted consensus count,
    #             residual threshold = 3*1.4826*MAD of the weighted-OLS residuals,
    #             200 trials, fixed seed, final line = bin-weighted LSQ on the consensus
    #             set (weighting the consensus vote itself was tested and is worse)
    # All three methods are computed so the fit plot can compare them; the selected one
    # drives the overlays / residuals / match slider.
    _N_BINS = 20


    def _centers(gt):
        out = []
        for r, e in zip(gt["gt_rect"], gt["exist"]):
            if e == 1 and len(r) == 4:
                out.append((r[0] + r[2] / 2, r[1] + r[3] / 2))
        return np.array(out, dtype=float).reshape(-1, 2)


    _c_rgb = _centers(gt_rgb)
    _c_ir = _centers(gt_ir)
    n = min(len(_c_rgb), len(_c_ir))
    _c_rgb, _c_ir = _c_rgb[:n], _c_ir[:n]


    def _bin_weights(x):
        edges = np.linspace(x.min(), x.max(), _N_BINS + 1)
        idx = np.clip(np.digitize(x, edges) - 1, 0, _N_BINS - 1)
        cnt = np.bincount(idx, minlength=_N_BINS).astype(float)
        w = 1.0 / np.maximum(cnt[idx], 1.0)
        return w / w.mean()


    def _fit_axis(x, y, method):
        w = _bin_weights(x)
        X = np.column_stack([x, np.ones(len(x))])
        if method == "ransac":
            # uniform trial sampling + UNWEIGHTED consensus count, bin-weighted final
            # refit on the consensus set (empirically best: weighting the consensus vote
            # over-corrects and lets sparse noisy edge bins dominate)
            beta0 = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
            r0 = y - X @ beta0
            thr = max(3.0 * 1.4826 * np.median(np.abs(r0 - np.median(r0))), 10.0)
            reg = RANSACRegressor(
                LinearRegression(), residual_threshold=thr, max_trials=200, random_state=0
            )
            reg.fit(x[:, None], y, sample_weight=w)
            return np.array(
                [float(reg.estimator_.coef_[0]), float(reg.estimator_.intercept_)]
            ), reg.inlier_mask_
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        keep = np.ones(len(x), bool)
        if method == "mad":
            for _ in range(3):
                r = y - X @ beta
                med = np.median(r[keep])
                mad = 1.4826 * np.median(np.abs(r[keep] - med))
                keep = np.abs(r - med) <= 3.0 * max(mad, 1e-9)
                beta = np.linalg.lstsq(X[keep] * w[keep, None], y[keep] * w[keep], rcond=None)[0]
        return beta, keep


    _img_c = np.array([1920.0, 1080.0]) / 2
    seq_fits = {}
    for _m in ("none", "mad", "ransac"):
        _bx, _kx = _fit_axis(_c_ir[:, 0], _c_rgb[:, 0], _m)
        _by, _ky = _fit_axis(_c_ir[:, 1], _c_rgb[:, 1], _m)
        _M = np.array([[_bx[0], 0, _bx[1]], [0, _by[0], _by[1]]], np.float64)
        _pred = (_M @ np.vstack([_c_ir.T, np.ones(n)]))[:2].T
        _rp = np.linalg.norm(_pred - _img_c, axis=1)
        _ra = np.linalg.norm(_c_rgb - _img_c, axis=1)
        _keep = _kx & _ky
        _ok = (_rp > 20) & _keep
        _a = (
            float(
                np.linalg.lstsq(_rp[_ok, None], _ra[_ok] / np.maximum(_rp[_ok], 1e-9) - 1, rcond=None)[
                    0
                ][0]
            )
            if _ok.sum() >= 5
            else 0.0
        )
        seq_fits[_m] = {"bx": _bx, "by": _by, "keep": _keep, "a": _a}

    # frame error + buckets for the SELECTED method (match slider); only preprocessed
    # frame ids get a finite error so the slider never jumps to an unprocessed frame
    _sel = str(outlier_select.value)
    _bx, _by = seq_fits[_sel]["bx"], seq_fits[_sel]["by"]
    _ids = set(frame_ids)
    frame_err = np.full(len(gt_rgb["gt_rect"]), np.inf)
    for _i, (_rv, _ri, _ev, _ei) in enumerate(
        zip(gt_rgb["gt_rect"], gt_ir["gt_rect"], gt_rgb["exist"], gt_ir["exist"])
    ):
        if _i in _ids and _ev == 1 and _ei == 1 and len(_rv) == 4 and len(_ri) == 4:
            cx_i = _bx[0] * (_ri[0] + _ri[2] / 2) + _bx[1]
            cy_i = _by[0] * (_ri[1] + _ri[3] / 2) + _by[1]
            frame_err[_i] = float(np.hypot(cx_i - (_rv[0] + _rv[2] / 2), cy_i - (_rv[1] + _rv[3] / 2)))
    valid_err = frame_err[np.isfinite(frame_err)]


    def bucket_frames(p: int) -> np.ndarray:
        # frames whose selected-method error lies in the +-5 percentile window around p
        lo = np.quantile(valid_err, max(p - 5, 0) / 100)
        hi = np.quantile(valid_err, min(p + 5, 100) / 100)
        bucket = np.flatnonzero((frame_err >= lo) & (frame_err <= hi))
        if len(bucket) == 0:
            bucket = np.array([int(np.argmin(np.abs(frame_err - np.quantile(valid_err, p / 100))))])
        return bucket


    _lines = "<br>".join(
        f"**{_m}**: scale=({seq_fits[_m]['bx'][0]:.4f}, {seq_fits[_m]['by'][0]:.4f}), "
        f"offset=({seq_fits[_m]['bx'][1]:.1f}, {seq_fits[_m]['by'][1]:.1f}), "
        f"a_seq={seq_fits[_m]['a']:.5f}, kept={seq_fits[_m]['keep'].mean():.0%}"
        for _m in ("none", "mad", "ransac")
    )
    mo.md(
        f"per-sequence fit on **{seq.name}** ({n} gt boxes), N_BINS={_N_BINS} bin-weighted, "
        f"one line per axis (no subfits), selected outlier method = **{_sel}**<br>{_lines}"
    )
    return bucket_frames, seq_fits


@app.cell
def _(mo):
    mo.md(r"""
    ### 7. Cached PNG renderers (overlays, fit plots, residuals)
    Defines the three `functools.lru_cache`-backed renderers used by the display cells:
    `overlay_png` (RGB/IR alpha blend after warping the IR frame by the per-sequence affine,
    plus an optional radial remap via a lookup table, with GT boxes overlaid), `fit_png`
    (per-axis IR->RGB center scatter with all per-method lines + the global reference), and
    `vectors_png` (mapped-IR-center residuals around the RGB box center, and residual length
    vs. target radius). Runs once per sequence/method change, not on widget changes, so the
    sliders stay cheap.
    """)
    return


@app.cell(hide_code=True)
def _(
    LinearRegression,
    cv2,
    frame_ids,
    functools,
    get_frames,
    gt_ir,
    gt_rgb,
    io,
    np,
    outlier_select,
    plt,
    reg_x,
    reg_y,
    seq,
    seq_fits,
):
    # Render functions live in this cell (runs once per sequence/method, not on widget
    # changes), with lru_cached PNG bytes. Variants:
    #   seq-reg    : per-sequence bin-weighted line with the selected outlier method
    #   seq-radial : seq-reg + per-sequence radial scaling m(r) = 1 + a_seq*r
    _variants = ("seq-reg", "seq-radial")
    _sel = str(outlier_select.value)
    _f = seq_fits[_sel]
    _M_seq = np.array([[_f["bx"][0], 0, _f["bx"][1]], [0, _f["by"][0], _f["by"][1]]], np.float64)
    _a_seq = _f["a"]
    # regression objects for every outlier method (fit plot draws all of them)
    _regs = {}
    for _m, _ff in seq_fits.items():
        _rx = LinearRegression()
        _rx.coef_ = np.array([_ff["bx"][0]])
        _rx.intercept_ = _ff["bx"][1]
        _ry = LinearRegression()
        _ry.coef_ = np.array([_ff["by"][0]])
        _ry.intercept_ = _ff["by"][1]
        _regs[_m] = (_rx, _ry)
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


    def _fwd(p, variant):
        # forward map ir coords -> rgb coords for a variant (selected outlier method)
        q = (_M_seq @ np.vstack([p.T, np.ones(len(p))]))[:2].T
        if variant == "seq-radial":
            d = q - _img_c0
            r = np.linalg.norm(d, axis=1)
            q = _img_c0 + d * (1.0 + _a_seq * r)[:, None]
        return q


    _mask_c = np.isfinite(_cur_ir).all(axis=1)
    _pred_cur = {v: _fwd(_cur_ir[_mask_c], v) for v in _variants}


    @functools.lru_cache(maxsize=8)
    def _radial_maps():
        # dst->src remap for a forward radial multiplier m(r) = 1 + a_seq*r about the
        # image center, inverted through a lookup table
        ys, xs = np.mgrid[0:_H, 0:_W].astype(np.float64)
        rq = np.hypot(xs - _img_c0[0], ys - _img_c0[1])
        rp_lut = np.linspace(0, rq.max() * 1.5, 20000)
        rq_lut = rp_lut * (1.0 + _a_seq * rp_lut)
        rp = np.interp(rq, rq_lut, rp_lut)
        scale = np.where(rq > 1e-9, rp / np.maximum(rq, 1e-9), 1.0)
        map_x = (_img_c0[0] + (xs - _img_c0[0]) * scale).astype(np.float32)
        map_y = (_img_c0[1] + (ys - _img_c0[1]) * scale).astype(np.float32)
        return map_x, map_y


    @functools.lru_cache(maxsize=256)
    def overlay_png(frame: int, bbox: bool, alpha: float, variant: str) -> bytes:
        rgb, ir = get_frames(frame)
        ir_w = cv2.warpAffine(ir, _M_seq, (rgb.shape[1], rgb.shape[0]))
        if variant == "seq-radial":
            mx, my = _radial_maps()
            ir_aligned = cv2.remap(ir_w, mx, my, cv2.INTER_LINEAR)
        else:
            ir_aligned = ir_w

        fig, ax = plt.subplots(figsize=(10, 5.6), dpi=80)
        ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), alpha=alpha)
        ax.imshow(cv2.cvtColor(ir_aligned, cv2.COLOR_BGR2RGB), alpha=1 - alpha)

        err = np.inf
        if bbox and np.isfinite(_cur_ir[frame]).all() and np.isfinite(_cur_rgb[frame]).all():
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
            wc = _fwd(corners, variant)
            ax.add_patch(plt.Polygon(wc, fill=False, edgecolor="red", lw=2, label="ir gt (aligned)"))
            ax.legend(loc="upper right", fontsize=7)
            pc = _fwd(np.array([[ri[0] + ri[2] / 2, ri[1] + ri[3] / 2]]), variant)[0]
            ac = np.array([rv[0] + rv[2] / 2, rv[1] + rv[3] / 2])
            err = float(np.hypot(pc[0] - ac[0], pc[1] - ac[1]))

        ax.set_title(
            f"[{variant} | outlier={_sel}] frame {frame} | err {err:.1f}px | rgb a {alpha:.2f}, ir a {1 - alpha:.2f}",
            fontsize=9,
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    @functools.lru_cache(maxsize=128)
    def fit_png(frame: int, variant: str) -> bytes:
        # fit plot per variant: the current sequence's own (ir -> rgb) gt centers per
        # axis; red = selected outlier method's per-seq line (straight for seq-reg,
        # binned prediction curve for seq-radial), thin dotted = the other outlier
        # methods' lines, dashed gray = global plain-OLS reference
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
        for ax2, i, name, reg_g in zip(axes, [0, 1], ["x", "y"], [reg_x, reg_y]):
            ax2.scatter(
                _cur_ir[_mask_c, i],
                _cur_rgb[_mask_c, i],
                s=10,
                color="black",
                label=f"sequence ({seq.name})",
            )
            if np.isfinite(_cur_ir[frame, i]):
                ax2.scatter(
                    [_cur_ir[frame, i]],
                    [_cur_rgb[frame, i]],
                    marker="*",
                    s=150,
                    color="magenta",
                    zorder=5,
                    label=f"frame {frame}",
                )
            ir_axis = np.linspace(_cur_ir[_mask_c, i].min(), _cur_ir[_mask_c, i].max(), 100)
            if variant == "seq-reg":
                for _m, (_rx, _ry) in _regs.items():
                    _reg = (_rx, _ry)[i]
                    if _m == _sel:
                        ax2.plot(
                            ir_axis,
                            _reg.predict(ir_axis[:, None]),
                            color="red",
                            lw=2,
                            label=f"per-seq [{_sel}]",
                        )
                    else:
                        ax2.plot(
                            ir_axis,
                            _reg.predict(ir_axis[:, None]),
                            ls=":",
                            lw=1,
                            alpha=0.8,
                            label=f"per-seq [{_m}]",
                        )
            else:
                pred = _pred_cur[variant]
                nb = 60
                edges = np.linspace(_cur_ir[_mask_c, i].min(), _cur_ir[_mask_c, i].max(), nb + 1)
                idx = np.clip(np.digitize(_cur_ir[_mask_c, i], edges) - 1, 0, nb - 1)
                cnt = np.bincount(idx, minlength=nb)
                bx = np.bincount(idx, weights=_cur_ir[_mask_c, i], minlength=nb)
                bp = np.bincount(idx, weights=pred[:, i], minlength=nb)
                ok = cnt > 0
                ax2.plot(
                    bx[ok] / cnt[ok],
                    bp[ok] / cnt[ok],
                    color="red",
                    label=f"per-seq reg+radial [{_sel}]",
                )
            ax2.plot(
                ir_axis,
                reg_g.predict(ir_axis[:, None]),
                color="gray",
                ls="--",
                label="global (all train, plain OLS)",
            )
            ax2.set_xlabel(f"ir {name} center")
            ax2.set_ylabel(f"rgb {name} center")
            ax2.legend(fontsize=7)
        fig.suptitle(f"per-sequence fit [{variant}] - {seq.name} (one line per axis, bin-weighted)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    @functools.lru_cache(maxsize=128)
    def vectors_png(frame: int, variant: str) -> bytes:
        # residual plots for one variant on the current sequence: mapped ir gt box
        # center relative to the rgb gt box center, and residual length vs target radius
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
            v_ir.append(_fwd(ci, variant)[0] - _img_c0)
            rv = gt_rgb["gt_rect"][f]
            v_rgb.append(np.array([rv[0] + rv[2] / 2, rv[1] + rv[3] / 2]) - _img_c0)
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
        axes[1].set_ylabel("|rgb box center -> ir center (mapped)| (px)")
        axes[1].legend(fontsize=8)
        axes[1].set_title("residual length vs target radius")

        fig.suptitle(
            f"center vectors [{variant} | outlier={_sel}] - {seq.name} - current frame {frame} "
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
    ### 8. Widgets & reactive frame state
    Defines `mo.state` for the current frame plus the widgets: frame slider (steps over the
    preprocessed frame ids, `on_change=set_frame`), match-quality slider (0 = very good ...
    100 = very bad; jumps to a random frame in the selected method's +/-5-percentile error
    bucket via `bucket_frames`), GT-bbox checkbox, and RGB-opacity slider (debounced so it
    re-renders only on release). Widgets live in their own cell because a marimo cell cannot
    consume its own definitions.
    """)
    return


@app.cell(hide_code=True)
def _(bucket_frames, frame_ids, mo, np):
    # widgets are defined here (marimo cells cannot consume their own definitions)
    frame_state, set_frame = mo.state(frame_ids[0])
    _step = frame_ids[1] - frame_ids[0] if len(frame_ids) > 1 else 1


    def pick_random_duo(p):
        # match slider moved: jump to a random frame in that error bucket
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
        label="match quality (0 = very good, 100 = very bad, selected-method error) -> random frame in bucket",
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
    ### 9. Video overlays (seq-reg & seq-radial)
    Stacks all controls on top, then shows the two overlay images for the current frame:
    `seq-reg` (affine only) and `seq-radial` (affine + per-sequence radial scaling
    `m(r) = 1 + a_seq*r`), rendered from the cached `overlay_png`.
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
    outlier_select,
    overlay_png,
    seq_select,
):
    # cell 1: controls on top, then the video overlays below each other
    _frame = int(frame_state())
    _bbox = bool(bbox_box.value)
    _alpha = round(float(opacity.value), 2)
    _controls = mo.vstack([seq_select, outlier_select, frame_slider, match_slider, bbox_box, opacity])
    mo.vstack(
        [
            _controls,
            mo.image(overlay_png(_frame, _bbox, _alpha, "seq-reg"), width="100%"),
            mo.image(overlay_png(_frame, _bbox, _alpha, "seq-radial"), width="100%"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 10. Fit plots (per axis, both variants)
    Per-axis IR->RGB center scatter for the current sequence with the current frame
    highlighted (magenta star): red line = selected outlier method, dotted = the other
    methods, dashed gray = global OLS reference (straight lines for `seq-reg`, binned
    prediction curve for `seq-radial`).
    """)
    return


@app.cell(hide_code=True)
def _(fit_png, frame_state, mo):
    # cell 2: fit plots, one per variant, below each other
    _frame = int(frame_state())
    mo.vstack([mo.image(fit_png(_frame, v), width="100%") for v in ("seq-reg", "seq-radial")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 11. Residual plots (both variants)
    Center-residual diagnostics for the current frame: mapped IR GT box center relative to
    the RGB GT box center in xy (left), and residual length vs. distance of the RGB box
    center from the image center (right).
    """)
    return


@app.cell(hide_code=True)
def _(frame_state, mo, vectors_png):
    # cell 3: residual plots, one per variant, below each other
    _frame = int(frame_state())
    mo.vstack([mo.image(vectors_png(_frame, v), width="100%") for v in ("seq-reg", "seq-radial")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 12. IoU of converted IR boxes vs RGB GT - all train sequences
    For every train sequence, converts the IR GT boxes into RGB coordinates under each
    variant (`global-reg`, `global-radial`, and per-sequence reg/radial x none/mad/ransac -
    one bin-weighted line per axis per sequence, no subfits) and computes IoU against the
    RGB GT boxes. All per-frame IoUs are pooled per variant (in-sample for `seq-*`).
    """)
    return


@app.cell(hide_code=True)
def _(
    LinearRegression,
    RANSACRegressor,
    a_global,
    dataset_path,
    json,
    np,
    offset_x,
    offset_y,
    scale_x,
    scale_y,
):
    # IoU between the converted ir gt boxes and the rgb gt boxes over ALL TRAIN
    # sequences (in-sample for the per-sequence variants). Each train sequence gets ONE
    # bin-weighted line per axis (no subfits), computed with each outlier method
    # (none / mad / ransac), plus its per-sequence radial gradient.
    _M_reg = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y]], np.float64)
    _img_c = np.array([1920.0, 1080.0]) / 2
    _methods = ("none", "mad", "ransac")
    _variants = (
        "global-reg",
        "global-radial",
        *[f"seq-reg-{m}" for m in _methods],
        *[f"seq-radial-{m}" for m in _methods],
    )
    _N_BINS = 20


    def _bin_weights(x):
        edges = np.linspace(x.min(), x.max(), _N_BINS + 1)
        idx = np.clip(np.digitize(x, edges) - 1, 0, _N_BINS - 1)
        cnt = np.bincount(idx, minlength=_N_BINS).astype(float)
        w = 1.0 / np.maximum(cnt[idx], 1.0)
        return w / w.mean()


    def _fit_axis(x, y, method):
        w = _bin_weights(x)
        X = np.column_stack([x, np.ones(len(x))])
        if method == "ransac":
            # uniform trial sampling + UNWEIGHTED consensus count, bin-weighted final
            # refit on the consensus set (empirically best: weighting the consensus vote
            # over-corrects and lets sparse noisy edge bins dominate)
            beta0 = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
            r0 = y - X @ beta0
            thr = max(3.0 * 1.4826 * np.median(np.abs(r0 - np.median(r0))), 10.0)
            reg = RANSACRegressor(
                LinearRegression(), residual_threshold=thr, max_trials=200, random_state=0
            )
            reg.fit(x[:, None], y, sample_weight=w)
            return np.array(
                [float(reg.estimator_.coef_[0]), float(reg.estimator_.intercept_)]
            ), reg.inlier_mask_
        beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        keep = np.ones(len(x), bool)
        if method == "mad":
            for _ in range(3):
                r = y - X @ beta
                med = np.median(r[keep])
                mad = 1.4826 * np.median(np.abs(r[keep] - med))
                keep = np.abs(r - med) <= 3.0 * max(mad, 1e-9)
                beta = np.linalg.lstsq(X[keep] * w[keep, None], y[keep] * w[keep], rcond=None)[0]
        return beta, keep


    def _fwd(p, base, av):
        q = (base @ np.vstack([p.T, np.ones(len(p))]))[:2].T
        if av is not None:
            d = q - _img_c
            r = np.linalg.norm(d, axis=1)
            q = _img_c + d * (1.0 + av * r)[:, None]
        return q


    _acc = {v: [] for v in _variants}
    for _sd in sorted((dataset_path / "train").iterdir()):
        if not _sd.is_dir():
            continue
        _gr = json.loads((_sd / "visible.json").read_text())
        _gi = json.loads((_sd / "infrared.json").read_text())
        _pairs = [
            (_rv, _ri)
            for _rv, _ri, _ev, _ei in zip(_gr["gt_rect"], _gi["gt_rect"], _gr["exist"], _gi["exist"])
            if _ev == 1 and _ei == 1 and len(_rv) == 4 and len(_ri) == 4
        ]
        if len(_pairs) < 4:
            continue
        _rv = np.array([p[0] for p in _pairs], float)
        _ri = np.array([p[1] for p in _pairs], float)
        _ci = _ri[:, :2] + _ri[:, 2:] / 2
        _cr = _rv[:, :2] + _rv[:, 2:] / 2
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
        _per_method = {}
        for _m in _methods:
            _bx, _kx = _fit_axis(_ci[:, 0], _cr[:, 0], _m)
            _by, _ky = _fit_axis(_ci[:, 1], _cr[:, 1], _m)
            _M_seq = np.array([[_bx[0], 0, _bx[1]], [0, _by[0], _by[1]]], np.float64)
            _pred = (_M_seq @ np.vstack([_ci.T, np.ones(len(_ci))]))[:2].T
            _rp = np.linalg.norm(_pred - _img_c, axis=1)
            _ra = np.linalg.norm(_cr - _img_c, axis=1)
            _ok = (_rp > 20) & (_kx & _ky)
            _a = (
                float(
                    np.linalg.lstsq(
                        _rp[_ok, None], _ra[_ok] / np.maximum(_rp[_ok], 1e-9) - 1, rcond=None
                    )[0][0]
                )
                if _ok.sum() >= 5
                else 0.0
            )
            _per_method[_m] = (_M_seq, _a)
        for v in _variants:
            if v == "global-reg":
                _base, _av = _M_reg, None
            elif v == "global-radial":
                _base, _av = _M_reg, a_global
            elif v.startswith("seq-reg-"):
                _base, _av = _per_method[v[len("seq-reg-") :]][0], None
            else:
                _base, _av = _per_method[v[len("seq-radial-") :]]
            _q = _fwd(_corners.reshape(-1, 2), _base, _av).reshape(-1, 4, 2)
            _px0, _py0 = _q.min(axis=1).T
            _px1, _py1 = _q.max(axis=1).T
            _iw = np.clip(np.minimum(_px1, _rx1) - np.maximum(_px0, _rx0), 0, None)
            _ih = np.clip(np.minimum(_py1, _ry1) - np.maximum(_py0, _ry0), 0, None)
            _inter = _iw * _ih
            _union = (_px1 - _px0) * (_py1 - _py0) + (_rx1 - _rx0) * (_ry1 - _ry0) - _inter
            _acc[v].append(np.where(_union > 0, _inter / np.maximum(_union, 1e-12), 0.0))
    iou_ps = {v: np.concatenate(_acc[v]) for v in _variants}
    return (iou_ps,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 13. IoU metrics table & AP curves
    Markdown table over all train sequences (mean/median IoU, AP@0.5, AP@0.75, mAP@.5:.95,
    #frames; RANSAC variants bolded) plus two plots: IoU histograms of the `seq-reg-*`
    variants and AP-vs-threshold curves for all variants.
    """)
    return


@app.cell(hide_code=True)
def _(functools, io, iou_ps, mo, np, plt):
    # IoU metrics over all train sequences: per-sequence bin-weighted fits with each
    # outlier method vs the global baseline. In-sample for the per-sequence variants.
    # AP@t = fraction of frames with IoU >= t (one box per modality per frame).
    _thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    # ransac variants first (main comparison, bolded in the table, thicker in the plot)
    _variants = tuple(
        sorted(
            iou_ps.keys(),
            key=lambda v: (0 if v.endswith("-ransac") else 1 if v.startswith("seq-") else 2, v),
        )
    )
    _colors = {
        "global-reg": "C0",
        "global-radial": "C1",
        "seq-reg-none": "C2",
        "seq-reg-mad": "C3",
        "seq-reg-ransac": "C4",
        "seq-radial-none": "C5",
        "seq-radial-mad": "C6",
        "seq-radial-ransac": "C7",
    }


    def _ap(ious, t):
        return float(np.mean(ious >= t)) if len(ious) else float("nan")


    def _row(variant):
        x = iou_ps[variant]
        _v = f"**{variant}**" if variant.endswith("-ransac") else variant
        return (
            f"| all train | {_v} | {np.mean(x):.3f} | {np.median(x):.3f} "
            f"| {_ap(x, 0.5):.3f} | {_ap(x, 0.75):.3f} "
            f"| {np.mean([_ap(x, t) for t in _thresholds]):.3f} | {len(x)} |"
        )


    _table = "\n".join(
        [
            "IoU of converted ir gt box vs rgb gt box - all train sequences pooled. "
            "seq-* variants: one bin-weighted line per axis fitted on each sequence's own "
            "labels (in-sample; the test-without-leakage question is an open assumption), "
            "with outlier method none / mad / ransac. "
            "AP@t = fraction of frames with IoU >= t; mAP = mean AP over t = 0.50:0.05:0.95",
            "",
            "| scope | variant | mean IoU | median IoU | AP@0.5 | AP@0.75 | mAP@.5:.95 | #frames |",
            "|---|---|---|---|---|---|---|---|",
            *[_row(v) for v in _variants],
        ]
    )


    @functools.lru_cache(maxsize=4)
    def _iou_png(_key: str) -> bytes:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
        for v in _variants:
            if v.startswith("seq-reg-"):
                axes[0].hist(
                    iou_ps[v], bins=np.linspace(0, 1, 41), alpha=0.6, color=_colors[v], label=v
                )
        axes[0].set_xlabel("IoU (converted ir box vs rgb gt box)")
        axes[0].set_ylabel("frames")
        axes[0].legend(fontsize=7)
        axes[0].set_title(
            f"IoU distribution (seq-reg variants) - all train ({len(iou_ps['global-reg'])} frames)"
        )
        for v in _variants:
            _main = v.endswith("-ransac")
            axes[1].plot(
                _thresholds,
                [_ap(iou_ps[v], t) for t in _thresholds],
                color=_colors[v],
                label=v,
                lw=2.5 if _main else 1.0,
                alpha=1.0 if _main else 0.45,
            )
        axes[1].set_xlabel("IoU threshold t")
        axes[1].set_ylabel("AP@t")
        axes[1].set_ylim(0, 1.02)
        axes[1].legend(fontsize=7)
        axes[1].set_title("AP vs IoU threshold - all train")
        fig.suptitle(
            "IoU metrics: per-sequence bin-weighted fits by outlier method vs global - all train (in-sample for seq-*)"
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()


    mo.vstack([mo.md(_table), mo.image(_iou_png("all"), width="100%")])
    return


if __name__ == "__main__":
    app.run()
