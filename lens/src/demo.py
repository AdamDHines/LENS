# fast_demo.py  ───────────────────────────────────────────────────────────
import glob, os, re
from bisect import bisect_right

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from matplotlib.patches import Rectangle

# ─────────────────────────────────────────────────────────────────────────
def _sorted_pngs(folder):
    files = glob.glob(os.path.join(folder, "*.png"))
    files.sort(key=lambda p: int(re.search(r"_(\d+)\.png$", p).group(1)))
    return files


# ─────────────────────────────────────────────────────────────────────────
def demo(
    data_dir, dataset, camera,
    query, reference,
    dist_matrix_seq, GTtol,
    N, R, LENS_R, LENS_P,
    event_streams,
    skip=60,
):
    # ─── bookkeeping to map global frame → (query_i, event_i) ───────────
    lengths = [
        int(np.ceil((e.shape[0] if e.ndim == 3 else 1) / skip))
        for e in event_streams
    ]
    cumlen   = np.cumsum(lengths)
    n_frames = int(cumlen[-1])

    def locate(global_f):
        q_idx = bisect_right(cumlen, global_f)
        base  = 0 if q_idx == 0 else cumlen[q_idx - 1]
        e_idx = (global_f - base) * skip
        return q_idx, e_idx

    # ─── reference images ──────────────────────────────────────────────
    r_dir    = os.path.join(data_dir, dataset, camera, reference)
    ref_pngs = _sorted_pngs(r_dir)

    # ─── figure & static artists ───────────────────────────────────────
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(14, 8), dpi=120, constrained_layout=True,
                     facecolor="#f8f9fa")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
            width_ratios=[1, 1, 1.25], height_ratios=[1, 1])
    ax_q, ax_ref, ax_heat  = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_line, ax_rec, ax_pr = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for ax in (ax_q, ax_ref):
        ax.axis("off")
    ax_q.set_title("Query – event stream", fontsize=12, weight="bold")
    ax_ref.set_title("Matched reference", fontsize=12, weight="bold")
    ax_heat.set_title("Similarity matrix", fontsize=12, weight="bold")

    # similarity matrix
    vmin, vmax = dist_matrix_seq.min(), dist_matrix_seq.max()
    vis_mat    = np.full_like(dist_matrix_seq, np.nan, dtype=float)
    im_heat    = ax_heat.imshow(vis_mat, aspect="auto", cmap="viridis",
                                vmin=vmin, vmax=vmax)
    ax_heat.set_xlabel("Query idx"); ax_heat.set_ylabel("Reference idx")

    # recall @ K
    ax_rec.plot(N, R, marker="o", linewidth=2, color="#2087c2")
    ax_rec.scatter(N, R, s=50, c="#20c27e", zorder=3, edgecolors="white")
    ax_rec.set_xticks(N); ax_rec.set_ylim(0, 1)
    ax_rec.set_title("Recall @ K", fontsize=12, weight="bold"); ax_rec.set_ylabel("Recall")

    # PR curve
    ax_pr.plot(LENS_R, LENS_P, linewidth=2, color="#c22088")
    ax_pr.scatter(LENS_R, LENS_P, s=40, color="#2087c2", zorder=3, edgecolors="white")
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.05)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR curve", fontsize=12, weight="bold")

    # distance vector
    line, = ax_line.plot([], [], lw=2, color="#ffb000")
    ax_line.set_xlim(0, dist_matrix_seq.shape[0] - 1)
    ax_line.set_title("Similarities for current query", fontsize=12, weight="bold")
    ax_line.set_xlabel("Reference idx"); ax_line.set_ylabel("Similarity")
    d_min, d_max = dist_matrix_seq.min(), dist_matrix_seq.max()
    ax_line.set_ylim(d_min * 0.95, d_max * 1.05)

    # first event slice sets size
    first_evt = event_streams[0][0] if event_streams[0].ndim == 3 else event_streams[0]
    im_q   = ax_q.imshow(first_evt, cmap="gray", vmin=0, vmax=1)
    blank  = np.zeros_like(first_evt)
    im_ref = ax_ref.imshow(blank, cmap="viridis", vmin=0, vmax=255)
    border = Rectangle((0, 0), 1, 1, transform=ax_ref.transAxes,
                       fill=False, linewidth=4)
    ax_ref.add_patch(border)

    # keep scatter markers so they persist even with blit=True
    scatter_pts = []
    processed   = set()

    # ─── init_func for blitting ─────────────────────────────────────────
    def init():
        return [im_q, im_ref, im_heat, line, border]

    # ─── update callback ────────────────────────────────────────────────
    def update(global_f):
        q_idx, e_idx = locate(global_f)
        evt_src = event_streams[q_idx]
        evt_slice = evt_src[e_idx] if evt_src.ndim == 3 else evt_src
        im_q.set_array(evt_slice)

        if q_idx not in processed:
            ref_idx = int(np.argmax(dist_matrix_seq[:, q_idx]))
            im_ref.set_data(imageio.imread(ref_pngs[ref_idx]))
            correct = bool(GTtol[ref_idx, q_idx])
            border.set_edgecolor("#2ecc71" if correct else "#e74c3c")

            vis_mat[:, q_idx] = dist_matrix_seq[:, q_idx]
            im_heat.set_data(vis_mat)
            col_col = "#2ecc71" if correct else "#e74c3c"
            scatter_pts.append(
                ax_heat.scatter(q_idx, ref_idx, s=50, c=col_col,
                                edgecolors="black", linewidths=0.5, zorder=4)
            )
            line.set_data(np.arange(dist_matrix_seq.shape[0]),
                          dist_matrix_seq[:, q_idx])
            processed.add(q_idx)

        return [im_q, im_ref, im_heat, line, border] + scatter_pts

    # ─── run ────────────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init,
        interval=1,
        blit=True, repeat=False
    )
    plt.show()
