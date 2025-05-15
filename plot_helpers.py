import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional

def plot_forecast(
    true_history: np.ndarray,
    true_future: Optional[np.ndarray],
    median_forecast: np.ndarray,
    quantile_forecasts: np.ndarray,
    raw_forecast: Optional[np.ndarray] = None,
    encoder_attention_map: Optional[np.ndarray] = None,
    decoder_attention_map: Optional[np.ndarray] = None,
    meal_flags: Optional[np.ndarray] = None,
    loss_value: Optional[float] = None,
    show_observed_future: bool = True,
) -> plt.Figure:
    """
    Plot historical + observed + forecast with uncertainty bands and optional attention heatmap.
    
    Parameters
    ----------
    true_history : 1D array, length = H
        The observed values before t=0.
    true_future : 1D array or None, length = F
        The actual observed values after t=0 (if available).
    median_forecast : 1D array, length = F
        The median forecast values.
    quantile_forecasts : 2D array, shape = (F, Q)
        The quantile forecasts (each column is one quantile level).
    raw_forecast : 1D array or None, length = F
        The raw model outputs, for loss‐computing or plotting (optional).
    encoder_attention_map : 2D array or None, shape = (F, H)
        If provided, shown as a heatmap below the time series.
    decoder_attention_map : 2D array or None, shape = (F, F)
        If provided, shown as a heatmap below the time series.
    meal_flags : 1D bool array or None, length = H+F
        If provided, draws vertical tick at each True.
    loss_value : float or None
        If provided, appended to the title.
    show_observed_future : bool
        Whether to plot `true_future` (if given).
    """
    # — style & palette —
    sns.set_style("white")
    PALETTE = {
        "historical":  "#2E4057",
        "observed":    "#CB4335",
        "forecast":    "#2471A3",
        "uncertainty":"#2471A3",
        "meal":        "#7D3C98",
        "attention":   ["white", "#2E4057"]
    }
    attention_cmap = LinearSegmentedColormap.from_list(
        "attention_cmap", PALETTE["attention"], N=512
    )
    mpl.rcParams.update({
        'font.family':'serif',
        'font.serif':['DejaVu Serif','Liberation Serif','serif'],
        'font.size':10,
        'axes.titlesize':12,
        'axes.labelsize':11,
        'xtick.labelsize':10,
        'ytick.labelsize':10,
        'legend.fontsize':10,
        'axes.linewidth':1.2,
        'xtick.direction':'in',
        'ytick.direction':'in',
        'xtick.major.width':1.0,
        'ytick.major.width':1.0,
    })

    # — unwrap shapes —
    H = true_history.size
    F = median_forecast.size
    total_len = H + F

    # — build index vectors —
    hist_idx = np.arange(H)
    fut_idx  = H + np.arange(F)
    combined_idx = np.arange(total_len)

    # — relative time labels —
    rel_steps = np.concatenate([np.arange(-H+1, 1), np.arange(1, F+1)])
    zero_pos  = int(np.where(rel_steps == 0)[0][0])
    span      = max(-rel_steps.min(), rel_steps.max())
    tick_rel  = np.arange(-span, span+1, 4)
    tick_pos  = zero_pos + tick_rel
    valid     = (tick_pos >= 0) & (tick_pos < total_len)
    tick_pos  = tick_pos[valid]
    tick_lbl  = tick_rel[valid]

    # — start figure —
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    nrows = 2 if encoder_attention_map is not None else 1
    height_ratios = [3, 1] if encoder_attention_map is not None else [1]
    gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios,
                           hspace=0.3, figure=fig)
    ax_ts = fig.add_subplot(gs[0])
    ax_at = fig.add_subplot(gs[1], sharex=ax_ts) if encoder_attention_map is not None else None

    # — plot history —
    ax_ts.plot(hist_idx, true_history,
               color=PALETTE["historical"], lw=2, marker="o", ms=6,
               markerfacecolor="white", markeredgecolor=PALETTE["historical"],
               label="Historical")

    # — plot observed future —
    if show_observed_future and (true_future is not None) and true_future.size:
        ext_i = np.concatenate([[H-1], fut_idx])
        ext_v = np.concatenate([[true_history[-1]], true_future])
        ax_ts.plot(ext_i, ext_v,
                   color=PALETTE["observed"], lw=2, marker="o", ms=6,
                   markerfacecolor="white", markeredgecolor=PALETTE["observed"],
                   label="Observed")

    # — plot median forecast —
    ext_i = np.concatenate([[H-1], fut_idx])
    ext_med = np.concatenate([[true_history[-1]], median_forecast])
    ax_ts.plot(ext_i, ext_med,
               color=PALETTE["forecast"], lw=2, ls="--", label="Forecast")

    # — plot uncertainty bands —
    for q in range(quantile_forecasts.shape[1] - 1):
        low  = np.minimum(quantile_forecasts[:, q],   quantile_forecasts[:, q+1])
        high = np.maximum(quantile_forecasts[:, q], quantile_forecasts[:, q+1])
        alpha = 0.05 + abs(q - (quantile_forecasts.shape[1]//2))*0.02
        ax_ts.fill_between(
            ext_i,
            np.concatenate([[true_history[-1]], low]),
            np.concatenate([[true_history[-1]], high]),
            color=PALETTE["uncertainty"], alpha=alpha
        )

    # — meal event lines —
    if meal_flags is not None:
        for pos, flag in enumerate(meal_flags):
            if flag:
                ax_ts.axvline(pos, ls=":", lw=1.2, color=PALETTE["meal"],
                              label="Meal"
                                if "Meal" not in ax_ts.get_legend_handles_labels()[1]
                                else None)

    # — axes formatting —
    ax_ts.set_xticks(tick_pos)
    ax_ts.set_xticklabels(tick_lbl)
    ax_ts.axvline(zero_pos, color="black", lw=1.2, label="t=0")
    ax_ts.set_ylabel("Glucose (mmol/L)")
    ax_ts.legend(loc="upper left", frameon=False, handlelength=1.5, handletextpad=0.5)
    for spine in ["top", "right"]:
        ax_ts.spines[spine].set_visible(False)

    # — title with optional loss —
    title = "Glucose Forecast"
    if loss_value is not None:
        title += f" (Loss: {loss_value:.3f})"
    ax_ts.set_title(title)

    # — attention heatmap —
    if encoder_attention_map is not None and decoder_attention_map is not None and ax_at is not None:
        attention_map = np.concatenate([encoder_attention_map, decoder_attention_map], axis=1)
        ax_at.imshow(attention_map, aspect="auto", origin="lower", cmap=attention_cmap)
        ax_at.set_ylabel("Forecast Step")
        ax_at.set_xlabel("Relative Timestep\n(15 min intervals)")
        ax_at.set_xticks(tick_pos)
        ax_at.set_xticklabels(tick_lbl)
        y_ticks = np.arange(0, attention_map.shape[0], 4)
        ax_at.set_yticks(y_ticks)
        ax_at.set_yticklabels(y_ticks + 1)
        for spine in ["top", "right"]:
            ax_at.spines[spine].set_visible(False)
        ax_at.spines["left"].set_edgecolor("grey")
        ax_at.spines["bottom"].set_edgecolor("grey")
        ax_at.text(
            0.98, 0.02, "Attention Map", transform=ax_at.transAxes,
            ha="right", va="bottom",
            fontsize=mpl.rcParams["legend.fontsize"],
            family=mpl.rcParams["font.family"],
            color="black", alpha=0.7
        )

    fig.tight_layout()
    return [fig]


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    H, F, Q = 32, 16, 5

    # 1) simulate core arrays
    hist = np.cumsum(np.random.randn(H))
    fut  = np.cumsum(np.random.randn(F))
    med  = np.cumsum(np.random.randn(F))
    quants = np.stack([med + (i - Q//2)*0.5 for i in range(Q)], axis=1)
    meals  = np.random.rand(H+F) < 0.1  # random 10% meal events
    encoder_attention_map    = np.random.rand(F, H)     # e.g. 8 forecast‐step attention
    decoder_attention_map    = np.random.rand(F, F)     # e.g. 8 forecast‐step attention
    
    # Add a causal mask to the decoder attention map
    decoder_attention_map[np.triu_indices(F, k=1)] = 0

    # 2) plot
    fig = plot_helpers.plot_forecast(
        true_history=hist,
        true_future=fut,
        median_forecast=med,
        quantile_forecasts=quants,
        meal_flags=meals,
        encoder_attention_map=encoder_attention_map,
        decoder_attention_map=decoder_attention_map,
        loss_value=0.123
    )
    plt.show()
