import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional

import torch

def plot_forecast(
    true_history: np.ndarray,
    true_future: Optional[np.ndarray],
    median_forecast: np.ndarray,
    quantile_forecasts: np.ndarray,
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
    encoder_attention_map : 2D or 3D array or None
        If provided, shown as a heatmap below the time series.
        Can be shape (F, H) for single-head attention or (n_heads, F, H) for multi-head.
    decoder_attention_map : 2D or 3D array or None
        If provided, shown as a heatmap below the time series.
        Can be shape (F, F) for single-head attention or (n_heads, F, F) for multi-head.
    meal_flags : 1D bool array or None, length = H+F
        If provided, draws vertical tick at each True.
    loss_value : float or None
        If provided, appended to the title.
    show_observed_future : bool
        Whether to plot `true_future` (if given).
    """
    # — Convert torch tensors to numpy arrays if needed —
    def convert_to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    true_history = convert_to_numpy(true_history)
    true_future = convert_to_numpy(true_future)
    median_forecast = convert_to_numpy(median_forecast)
    quantile_forecasts = convert_to_numpy(quantile_forecasts)
    encoder_attention_map = convert_to_numpy(encoder_attention_map)
    decoder_attention_map = convert_to_numpy(decoder_attention_map)
    meal_flags = convert_to_numpy(meal_flags)
    
    # Process multi-head attention if provided
    # For multi-head attention, we average across heads 
    if encoder_attention_map is not None:
        if encoder_attention_map.ndim == 3:  # [n_heads, seq_len_q, seq_len_k]
            # Average across heads
            encoder_attention_map = encoder_attention_map.mean(axis=0)
            
    if decoder_attention_map is not None:
        if decoder_attention_map.ndim == 3:  # [n_heads, seq_len_q, seq_len_k]
            # Average across heads
            decoder_attention_map = decoder_attention_map.mean(axis=0)
    
    # — style & palette —
    sns.set_style("white")
    PALETTE = {
        "historical":  "#2E4057",
        "observed":    "#CB4335",
        "forecast":    "#2471A3",
        "uncertainty":"#2471A3",
        "meal":        "#7D3C98",
        "attention": [
            "#1B1B2F",  # dark base
            "#3C2C54",  # plum
            "#722744",  # red-violet
            "#A7442A",  # copper rose
            "#DA7422",  # amber
            "#F6B78C",  # peach glow
            "#FBE3D1", # champagne
        ],        
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
    
    # Generate ticks from -H to F in steps of 4, ensuring 0 is included
    neg_ticks = np.arange(0, -H, -4)[::-1]  # From 0 to -H in steps of -4, then reverse
    if neg_ticks[0] != 0:  # Ensure 0 is included
        neg_ticks = np.concatenate([[0], neg_ticks])
    pos_ticks = np.arange(4, F+1, 4)  # From 4 to F in steps of 4
    tick_rel = np.concatenate([neg_ticks, pos_ticks])
    
    tick_pos  = zero_pos + tick_rel
    valid     = (tick_pos >= 0) & (tick_pos < total_len)
    tick_pos  = tick_pos[valid]
    tick_lbl  = tick_rel[valid]
    

    # — start figure —
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    nrows = 2 if encoder_attention_map is not None else 1
    ncols = 2 if encoder_attention_map is not None else 1  # Add an extra column for the colorbar
    height_ratios = [3, 1] if encoder_attention_map is not None else [1]
    width_ratios = [60, 1] if encoder_attention_map is not None else [1]  # Main plot gets 20x width of colorbar

    if encoder_attention_map is not None:
        # Create grid with extra column for colorbar
        gs = gridspec.GridSpec(nrows, ncols, height_ratios=height_ratios, width_ratios=width_ratios,
                               hspace=0.08, wspace=0.02, figure=fig)
        ax_ts = fig.add_subplot(gs[0, 0])  # Time series in top-left
        ax_at = fig.add_subplot(gs[1, 0], sharex=ax_ts)  # Attention map in bottom-left
        cax = fig.add_subplot(gs[1, 1])  # Colorbar in bottom-right
    else:
        # Original layout when there's no attention map
        gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios,
                               hspace=0.05, figure=fig)
        ax_ts = fig.add_subplot(gs[0])
        ax_at = None
        cax = None

    # — plot history —
    ax_ts.plot(hist_idx, true_history,
               color=PALETTE["historical"], lw=2, marker="o", ms=6,
               markerfacecolor="white", markeredgecolor=PALETTE["historical"],
               label="Historical")

    # Make t=0 transition point clear
    ax_ts.axvline(x=H-1, color='black', lw=1.2, label="t=0")

    # — plot observed future —
    if show_observed_future and (true_future is not None) and true_future.size:
        # For continuity, connect last historical point to first future point
        connection_x = np.array([H-1, H])
        connection_y = np.array([true_history[-1], true_future[0]])
        ax_ts.plot(connection_x, connection_y, color=PALETTE["observed"], lw=1, alpha=0.7)
        
        # Now plot the future observations
        ax_ts.plot(fut_idx, true_future,
                   color=PALETTE["observed"], lw=2, marker="o", ms=6,
                   markerfacecolor="white", markeredgecolor=PALETTE["observed"],
                   label="Observed")

    # — plot median forecast —
    ext_i = np.concatenate([[H-1], fut_idx])
    ext_med = np.concatenate([[true_history[-1]], median_forecast])
    ax_ts.plot(ext_i, ext_med,
               color=PALETTE["forecast"], lw=2, ls="--", label="Forecast")

    # — plot uncertainty bands —
    # Flag to ensure we only add the quantile explanation to the legend once
    quantile_legend_added = False
    
    for q in range(quantile_forecasts.shape[1] - 1):
        low  = np.minimum(quantile_forecasts[:, q],   quantile_forecasts[:, q+1])
        high = np.maximum(quantile_forecasts[:, q], quantile_forecasts[:, q+1])
        alpha = 0.05 + abs(q - (quantile_forecasts.shape[1]//2))*0.02
        
        # Only add the legend entry for the middle quantile (most visible)
        label = None
        if q == quantile_forecasts.shape[1]//2 and not quantile_legend_added:
            label = "Forecast Quantiles"
            quantile_legend_added = True
        
        # First, create a smooth transition from the last historical value
        last_hist_val = true_history[-1]
        ax_ts.fill_between(
            ext_i,
            np.concatenate([[last_hist_val], low]),
            np.concatenate([[last_hist_val], high]),
            color=PALETTE["uncertainty"], alpha=alpha,
            label=label
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
    ax_ts.set_ylabel("Glucose (mmol/L)")
    ax_ts.legend(loc="upper left", frameon=True, framealpha=0.6, facecolor='white', edgecolor='none', 
                handlelength=1.5, handletextpad=0.5)
    for spine in ["top", "right"]:
        ax_ts.spines[spine].set_visible(False)

    # # — title with optional loss —
    # title = "Glucose Forecast" 
    # if loss_value is not None:
    #     title += f" (Loss: {loss_value:.4f})"
        
    # # Add the title with our new formatting
    # ax_ts.set_title(title, fontsize=14)
    
    # Add better labels
    ax_ts.set_ylabel("Glucose (mmol/L)", fontsize=12)
    
    # — attention heatmap —
    if encoder_attention_map is not None and ax_at is not None:
        # Combine encoder and decoder attention maps if both are provided
        if decoder_attention_map is not None:
            attention_map = np.concatenate([encoder_attention_map, decoder_attention_map], axis=1)
        else:
            attention_map = encoder_attention_map
            
        # Reduce space between plots
        fig.subplots_adjust(hspace=0.00)
        
        # Plot the attention map
        im = ax_at.imshow(attention_map, aspect="auto", origin="lower", cmap=attention_cmap)
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
        ax_at.spines["bottom"].set_visible(False)  # Hide the bottom (x-axis) line
        ax_at.text(
            0.98, 0.02, "Attention Map", transform=ax_at.transAxes,
            ha="right", va="bottom",
            fontsize=mpl.rcParams["legend.fontsize"],
            family=mpl.rcParams["font.family"],
            color="white", alpha=0.7
        )
        
        # Use the dedicated colorbar axis created in the GridSpec
        cbar = plt.colorbar(im, cax=cax)
        cbar.outline.set_visible(False)  # Remove the boundary/outline of the colorbar
        cax.set_ylabel("Attention Weight", fontsize=mpl.rcParams["axes.labelsize"]-1)
        cax.tick_params(labelsize=mpl.rcParams["xtick.labelsize"]-1)
        # Keep the tick labels but remove the tick markers
        cbar.ax.tick_params(size=0)  # This removes just the tick marks while keeping labels
    else:
        # When we only have the time series plot
        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.15)

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
    figs = plot_helpers.plot_forecast(
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
