import torch
import torchmetrics
from typing import List, Optional

from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from pytorch_forecasting.metrics import RMSE

from utils import calculate_symmetric_quantiles
from config import Config

def build_loss(config: Config) -> torch.nn.Module:
    """Build and return the loss function based on the configuration."""
    
    quantiles = calculate_symmetric_quantiles(config.num_quantiles)
    if config.loss == "QuantileLoss":
        return QuantileLoss(quantiles=quantiles)
    elif config.loss == "ApproximateCRPS":
        return ApproximateCRPSMetric(quantiles=quantiles)
    elif config.loss == "RMSE":
        return RMSE()
    else:
        raise ValueError(f"Invalid loss function: {config.loss}")
    

class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles, **kwargs)
        
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses


    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred


    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred



class ApproximateCRPSMetric(QuantileLoss):
    """
    Approximates the Continuous Ranked Probability Score (CRPS) for quantile forecasts.
    
    This version integrates over the full [0, 1] quantile range by:
      - Approximating the integral over [0, q_last] using a discrete Riemann sum.
      - Adding an extra term for the interval [q_last, 1].
    
    The `forward` method returns the CRPS loss and
    `to_prediction` is overridden to return a point forecast (the median) with an
    extra dimension (shape `[batch, time, 1]`) so that the base metric’s assertion is met.
    """
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Compute the 2x pinball losses for each quantile.
        pinball_losses = super().loss(y_pred, target)
        # pinball_losses shape: [batch, time, num_quantiles]
        
        # 2. Create a tensor for the quantiles (ensuring same device and dtype).
        q_t = torch.tensor(self.quantiles, device=y_pred.device, dtype=pinball_losses.dtype)
        
        # 3. Compute differences for integration over [0, q_last]:
        #    For the first quantile, use (q0 - 0); for others, q_i - q_{i-1}.
        differences = torch.diff(q_t, prepend=torch.tensor([0.0], device=y_pred.device, dtype=pinball_losses.dtype))
        differences = differences.view(1, 1, -1)  # reshape for broadcasting
        
        # 4. Approximate the integral over [0, q_last] via a Riemann sum.
        crps = (pinball_losses * differences).sum(dim=-1, keepdim=True)
        
        # 5. Add the extra term for the region [q_last, 1]:
        last_gap = 1.0 - q_t[-1]  # gap from last quantile to 1
        last_term = last_gap * pinball_losses[..., -1:]
        crps += last_term
        
        return crps


###############################################################################
# Weighted Interval Score (WIS)
###############################################################################

class WeightedIntervalScore(QuantileLoss):
    r"""
    Computes the Weighted Interval Score (WIS) for quantile forecasts.
    
    For a set of symmetric prediction intervals (i.e. quantile pairs) and the median,
    the WIS is defined as

    \[
    \text{WIS} = \frac{1}{K + 0.5} \left[ \frac{1}{2}\,|y - m| + \sum_{k=1}^{K} \left\{ \frac{\alpha_k}{2}(U_k - L_k) + (L_k - y)_+ + (y - U_k)_+ \right\}\right],
    \]
    
    where:
      - \(m\) is the median forecast,
      - \(L_k\) and \(U_k\) are the lower and upper forecasts of the \(k\)th interval,
      - \(\alpha_k = 2 \, q_k\) is the nominal miscoverage probability (with \(q_k\) the lower quantile level),
      - \(K = \frac{n_{\text{quantiles}}-1}{2}\) is the number of intervals, and
      - \((x)_+ = \max(x,0)\).
      
    The `loss` method returns a tensor of shape `[batch, time, 1]`.
    """
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # y_pred is assumed to have shape [batch, time, n_quantiles]
        # and self.quantiles is a sorted list that contains 0.5.
        
        # ---- Median Component ----
        median_idx = self.quantiles.index(0.5)
        median = y_pred[..., median_idx]  # shape: [batch, time]
        median_comp = 0.5 * torch.abs(target - median)
        
        # ---- Interval Components ----
        # Number of prediction intervals (assumes median is included).
        K = (len(self.quantiles) - 1) // 2
        
        interval_comp = 0.0
        for i in range(K):
            # Lower prediction at index i and upper prediction at index -(i+1)
            L = y_pred[..., i]          # lower quantile forecast
            U = y_pred[..., -(i + 1)]     # upper quantile forecast
            
            # For symmetric intervals, the nominal miscoverage is:
            #   alpha = 2 * (lower quantile level)
            alpha = 2 * self.quantiles[i]
            
            # The weighted contribution of the interval:
            #   (alpha/2)*(U - L) + (L - y)_+ + (y - U)_+
            width_term   = (alpha / 2.0) * (U - L)
            lower_penalty = torch.clamp(L - target, min=0)
            upper_penalty = torch.clamp(target - U, min=0)
            
            interval_comp = interval_comp + width_term + lower_penalty + upper_penalty
        
        # ---- Combine Components ----
        denom = K + 0.5
        wis = (median_comp + interval_comp) / denom
        # Return with an extra dimension to satisfy downstream expectations.
        return wis.unsqueeze(-1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        # Return the median forecast with an extra dimension.
        median_idx = self.quantiles.index(0.5)
        median = y_pred[..., median_idx]
        return median.unsqueeze(-1)

    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(y_pred, target)
