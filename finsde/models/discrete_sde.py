"""Based on https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py"""

import math
import torch
import torch.nn as nn
import torchsde
from torch.distributions import Normal, kl_divergence

import lightning as L


class Encoder(nn.Module):
    """Encoder module for latent SDE. Internally uses bidirectional GRUs.
    Has three 'modes':
        - full: uses the full GRU output over time as time-varying context, and GRU final state to set IC
        - ic_only: only uses GRU final state to set IC
        - constant: uses the GRU final state to set IC and provide constant context over time
    """
    def __init__(self, input_size, hidden_size, context_size, latent_size, num_layers=2, context_mode="full"):
        super(Encoder, self).__init__()
        assert context_mode in ["full", "ic_only", "constant"]
        self.context_mode = context_mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.latent_size = latent_size
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            bidirectional=True,
        )
        self.qz0_net = nn.Linear(hidden_size * num_layers * 2, latent_size * 2)
        if self.context_mode == "full":
            self.lin = nn.Linear(hidden_size * 2, context_size)
        elif self.context_mode == "constant":
            self.lin = nn.Linear(hidden_size * num_layers * 2, context_size)

    def forward(self, inp):
        """Forward pass through encoder. Returns context and initial latent state."""
        seq_out, fin_out = self.gru(inp)
        qz0_params = self.qz0_net(fin_out.permute(1, 0, 2).flatten(start_dim=1))
        if self.context_mode == "full":
            ctx = self.lin(seq_out)
        elif self.context_mode == "constant":
            ctx = self.lin(fin_out.permute(1, 0, 2).flatten(start_dim=1))  # (batch_size, context_size)
            ctx = ctx.unsqueeze(0).expand(seq_out.shape[0], -1, -1)  # (T, batch_size, context_size)
        else:
            ctx = torch.full(
                (seq_out.shape[0], seq_out.shape[1], self.context_size), 
                float("nan"), device=seq_out.device,
            )
        return ctx, qz0_params
    

class DiscreteLatentSDE(nn.Module):
    """Discrete latent SDE model (also called Stochastic Residual Dynamics Model (SRDM))."""
    def __init__(
        self, 
        data_size, 
        latent_size, 
        context_size, 
        hidden_size, 
        output_size, 
        context_mode="full", 
        alpha=0.1,
        forecast_mode="prior",
        kl_estimator="analytic",
        posterior_samples: int = 1,
        column_dropout: float = 0.0,
        column_group_size: int = 6,
    ):
        super(DiscreteLatentSDE, self).__init__()
        self.alpha = alpha
        assert column_dropout >= 0.0 and column_dropout <= 1.0
        self.column_dropout = column_dropout
        assert column_group_size > 0
        self.column_group_size = int(column_group_size)
        # Encoder.
        assert context_mode in ["full", "ic_only", "constant"]
        self.encoder = Encoder(
            input_size=data_size, 
            hidden_size=hidden_size, 
            context_size=context_size, 
            latent_size=latent_size, 
            context_mode=context_mode,
        )

        # Decoder.
        if context_mode != "ic_only":
            self.f_net = nn.Sequential(
                nn.Linear(latent_size + context_size, hidden_size),
                nn.Softplus(),
                nn.Linear(hidden_size, hidden_size),
                nn.Softplus(),
                nn.Linear(hidden_size, latent_size),
            )
        else:
            self.f_net = nn.Identity()  # placeholder, should never be called
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.g_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
            nn.Sigmoid()
        )

        self.mu_projector = nn.Linear(latent_size, data_size)
        self.logstd_projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self.output_projector = nn.Linear(latent_size, output_size)

        self._ctx = None

        assert forecast_mode in ["prior", "posterior"]
        self.forecast_mode = forecast_mode
        assert kl_estimator in ["analytic", "sample"]
        self.kl_estimator = kl_estimator
        self.posterior_samples = posterior_samples

    def contextualize(self, ctx):
        """Set context to use for forward passes."""
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        """Posterior drift function."""
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        if torch.all(torch.isnan(ctx[i])):
            return self.h_net(y)
        else:
            return self.f_net(torch.cat((y, ctx[i]), dim=-1))

    def h(self, t, y):
        """Prior drift function."""
        return self.h_net(y)

    def g(self, t, y):  # Non-diagonal diffusion.
        "Diffusion function."
        out = self.g_net(y)
        return out

    def forward(self, xs, ts_in, ts_out, forecast_mode=None, n_samples=None, return_trajectory=False):
        """Forward pass through model. Returns output and log likelihood of input data."""
        if self.column_dropout > 0.0:
            n_cols = xs.shape[-1]
            n_groups = n_cols // self.column_group_size
            mask = torch.bernoulli(torch.ones(n_groups, device=xs.device) * (1 - self.column_dropout))
            mask = mask.repeat_interleave(self.column_group_size, dim=0)[:n_cols]
            xs_in = xs * mask.unsqueeze(0).unsqueeze(0)  # (T, batch_size, n_cols)
        else:
            xs_in = xs
        # Contextualization is only needed for posterior inference.
        forecast_mode = forecast_mode if forecast_mode is not None else self.forecast_mode
        n_samples = n_samples if n_samples is not None else self.posterior_samples
        ctx, qz0_params = self.encoder(torch.flip(xs_in, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        if forecast_mode == "prior":
            ctx = torch.cat([
                ctx,
                torch.full(
                    (len(ts_out) - len(ts_in), ctx.shape[1], ctx.shape[2]), 
                    float("nan"), device=ctx.device),
            ])
        else:
            # posterior forecasting
            ctx = torch.cat([
                    ctx,
                    ctx[-1].unsqueeze(0).expand(len(ts_out) - len(ts_in), -1, -1)
            ])
        ctx = ctx.unsqueeze(2).expand(-1, -1, n_samples, -1)  # (T, batch_size, n_samples, context_size)
        self.contextualize((ts_in, ctx))
        qz0_mean, qz0_logstd = qz0_params.chunk(chunks=2, dim=1)
        z0 = qz0_mean.unsqueeze(-1) + qz0_logstd.exp().unsqueeze(-1) * torch.randn(
            *qz0_mean.shape, n_samples, device=qz0_mean.device)
        z0 = z0.permute(0, 2, 1)  # (batch_size, n_samples, latent_size)

        zs = [z0]
        log_ratio = 0
        for t in ts_out[:-1]:
            posterior_mean = zs[-1] + self.alpha * self.f(t, zs[-1])
            posterior_std = math.sqrt(self.alpha) * self.g(t, zs[-1])
            posterior_sample = posterior_mean + posterior_std * torch.randn_like(posterior_mean)
            prior_mean = zs[-1] + self.alpha * self.h(t, zs[-1])
            prior_std = posterior_std
            if self.kl_estimator == "analytic":
                log_ratio += kl_divergence(
                    Normal(loc=posterior_mean, scale=posterior_std),
                    Normal(loc=prior_mean, scale=prior_std),
                ).mean(dim=1)
            else:
                log_ratio += Normal(loc=prior_mean, scale=prior_std).log_prob(posterior_sample).mean(dim=1)
            zs.append(posterior_sample)
        zs = torch.stack(zs, dim=0).mean(dim=2)  # (T, batch_size, latent_size)

        _xs = self.mu_projector(zs[:xs.shape[0]])
        _xs_logstd = torch.exp(torch.clamp(self.logstd_projector(zs[:xs.shape[0]]), min=-10, max=2))
        xs_dist = Normal(loc=_xs, scale=_xs_logstd)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=1).mean(dim=0)

        output = self.output_projector(zs[xs.shape[0]:])
        if return_trajectory:
            return output, log_pxs, logqp0 + logqp_path, zs
        return output, log_pxs, logqp0 + logqp_path
