import torch
import torch.nn as nn
import torchsde
from torch.distributions import Normal, kl_divergence

import lightning as L


class Encoder(nn.Module):
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
        seq_out, fin_out = self.gru(inp)
        qz0_params = self.qz0_net(fin_out.permute(1, 0, 2).flatten(start_dim=1))
        if self.context_mode == "full":
            ctx = self.lin(seq_out)
        elif self.context_mode == "constant":
            ctx = self.lin(fin_out.permute(1, 0, 2).flatten(start_dim=1))
        else:
            ctx = torch.full(
                (seq_out.shape[0], seq_out.shape[1], self.context_size), 
                float("nan"), device=seq_out.device,
            )
        return ctx, qz0_params


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self, 
        data_size, 
        latent_size, 
        context_size, 
        hidden_size, 
        output_size, 
        context_mode="full",
        forecast_mode="prior",
    ):
        super(LatentSDE, self).__init__()
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
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
            nn.Sigmoid()
        )
        # self.g_nets = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(1, hidden_size),
        #             nn.Softplus(),
        #             nn.Linear(hidden_size, 1),
        #             nn.Sigmoid()
        #         )
        #         for _ in range(latent_size)
        #     ]
        # )
        self.mu_projector = nn.Linear(latent_size, data_size)
        self.logstd_projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self.output_projector = nn.Linear(latent_size, output_size)

        self._ctx = None

        assert forecast_mode in ["prior", "posterior"]
        self.forecast_mode = forecast_mode

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        if torch.all(torch.isnan(ctx[i])):
            return self.h_net(y)
        else:
            return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Non-diagonal diffusion.
        out = self.g_net(y)
        return out
    # def g(self, t, y):  # Diagonal diffusion.
    #     y = torch.split(y, split_size_or_sections=1, dim=1)
    #     out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
    #     return torch.cat(out, dim=1)

    def forward(self, xs, ts_in, ts_out, adjoint=False, method="euler", forecast_mode=None):
        # Contextualization is only needed for posterior inference.
        forecast_mode = forecast_mode if forecast_mode is not None else self.forecast_mode
        ctx, qz0_params = self.encoder(torch.flip(xs, dims=(0,)))
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
        self.contextualize((ts_in, ctx))

        qz0_mean, qz0_logstd = qz0_params.chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts_out, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts_out, dt=1e-2, logqp=True, method=method)

        _xs = self.mu_projector(zs[:xs.shape[0]])
        _xs_logstd = torch.exp(torch.clamp(self.logstd_projector(zs[:xs.shape[0]]), min=-10, max=2))
        xs_dist = Normal(loc=_xs, scale=_xs_logstd)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)

        output = self.output_projector(zs[xs.shape[0]:])
        return output, log_pxs, logqp0 + logqp_path

    @torch.no_grad()
    def sample(self, batch_size, ts, bm=None):
        eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h'}, dt=1e-3, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)
        return _xs