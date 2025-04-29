"""Training script for all forecasting models
"""

from pathlib import Path
from omegaconf import OmegaConf
import hydra
import lightning as L
import wandb

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="config", config_name="config", version_base=None)
def train(config):
    # set up save directory
    run_dir = Path("results") / config.run_name
    # configure logging name
    arch = (
        "gru" if "GRU" in config.model.model._target_ else
        "lstm" if "LSTM" in config.model.model._target_ else
        "transformer" if "Transformer" in config.model.model._target_ else
        "sde" if "sde.Latent" in config.model.model._target_ else
        "srnn"
    )
    if arch == "srnn":
        fc_mode = "pr" if config.model.model.forecast_mode == "prior" else "po"
        ctx_mode = "f" if config.model.model.context_mode == "full" else "c" if config.model.model.context_mode == "constant" else "i"
        n_samples = str(config.model.model.posterior_samples) + "ps"
        arch = f"{arch}-{fc_mode}-{ctx_mode}-{n_samples}"
    task = (
        "multi" if "Multi" in config.datamodule._target_ else 
        "3s" if config.datamodule.pred_horizon == 3 else 
        "10s"
    )
    # initialize wandb if needed
    if "wandb_logger" in config.loggers.keys():
        wandb.init(
            project="finsde",
            name=f"{arch}-{task}-{config.run_name}",
            config=OmegaConf.to_container(config, resolve=True),
        )
    # save config for reference
    OmegaConf.save(config, run_dir / "config.yaml")
    # initialize training objects
    model = hydra.utils.instantiate(config.model)
    datamodule = hydra.utils.instantiate(config.datamodule)
    callbacks = hydra.utils.instantiate(config.callbacks)
    loggers = hydra.utils.instantiate(config.loggers)
    trainer = L.Trainer(
        callbacks=list(callbacks.values()), 
        logger=list(loggers.values()),
        **OmegaConf.to_object(config.trainer),
    )
    # train model
    trainer.fit(model, datamodule)
    # end wandb run
    if "wandb_logger" in config.loggers.keys():
        wandb.finish()


if __name__ == "__main__":
    train()