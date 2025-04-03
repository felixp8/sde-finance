from omegaconf import OmegaConf
import hydra
import lightning as L

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="config", config_name="config", version_base=None)
def train(config):
    model = hydra.utils.instantiate(config.model)
    datamodule = hydra.utils.instantiate(config.datamodule)
    callbacks = hydra.utils.instantiate(config.callbacks)
    loggers = hydra.utils.instantiate(config.loggers)
    # import pdb; pdb.set_trace()
    trainer = L.Trainer(
        callbacks=list(callbacks.values()), 
        logger=list(loggers.values()),
        **OmegaConf.to_object(config.trainer),
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()