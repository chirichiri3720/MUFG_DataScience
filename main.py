
import hydra
import logging
import experiment
from experiment import ExpBase
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="main")
def main(config:DictConfig):
    exp: ExpBase = getattr(experiment, config.exp.name)(config)
    exp.run()


if __name__ == "__main__":
    main()