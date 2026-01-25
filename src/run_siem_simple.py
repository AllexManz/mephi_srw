"""
Minimal example for SIEM integration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from integration.siem_integration import SIEMIntegration


@hydra.main(version_base=None, config_path="../configs", config_name="integration/default")
def main(cfg: DictConfig) -> None:
    """Run a single-pass SIEM analysis."""
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))

    integration = SIEMIntegration(
        model_path=cfg.model.model.name,
        cfg=cfg,
        use_lora=cfg.peft.peft.enabled
    )

    result = integration.analyze_events()
    print(result)


if __name__ == "__main__":
    main()
