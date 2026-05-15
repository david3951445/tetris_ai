from training.config import Config
from training.trainer import Trainer


def main():
    config = Config()
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()