import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ColaModel


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        '''
            https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html 
            # fast_dev_run : unit test by running n, if set to n. (else 1). The point is to detect any bugs in the training/validation loop without having to wait for a full epoch to crash. 
            # runs 1 train, val, test batch and program ends
            trainer = Trainer(fast_dev_run=True)

            # runs 7 train, val, test batches and program ends
            trainer = Trainer(fast_dev_run=7)
        '''
        fast_dev_run=False, 
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
