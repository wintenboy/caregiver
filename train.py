import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *

def main() :

    seed_everything(seed=42, workers=True)

    dm = CareGiverDataModule(
        data_path='testing',
        mode='clean',
        valid_size=0.1,
        max_seq_len=50,
        batch_size=16,
    )
    dm.prepare_data()
    dm.setup('fit')

    model = CareGiverClassification()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='{epoch:02d}-{val_acc:.3f}',
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
    )

    trainer.fit(model, dm)
if __name__ == '__main__':
    main()