import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--data_path',
                        type=str,
                        default='caregiver_data',
                        help='where to prepare data')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,
                        help='number of available gpus')
    parser.add_argument('--ckpt_path',
                        type=str,
                        help='checkpoint file path')
    parser.add_argument('--valid_size',
                        type=int,
                        defult=0.1,
                        help='size of validation file')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=200,
                        help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--work_dirs',
                        type=str,
                        default='caregiver_ckpt',
                        help='directory of checkpoints file')
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    dm = CareGiverDataModule(
        data_path=args.data_path,
        valid_size=args.valid_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    dm.prepare_data()
    dm.setup('fit')

    model = CareGiverClassification()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=args.work_dir,
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
        max_epochs=30,
        accelerator='gpu',
        devices=args.num_gpus,
        callbacks=[checkpoint_callback, early_stopping],
    )

    trainer.fit(model, dm)
if __name__ == '__main__':
    main()