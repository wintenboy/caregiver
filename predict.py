import torch
import argparse
from pytorch_lightning import Trainer, seed_everything

from model import *


def main():
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
                        default=-1,
                        help='number of available gpus')
    parser.add_argument('--ckpt_path',
                        type=str,
                        help='checkpoint file path')
    parser.add_argument('--valid_size',
                        type=float,
                        default=0.1,
                        help='size of validation file')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=200,
                        help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    dm = CareGiverDataModule(
        data_path=args.data_path,
        valid_size=args.valid_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    dm.setup('test')

    model = CareGiverClassification()

    trainer = Trainer(
        accelerator='gpu',
        strategy=None,
        devices=args.num_gpus,
    )
    prediction = trainer.predict(model, dm, ckpt_path=args.ckpt_path)
    print(prediction)



if __name__ == '__main__':
    main()