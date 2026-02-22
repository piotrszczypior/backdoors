from parser import get_args_parser, get_config
import torch
from config.ConfigLoader import GlobalConfig
from models.ModelFactory import ModelFactory
from dataset import ImageNetDataModule
from backdoors.BackdooredDatasetFactory import BackdooredDatasetFactory
from train import train
from torch.utils.data import DataLoader


def main(config: GlobalConfig):
    pass

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = get_config(args)
    main(config)
