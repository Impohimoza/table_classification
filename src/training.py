import argparse
import datetime
import os
import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.dataset import TableDataset
from .util.logconfig import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class TableTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--num-workers',
            help='Number of worker processes \
                for background data loading',
            default=4,
            type=int
        )
        parser.add_argument(
            '--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int
        )
        parser.add_argument(
            '--epochs',
            help='Number of epochs to train',
            default=20,
            type=int
        )
        parser.add_argument(
            '--tb-prefix',
            default='table',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )
        parser.add_argument(
            'comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )
        
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.model = self.initModel()
        self.optimizer = self.initOptimizer
    
    def initModel(self):
        pass
    
    def initOptimizer(self):
        return Adam(self.model.parameters(), lr=0.01)
    
    def initTrainDl(self):
        train_ds = TableDataset(val_stride=10, isValSet_bool=False)
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        
        return train_dl
    
    def initValDl(self):
        val_ds = TableDataset(val_stride=10, isValSet_bool=True)
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        
        return val_dl
    
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(
                'runs',
                self.cli_args.tb_prefix,
                self.time_str
            )
            
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment
            )
    
    def main(self):
        pass


if __name__ == '__main__':
    TableTrainingApp().main()