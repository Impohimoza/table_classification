import argparse
import datetime
import os
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .data.dataset import TableDataset
from .util.logconfig import logging
from .util.util import enumerateWithEstimate
from .models.model import TableModel

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_SIZE = 3
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2


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
            '--balanced',
            help="Balance the training data to half positive, half negative.",
            action='store_true',
            default=False,
        )
        parser.add_argument(
            '--img-size',
            help="Size input img (size*size)",
            default=512,
            type=int
        )
        parser.add_argument(
            '--conv-type',
            help='Type conv blocks on model',
            default='simple',
            type=str
        )
        parser.add_argument(
            '--depth',
            help='Depth conv',
            default=4,
            type=int
        )
        parser.add_argument(
            '--checkpoint-dir',
            help='Dir for save best model',
            default='./checkpoints',
            type=str
        )
        parser.add_argument(
            '--tb-prefix',
            default='table',
            help="Data prefix to use for Tensorboard run.\
                Defaults to chapter.",
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
        self.optimizer = self.initOptimizer()
        
        self.checkpoints_dir = self.cli_args.checkpoint_dir \
            + f'/{self.cli_args.comment}'
        self.best_loss = float('inf')
        
        os.makedirs(
            self.checkpoints_dir,
            exist_ok=True)
    
    def initModel(self) -> TableModel:
        """Function for load model

        Returns:
            TableModel: returns TableModel
        """
        model = TableModel(
            conv_blocks_type=self.cli_args.conv_type,
            depth=self.cli_args.depth,
            img_size=(3, self.cli_args.img_size, self.cli_args.img_size),)
        if self.use_cuda:
            log.info("Using CUDA; {} devices."
                     .format(torch.cuda.device_count()))
            torch.cuda.empty_cache()
            
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
            self.log_memory_usage("After model loading")
        return model
    
    def log_memory_usage(self, context=""):
        if self.use_cuda:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            log.info(f"GPU Memory {context}: \
                {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def initOptimizer(self) -> Adam:
        """Function for return optimizer

        Returns:
            Adam: Adam optimizer
        """
        return Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    def initTrainDl(self) -> DataLoader:
        """Function for make Train DataLoader (Tensor, Label)

        Returns:
            DataLoader: DataLoader(TableDataset)
        """
        train_ds = TableDataset(
            val_stride=9,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced))
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
        )
        
        return train_dl
    
    def initValDl(self) -> DataLoader:
        """Function for make Validation DataLoader (Tensor, Label)

        Returns:
            DataLoader: DataLoader(TableDataset)
        """
        val_ds = TableDataset(val_stride=9, isValSet_bool=True)
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
        )
        
        return val_dl
    
    def initTensorboardWriters(self):
        """Initial Tensorboard
        """
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
        """Train algorithm
        """
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        
        best_model_path = ''
        
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
            
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            f1_metric_val = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            
            # Сохранение лучшей модели
            current_val_los = valMetrics_t[METRICS_LOSS_NDX].mean()
            if current_val_los < self.best_loss:
                self.best_loss = current_val_los
                best_model_path = self.save_checkpoint(
                    epoch_ndx, current_val_los, f1_metric_val)
            
            if hasattr(self, 'trn_writer'):
                self.trn_writer.close()
                self.val_writer.close()
        self.load_checkpoint(best_model_path)
    
    def doTraining(self, epoch_ndx: int, train_dl: DataLoader) -> torch.Tensor:
        """Model training step function

        Args:
            epoch_ndx (int): index epoch
            train_dl (DataLoader): train dataloader

        Returns:
            torch.Tensor: metrics
        """
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
            dtype=torch.float16 if self.use_cuda else torch.float32
        )
        
        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers
        )
        
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            los_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )
            los_var.backward()
            self.optimizer.step()
        
        self.totalTrainingSamples_count += len(train_dl.dataset)
        
        return trnMetrics_g.to('cpu')
    
    def doValidation(self, epoch_ndx: int, val_dl: DataLoader) -> torch.Tensor:
        """Validation step function

        Args:
            epoch_ndx (int): index epoch
            val_dl (DataLoader): Validation DataLoader

        Returns:
            torch.Tensor: val metrics
        """
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
                dtype=torch.float16 if self.use_cuda else torch.float32
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
        return valMetrics_g.to('cpu')
    
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        """Function calculate loss

        Args:
            batch_ndx (int): index batch
            batch_tup (Tuple): (Tensor, Label)
            batch_size (int): batch size
            metrics_g (Tensor): tensor metrics

        Returns:
            int: Loss mean
        """
        input_t, label_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True,
                             memory_format=torch.channels_last)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1]
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
            label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
            probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
            loss_g.detach()

        return loss_g.mean()

    def logMetrics(
            self,
            epoch_ndx: int,
            mode_str: str,
            metrics_t: torch.Tensor,
            classificationThreshold: int = 0.5,
    ) -> int:
        """Log metrics write tensorboard

        Args:
            epoch_ndx (int): index epoch
            mode_str (str): log mode (trn, val)
            metrics_t (torch.Tensor): metrics
            classificationThreshold (int, optional): classification threshold. Defaults to 0.5.

        Returns:
            int: f1 metrics
        """

        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                + "{correct/all:-5.1f}% correct, "
                + "{pr/precision:-5.1f}% precision, "
                + "{pr/recall:-5.1f}% recall, "
                + "{pr/f1_score:-5.1f}% f1 score, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, \
                {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, \
                {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, epoch_ndx)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        
        return metrics_dict['pr/f1_score']

    def save_checkpoint(self, epoch: int, loss: int, f1_metric: int) -> str:
        """Save weights model

        Args:
            epoch (int): index epoch
            loss (int): loss
            f1_metric (int): f1 metrics

        Returns:
            str: Path to save model weights
        """
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
        }
        
        filename = f'best_model_epoch_{epoch}_loss' \
            + f'_{loss:.4f}_f1_{f1_metric:.4f}.pth'
        
        # Удаляем предыдущую лучшую модель
        for f in os.listdir(self.checkpoints_dir):
            if f.startswith('best_model'):
                os.remove(os.path.join(self.checkpoints_dir, f))
        best_model_path = os.path.join(self.checkpoints_dir, filename)
        torch.save(checkpoint, best_model_path)
        return best_model_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load weights model

        Args:
            checkpoint_path (str): path to save model weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('best_loss', 0)
        
        log.info('Load best model E{}, {:.4f}'.format(epoch, loss))


if __name__ == '__main__':
    TableTrainingApp().main()