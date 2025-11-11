import functools
from collections import namedtuple
import glob
import os
import copy
import random

import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

from ..util.logconfig import logging
from ..util.disk import getCache

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

TableInfoTuple = namedtuple(
    'TableInfoTuple',
    'img_name, isDirty'
)

DATASET_DIR = 'dataset'
img_cache = getCache('img_cache')


@functools.lru_cache(1)
def getTableInfoList() -> TableInfoTuple:
    """
    Function to retrieve information from a table dataset

    Returns:
        TableInfoTuple: Tuple ('img_name, isDirty')
    """
    imgPath_list = glob.glob(DATASET_DIR + '/*/*.jpg')
    tableInfo_list = [TableInfoTuple(os.path.splitext(img_path)[0]
                                     .split('/')[-1], True)
                      if img_path.split('/')[-2] == 'dirty'
                      else TableInfoTuple(os.path.splitext(img_path)[0]
                                          .split('/')[-1], False)
                      for img_path in imgPath_list]
    random.shuffle(tableInfo_list)
    return tableInfo_list


class Table:
    def __init__(self, img_name: str):
        img_path = glob.glob(DATASET_DIR + f'/*/{img_name}.jpg')
        
        table_img = cv2.imread(img_path[0])
        table_img = cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB)
        
        self.img_name = img_name
        self.table_img = table_img
    
    def getTableImage(self) -> np.ndarray:
        """
        Function to get an image by its name
        Returns:
            np.ndarray: representation of the image in np.ndarray
        """
        return self.table_img


@functools.lru_cache(1, typed=True)
def getTable(img_name: str) -> Table:
    return Table(img_name)


@img_cache.memoize(typed=True)
def getTableImage(img_name: str) -> np.ndarray:
    """
    Function corresponds to function getTableImage in class Table.
    Only with results caching
    Args:
        img_name (str): image name

    Returns:
        np.ndarray: representation of the image in np.ndarray
    """
    table = getTable(img_name)
    img_a = table.getTableImage()
    return img_a


class TableDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool: str = None,
                 ratio_int=0,
                 img_name: str = None,
                 img_size=(280, 280)):
        """
        Class for work with dataset Table
        Args:
            val_stride (int, optional): Defines the step \
            for the validation sample. Defaults to 0.
            isValSet_bool (_type_, optional): Defines type \
                dataset(train, valid). Defaults to None.
            ratio_int (int, optional): Determines the ratio of positive \
                and negative classes in a data set. Defaults to 0.
            img_name (_type_, optional): Argument for obtaining a dataset \
                from a single image. Defaults to None.
        """
        self.ratio_int = ratio_int
        self.img_size = img_size
        self.tableInfo_list = copy.copy(getTableInfoList())
        
        if img_name:
            self.tableInfo_list = [
                x for x in self.tableInfo_list if x.img_name == img_name
            ]
        
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.tableInfo_list = self.tableInfo_list[::val_stride]
            assert self.tableInfo_list
        elif val_stride > 0:
            del self.tableInfo_list[::val_stride]
            assert self.tableInfo_list
        
        self.negative_list = [
            x for x in self.tableInfo_list if not x.isDirty
        ]
        self.pos_list = [
            x for x in self.tableInfo_list if x.isDirty
        ]
    
    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.pos_list)
    
    def __len__(self):
        if self.ratio_int:
            return 3000
        else:
            return len(self.tableInfo_list)
    
    def __getitem__(self, ndx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list)
                tableInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                tableInfo_tup = self.pos_list[pos_ndx]
        else:
            tableInfo_tup = self.tableInfo_list[ndx]
        
        img_a = getTableImage(tableInfo_tup.img_name)
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size),
        ])
        img_t = data_transform(img_a).to(dtype=torch.float32)
        
        pos_t = torch.tensor([
            not tableInfo_tup.isDirty,
            tableInfo_tup.isDirty
        ],
            dtype=torch.long
        )
        
        return img_t, pos_t
        
        