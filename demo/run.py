print('run')
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

print('run1')
from detectron2.config import get_cfg
print('run2')
from detectron2.data.detection_utils import read_image
print('run3')
from detectron2.utils.logger import setup_logger
print('run4')

'''
from predictor import VisualizationDemo

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
'''
