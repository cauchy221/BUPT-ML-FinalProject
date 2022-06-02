import logging
import torch
import os
import argparse


logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a model for sentiment analysis")
    # data
    parser.add_argument("--modle_path", type=str, default="bert-base-uncased")