"""Data pipeline: Dataset, sampler, stratification, normalization, augmentation, .mat I/O."""

from neural_data_decoding.data.dataset import SyntheticTrialDataset
from neural_data_decoding.data.mat_dataset import MatFileTrialDataset
from neural_data_decoding.data.mat_files import load_mat

__all__ = ["MatFileTrialDataset", "SyntheticTrialDataset", "load_mat"]
