import os
import pickle
from collections import defaultdict
from typing import Tuple

import tensorflow as tf
from absl import logging
from tqdm import tqdm

from baselines.diabetic_retinopathy_detection.utils import load_eval_results


def load_dataset_dir(base_path, dataset_subdir):
  results = defaultdict(list)
  dataset_subdir_path = os.path.join(base_path, dataset_subdir)
  random_seed_dirs = tf.io.gfile.listdir(dataset_subdir_path)
  seeds = [int(random_seed_dir.split('_')[-1].split('/')[0])
           for random_seed_dir in random_seed_dirs]
  seeds = sorted(seeds)
  for seed in tqdm(seeds, desc="loading seed results...", disable=True):
    eval_results = load_eval_results(
      eval_results_dir=dataset_subdir_path, epoch=seed)
    for arr_name, arr in eval_results.items():
      if arr.ndim > 0 and arr.shape[0] > 1:
        results[arr_name].append(arr)
  return results


def load_list_datasets_dir(base_path):
  dataset_results = {}
  dataset_subdirs = [
    file_or_dir for file_or_dir in tf.io.gfile.listdir(base_path)
    if tf.io.gfile.isdir(os.path.join(base_path, file_or_dir))]

  for dataset_subdir in tqdm(dataset_subdirs, desc="loading datasets results..", disable=True):
    dataset_name = dataset_subdir.strip("/")
    logging.info(dataset_name)
    dataset_results[dataset_name] = load_dataset_dir(base_path=base_path, dataset_subdir=dataset_subdir)
  return dataset_results


def load_model_dir_result_with_cache(model_dir_path, cache_file_name="cache",
                                     invalid_cache=False):
  cache_path = os.path.join(model_dir_path, cache_file_name)
  if tf.io.gfile.exists(cache_path) and not invalid_cache:
    logging.info(f"Reading cache from {cache_path}")
    with tf.io.gfile.GFile(cache_path, "rb") as f:
      dataset_results = pickle.load(f)
  else:
    # Tuning domain is either `indomain`, `joint` in our implementation.
    # not using lambda otherwise it is not possible to pickle
    eval_types = [agg for agg in tf.io.gfile.listdir(model_dir_path)
                if tf.io.gfile.isdir(os.path.join(model_dir_path, agg))]
    dataset_results = {}
    for eval_type in tqdm(eval_types):
      dataset_results[eval_type] = load_list_datasets_dir(os.path.join(model_dir_path, eval_type))
    if not len(dataset_results):
      logging.info(f"{model_dir_path} is empty directory, won't create cache file")
      return {}
    logging.info(f"Caching result in {model_dir_path} in file {cache_path}...")
    with tf.io.gfile.GFile(cache_path, "wb") as f:
      pickle.dump(dataset_results, f)

  dataset_results = {k.strip("/"): v for k, v in dataset_results.items()}
  return dataset_results


def parse_model_dir_name(model_dir: str) -> Tuple:
  try:
    model_type, ensemble_str, tuning_domain, mc_str = model_dir.split('_')
  except:
    raise ValueError('Expected model directory in format '
                     '{model_type}_k{k}_{tuning_domain}_mc{n_samples}')
  k = int(ensemble_str[1:])  # format f'k{k}'
  num_mc_samples = mc_str[2:][:-1]  # format f'mc{num_mc_samples}/'
  is_deterministic = model_type == 'deterministic' and k == 1
  key = (model_type, k, is_deterministic, tuning_domain, num_mc_samples)
  return key


def fast_load_dataset_to_model_results(results_dir, model_dir_cache_file_name="cache",
                                       invalid_cache=False):
  dataset_to_model_results = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

  model_dirs = tf.io.gfile.listdir(results_dir)
  for model_dir in tqdm(model_dirs, desc="loading model results..."):
    model_dir_path = os.path.join(results_dir, model_dir)
    key = parse_model_dir_name(model_dir)
    model_result = load_model_dir_result_with_cache(
      model_dir_path=model_dir_path,
      cache_file_name=model_dir_cache_file_name,
      invalid_cache=invalid_cache,
    )
    for eval_type, eval_dict in model_result.items():
      for dataset, array_dict in eval_dict.items():
        model_type, _, is_deterministic, tuning_domain, num_mc_samples = key
        k = {
          'single': 1,
          'ensemble': 3
        }[eval_type.strip('/')]
        updated_key = (model_type, k, is_deterministic, tuning_domain, num_mc_samples)
        assert updated_key not in dataset_to_model_results[dataset], f"already have keys " \
                                                                     f"{dataset_to_model_results[dataset].keys()}"
        dataset_to_model_results[dataset][updated_key] = array_dict
  return dataset_to_model_results
