#import os.path

#from absl import app
#from absl import flags
#from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um
import tensorflow_datasets as tfds

import utils #from baselines/cifar


def one_vs_all_loss_fn(dm_alpha: float = 1., from_logits: bool = True):
    """Requires from_logits=True to calculate correctly."""
    if not from_logits:
        raise ValueError('One-vs-all loss requires inputs to the '
                         'loss function to be logits, not probabilities.')

    def one_vs_all_loss(labels: tf.Tensor, logits: tf.Tensor):
        r"""Implements the one-vs-all loss function.

        As implemented in https://arxiv.org/abs/1709.08716, multiplies the output
        logits by dm_alpha (if using a distance-based formulation) before taking K
        independent sigmoid operations of each class logit, and then calculating the
        sum of the log-loss across classes. The loss function is calculated from the
        K sigmoided logits as follows -

        \mathcal{L} = \sum_{i=1}^{K} -\mathbb{I}(y = i) \log p(\hat{y}^{(i)} | x)
        -\mathbb{I} (y \neq i) \log (1 - p(\hat{y}^{(i)} | x))

        Args:
          labels: Integer Tensor of dense labels, shape [batch_size].
          logits: Tensor of shape [batch_size, num_classes].

        Returns:
          A scalar containing the mean over the batch for one-vs-all loss.
        """
        eps = 1e-6
        logits = logits * dm_alpha
        n_classes = tf.cast(logits.shape[1], tf.float32)

        one_vs_all_probs = tf.math.sigmoid(logits)
        labels = tf.cast(tf.squeeze(labels), tf.int32)
        row_ids = tf.range(tf.shape(one_vs_all_probs)[0], dtype=tf.int32)
        idx = tf.stack([row_ids, labels], axis=1)

        # Shape of class_probs is [batch_size,].
        class_probs = tf.gather_nd(one_vs_all_probs, idx)

        loss = (
            tf.reduce_mean(tf.math.log(class_probs + eps)) +
            n_classes * tf.reduce_mean(tf.math.log(1. - one_vs_all_probs + eps)) -
            tf.reduce_mean(tf.math.log(1. - class_probs + eps)))

        return -loss

    return one_vs_all_loss

def load_datasets_basic(FLAGS):
    strategy = ub.strategy_utils.get_strategy(None, False)
    
    dataset_builder = ub.datasets.Cifar10Dataset(batch_size=FLAGS.batch_size,
                                                 eval_batch_size=FLAGS.eval_batch_size,
                                                 validation_percent=FLAGS.validation_percent)
    
    train_dataset = ub.utils.build_dataset(dataset_builder, 
                                           strategy, 
                                           'train', 
                                           as_tuple=True)
    val_dataset = ub.utils.build_dataset(dataset_builder, 
                                         strategy, 
                                         'validation', 
                                         as_tuple=True)
    test_dataset = ub.utils.build_dataset(dataset_builder, 
                                          strategy, 
                                          'test', 
                                          as_tuple=True)    
    
    return dataset_builder,train_dataset,val_dataset,test_dataset

def load_datasets_corrupted(FLAGS):
    train_dataset = utils.load_input_fn(split=tfds.Split.TRAIN,
                                         name=FLAGS.dataset,
                                         batch_size=FLAGS.batch_size,
                                         use_bfloat16=False)()
    test_datasets = {'clean': utils.load_input_fn(split=tfds.Split.TEST,
                                                  name=FLAGS.dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  use_bfloat16=False)()
                    }
    
    #load corrupted/modified cifar10 datasets
    load_c_input_fn = utils.load_cifar10_c_input_fn
    corruption_types, max_intensity = utils.load_corrupted_test_info(FLAGS.dataset)
    for corruption in corruption_types:
        for intensity in range(1, max_intensity + 1):
            input_fn = load_c_input_fn(corruption_name=corruption,
                                       corruption_intensity=intensity,
                                       batch_size=FLAGS.batch_size,
                                       use_bfloat16=False)
            test_datasets['{0}_{1}'.format(corruption, intensity)] = input_fn()
    return train_dataset, test_datasets

def add_dataset_flags(dataset_builder,FLAGS):
    FLAGS.steps_per_epoch = dataset_builder.info['num_train_examples'] // FLAGS.batch_size
    FLAGS.validation_steps = dataset_builder.info['num_validation_examples'] // FLAGS.eval_batch_size
    FLAGS.test_steps = dataset_builder.info['num_test_examples'] // FLAGS.eval_batch_size
    FLAGS.no_classes = 10 # awful but no way to infer from dataset...
    
    return FLAGS

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self