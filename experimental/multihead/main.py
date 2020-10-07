import os.path

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20_multihead import create_model as resnet20_multihead

class BrierScore(tf.keras.metrics.Mean):

  def update_state(self, y_true, y_pred, sample_weight=None):
    brier_score = um.brier_score(labels=y_true, probabilities=y_pred)
    super(BrierScore, self).update_state(brier_score)

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



flags.DEFINE_string('output_dir', 'testdir', 'Base output directory.')
flags.DEFINE_integer('seed', 1337, 'Random seed.')

flags.DEFINE_integer('batch_size', 512, 'The training batch size.')
flags.DEFINE_integer('eval_batch_size', 100, 'The evaluation batch size.')

flags.DEFINE_float('validation_percent', 0.1, 'Validation set percentage.')

flags.DEFINE_string('optimizer', 'adam', 'The optimizer to train with.')
flags.DEFINE_float('learning_rate', 0.01, 'The learning rate.')
flags.DEFINE_float('weight_decay',None,'The model decoupled weight decay rate.')

flags.DEFINE_integer('eval_frequency',100,'How many steps between evaluating on the (validation and) test set.')
flags.DEFINE_integer('epochs', 20, 'How many epochs to train for.')

FLAGS = flags.FLAGS

def run(trial_dir: str):
    """Run the experiment.

    Args:
    trial_dir: String to the dir to write checkpoints to and read them from.
    flag_string: Optional string used to record what flags the job was run with.
    """
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    strategy = ub.strategy_utils.get_strategy(None, False)

    with strategy.scope():
        dataset_builder = ub.datasets.Cifar10Dataset(batch_size=FLAGS.batch_size,
                                                     eval_batch_size=FLAGS.eval_batch_size,
                                                     validation_percent=FLAGS.validation_percent)  # Use 5000 validation images.
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

        optimizer = ub.optimizers.get(optimizer_name=FLAGS.optimizer,
                                      learning_rate_schedule='constant',
                                      learning_rate=FLAGS.learning_rate,
                                      weight_decay=FLAGS.weight_decay)

        # Setup model.
#         model = ub.models.ResNet20Builder(batch_size=FLAGS.batch_size, 
#                                           l2_weight=None)
        
#         metrics = {'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
#                    #'brier_score': BrierScore(name='brier_score'),
#                    'ece': um.ExpectedCalibrationError(num_bins=10, name='ece')
#                   }
#         model.compile(optimizer=optimizer,
#               #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               loss=one_vs_all_loss_fn(from_logits=True),
#               metrics=metrics.values())
        model = resnet20_multihead(batch_size=FLAGS.batch_size, 
                                          l2_weight=None)
        model.compile(optimizer=optimizer,
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss={'logits':one_vs_all_loss_fn(from_logits=True),'uncerts':None},
              metrics={'logits':tf.keras.metrics.SparseCategoricalAccuracy(),
                       'uncerts':[um.ExpectedCalibrationError(num_bins=10, name='ece')
                                  #BrierScore(name='brier_score')
                                 ]})
    
        # Train and eval.
        steps_per_epoch = (
            dataset_builder.info['num_train_examples'] // FLAGS.batch_size)
        validation_steps = (
            dataset_builder.info['num_validation_examples'] // FLAGS.eval_batch_size)

        history = model.fit(x=train_dataset,
                            batch_size=FLAGS.batch_size,
                            epochs=FLAGS.epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset,
                            validation_steps=validation_steps,
                            validation_freq=FLAGS.eval_frequency,
                            shuffle=False)
        logging.info(history)

        test_steps = dataset_builder.info['num_test_examples'] // FLAGS.eval_batch_size
        test_result = model.evaluate(x=test_dataset,
                                     batch_size=FLAGS.eval_batch_size,
                                     steps=test_steps)
        logging.info(test_result)

    if trial_dir:
      model.save_weights(
          os.path.join(trial_dir, 'model.ckpt-{}'.format(FLAGS.train_steps)))

def main(argv):
  del argv
  logging.info('Multihead CIFAR-10 ResNet-20 experiment!')
  trial_dir = os.path.join(FLAGS.output_dir, '0')
  logging.info('Saving to dir: %s', trial_dir)
  if not tf.io.gfile.exists(trial_dir):
    tf.io.gfile.makedirs(trial_dir)
  return run(trial_dir)


if __name__ == '__main__':
  app.run(main)
    