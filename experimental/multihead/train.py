import os.path

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20_multihead import create_model as resnet20_multihead
from resnet20_multihead import train_model as resnet20_multihead_train

from func import load_datasets_basic, load_datasets_corrupted
import utils

flags.DEFINE_integer('seed', 1337, 'Random seed.')
flags.DEFINE_string('output_dir', 'testdir', 'Base output directory.')

flags.DEFINE_string('optimizer', 'adam', 'The optimizer to train with.')
flags.DEFINE_float('learning_rate', 0.01, 'The learning rate.')
flags.DEFINE_float('weight_decay',None,'The model decoupled weight decay rate.')

flags.DEFINE_integer('batch_size', 128, 'The training batch size.')
flags.DEFINE_integer('eval_batch_size', 128, 'The evaluation batch size.')
flags.DEFINE_float('validation_percent', 0.1, 'Validation set percentage.')
flags.DEFINE_integer('eval_frequency',100,
                     'How many steps between evaluating on the (validation and) test set.')

flags.DEFINE_integer('num_bins',15,'How many bins in ECE')

flags.DEFINE_integer('epochs', 50, 'How many epochs to train for.')

flags.DEFINE_enum('activation','relu',
                  enum_values=['relu', 'sin'],
                  help='Pick activation type.')
FLAGS = flags.FLAGS


def main(argv):
    del argv
    logging.info('Multihead CIFAR-10 ResNet-20 train')

    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(FLAGS.output_dir,'logs'))

    dataset_builder,train_dataset,val_dataset,test_dataset = load_datasets_basic(FLAGS)

    optimizer = ub.optimizers.get(optimizer_name=FLAGS.optimizer,
                                  learning_rate_schedule='constant',
                                  learning_rate=FLAGS.learning_rate,
                                  weight_decay=FLAGS.weight_decay)

    model = resnet20_multihead(batch_size=FLAGS.batch_size,
                               l2_weight=None,
                               activation_type=FLAGS.activation)

    FLAGS.steps_per_epoch = dataset_builder.info['num_train_examples'] // FLAGS.batch_size
    FLAGS.validation_steps = dataset_builder.info['num_validation_examples'] // FLAGS.eval_batch_size
    FLAGS.test_steps = dataset_builder.info['num_test_examples'] // FLAGS.eval_batch_size    


    resnet20_multihead_train(model=model,
                            train_dataset=train_dataset,
                            val_dataset = val_dataset,
                            optimizer = optimizer,
                            FLAGS = FLAGS,
                            tensorboard_callback = tensorboard_callback
                           )

if __name__ == '__main__':
  app.run(main)