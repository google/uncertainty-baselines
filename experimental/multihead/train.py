import os.path

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20_multihead import create_model as resnet20_multihead
from func import one_vs_all_loss_fn
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
FLAGS = flags.FLAGS

def load_datasets():
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


def main(argv):
    del argv
    logging.info('Multihead CIFAR-10 ResNet-20 experiment!')
    logging.info('Saving to dir: %s', FLAGS.output_dir)
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
    
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(FLAGS.output_dir,'logs'))
    
    dataset_builder,train_dataset,val_dataset,test_dataset = load_datasets()
    
    optimizer = ub.optimizers.get(optimizer_name=FLAGS.optimizer,
                                  learning_rate_schedule='constant',
                                  learning_rate=FLAGS.learning_rate,
                                  weight_decay=FLAGS.weight_decay)
    
    model = resnet20_multihead(batch_size=FLAGS.batch_size,
                               l2_weight=None)
    
#     loss_funcs = {'logits':one_vs_all_loss_fn(from_logits=True),
#                   'probs':None,
#                   'certs':None}
    loss_funcs = {'logits':one_vs_all_loss_fn(from_logits=True)}
    
    metrics = {'logits':[tf.keras.metrics.SparseCategoricalAccuracy()],
               'probs':[um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece')],
               'certs':[um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece')]}
    
    model.compile(optimizer=optimizer,
                  loss=loss_funcs,
                  metrics=metrics)

    steps_per_epoch = dataset_builder.info['num_train_examples'] // FLAGS.batch_size
    validation_steps = dataset_builder.info['num_validation_examples'] // FLAGS.eval_batch_size
    test_steps = dataset_builder.info['num_test_examples'] // FLAGS.eval_batch_size    
    
    history = model.fit(x=train_dataset,
                        batch_size=FLAGS.batch_size,
                        epochs=FLAGS.epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_dataset,
                        validation_steps=validation_steps,
                        validation_freq=FLAGS.eval_frequency,
                        callbacks=[tensorboard_callback],
                        shuffle=False)
    
    #logging.info(history)
    model_dir = os.path.join(FLAGS.output_dir, 'model.ckpt-{}'.format(FLAGS.epochs))
    logging.info('Saving model to '+model_dir)
    model.save_weights(model_dir)
    
if __name__ == '__main__':
  app.run(main)