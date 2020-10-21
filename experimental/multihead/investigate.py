import os.path
import json

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20_multihead import create_model as resnet20_multihead
from func import one_vs_all_loss_fn
import utils #from baselines/cifar

flags.DEFINE_integer('seed', 1337, 'Random seed.')
flags.DEFINE_string('model_file', 'dir0/model.ckpt', 'Model file.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Pick dataset.')

flags.DEFINE_enum('certainty_variant','total',
                    enum_values=['total', 'partial','normalized'],
                    help='Pick variant of certainty measure')

flags.DEFINE_integer('batch_size', 128, 'The training batch size.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

FLAGS = flags.FLAGS

def load_datasets():
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

def main(argv):
    del argv
    logging.info('Multihead CIFAR-10 ResNet-20 investigation!')
    if not tf.io.gfile.exists(FLAGS.model_file+'.index'):
        raise ValueError(f'No model found in {FLAGS.model_file}')
        
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset, test_datasets = load_datasets()
    logging.info(f'Loaded clean test_datasets...')
    logging.info(f'Loaded corrupted test_datasets...{len(test_datasets)-1}')
    
    model = resnet20_multihead(batch_size=FLAGS.batch_size,
                               l2_weight=None,
                               certainty_variant=FLAGS.certainty_variant)
    load = model.load_weights(FLAGS.model_file).expect_partial()
    logging.info(f'Loaded model...{FLAGS.model_file}')
    
    metrics = {'acc':tf.keras.metrics.SparseCategoricalAccuracy(),
               'ece_probs':um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece'),
               'ece_certs':um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece')}
    
    results = {}
    for key in test_datasets.keys():
        #logging.info(key)
        for data in test_datasets[key]:
            x,y = data
            out = model(x)
            logits = out['logits']
            probs = out['probs']
            certs = out['certs']
            
            metrics['acc'].update_state(y,logits)
            metrics['ece_probs'].update_state(y,probs)
            metrics['ece_certs'].update_state(y,certs)

        results[key] = [metric.result().numpy().item() for metric in metrics.values()]
        
        logging.info(f'{key} : {results[key]}')
        for metric in metrics.values():
            metric.reset_states()
    
    investigate_file = os.path.join(os.path.dirname(FLAGS.model_file),'investigate_cv='+FLAGS.certainty_variant+'.json')
    with open(investigate_file,'w') as fp:
        json.dump(results,fp)
    logging.info(f'Saving results to... {investigate_file}')
        
    ece_probs = []
    ece_certs = []
    for result in results.values():
        ece_probs+=[result[1]]
        ece_certs+=[result[2]]
        
    def quartiles(data):
        return np.percentile(data,25),np.percentile(data,50),np.percentile(data,75)
    
    logging.info(f'ECE(on probs) 25th,50th,75th quartiles: {quartiles(ece_probs)}')
    logging.info(f'ECE(on certs) 25th,50th,75th quartiles: {quartiles(ece_certs)}')
    
# metrics['train/ece'].update_state(labels, probs)
# metrics['train/loss'].update_state(loss)
# metrics['train/negative_log_likelihood'].update_state(
#   negative_log_likelihood)
# metrics['train/accuracy'].update_state(labels, logits)

#     for metric in metrics.values():
#       metric.reset_states()
if __name__ == '__main__':
  app.run(main)