import os.path

from absl import app
from absl import flags
from absl import logging

#import pandas as pd
import numpy as np

import itertools
import json

import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20 import create_model as resnet20
from resnet20 import configure_model as resnet20_configure

from func import load_datasets_basic, load_datasets_corrupted, add_dataset_flags
from func import AttrDict, load_flags, save_flags


FLAGS = load_flags('FLAGS.json')

FLAGS.output_dir = '3_1_1vsall_dm_relu'
FLAGS.model_file = '3_1_1vsall_dm_relu/model.ckpt-250'

FLAGS.activation = 'relu'
FLAGS.certainty_variant = 'partial'
FLAGS.model_variant = '1vsall'
FLAGS.logit_variant = 'dm'

FLAGS.eval_frequency = 10
FLAGS.epochs= 300

logging.info('Multihead CIFAR-10 ResNet-20 train')

tf.random.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

callbacks, optimizer, loss_funcs, metrics = resnet20_configure(FLAGS)

model = resnet20(batch_size=FLAGS.batch_size,
                           l2_weight=FLAGS.weight_decay,
                           activation_type=FLAGS.activation, 
                           certainty_variant=FLAGS.certainty_variant, 
                           model_variant=FLAGS.model_variant,
                           logit_variant=FLAGS.logit_variant
                          )

model.compile(optimizer=optimizer,
              loss=loss_funcs,
              metrics=metrics)

dataset_builder,train_dataset,val_dataset,test_dataset = load_datasets_basic(FLAGS)
FLAGS = add_dataset_flags(dataset_builder,FLAGS)

history = model.fit(train_dataset,
                    #batch_size=FLAGS.batch_size,
                    epochs=FLAGS.epochs,
                    steps_per_epoch=FLAGS.steps_per_epoch,
                    validation_data=val_dataset,
                    validation_steps=FLAGS.validation_steps,
                    validation_freq=FLAGS.eval_frequency,
                    callbacks=callbacks,
                    shuffle=False)


# model_dir = os.path.join(FLAGS.output_dir, 'model.ckpt-{}'.format(FLAGS.epochs))
# logging.info('Saving model to '+model_dir)
# model.save_weights(model_dir)
# resnet20_save(model,FLAGS)