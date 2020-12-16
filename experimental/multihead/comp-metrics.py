import os.path

from absl import app
from absl import flags
from absl import logging

import pandas as pd
import numpy as np

import itertools
import json

import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

from resnet20 import create_model as resnet20
from resnet20 import configure_model as resnet20_configure
from resnet20 import load_model as resnet20_load

from func import load_datasets_basic, load_datasets_corrupted, add_dataset_flags
from func import load_datasets_OOD

from func import AttrDict, load_flags, save_flags

from metrics import BrierScore
from metrics import MMC
from metrics import nll

def prepare(FLAGS):

    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    callbacks, optimizer, loss_funcs, metrics = resnet20_configure(FLAGS)

    model = resnet20(batch_size=FLAGS.batch_size,
                                l2_weight=FLAGS.weight_decay,
                                certainty_variant=FLAGS.certainty_variant,
                                activation_type=FLAGS.activation,
                                model_variant=FLAGS.model_variant,
                                logit_variant=FLAGS.logit_variant
                              )

    model.compile(optimizer=optimizer,
                  loss=loss_funcs,
                  metrics=metrics)


    resnet20_load(model,FLAGS)
    
    return model

def set_models(certainty_variant='partial'):
    FLAGS = load_flags('FLAGS.json')
    models = {}
    
    FLAGS.output_dir = '1_vanilla_relu'
    FLAGS.model_file = '1_vanilla_relu/model.ckpt-250'

    FLAGS.activation = 'relu'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = 'vanilla'
    FLAGS.logit_variant = 'affine'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_vanilla_sin'
    FLAGS.model_file = '1_vanilla_sin/model.ckpt-250'

    FLAGS.activation = 'sin'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = 'vanilla'
    FLAGS.logit_variant = 'affine'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_vanilla_dm_relu'
    FLAGS.model_file = '1_vanilla_dm_relu/model.ckpt-250'

    FLAGS.activation = 'relu'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = 'vanilla'
    FLAGS.logit_variant = 'dm'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_vanilla_dm_sin'
    FLAGS.model_file = '1_vanilla_dm_sin/model.ckpt-250'

    FLAGS.activation = 'sin'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = 'vanilla'
    FLAGS.logit_variant = 'dm'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_1vsall_relu'
    FLAGS.model_file = '1_1vsall_relu/model.ckpt-250'

    FLAGS.activation = 'relu'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = '1vsall'
    FLAGS.logit_variant = 'affine'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_1vsall_sin'
    FLAGS.model_file = '1_1vsall_sin/model.ckpt-250'

    FLAGS.activation = 'sin'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = '1vsall'
    FLAGS.logit_variant = 'affine'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_1vsall_dm_relu'
    FLAGS.model_file = '1_1vsall_dm_relu/model.ckpt-250'

    FLAGS.activation = 'relu'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = '1vsall'
    FLAGS.logit_variant = 'dm'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)



    FLAGS.output_dir = '1_1vsall_dm_sin'
    FLAGS.model_file = '1_1vsall_dm_sin/model.ckpt-250'

    FLAGS.activation = 'sin'
    FLAGS.certainty_variant = certainty_variant
    FLAGS.model_variant = '1vsall'
    FLAGS.logit_variant = 'dm'

    FLAGS.eval_frequency = 10

    models[FLAGS.output_dir] = prepare(FLAGS)
    
    return FLAGS, models

#main loop
cvs = ['total','normalized']
#cvs = ['partial','total','normalized']

for certainty_variant in cvs:

    FLAGS, models = set_models(certainty_variant=certainty_variant)

    dataset_builder,train_dataset,val_dataset,test_dataset = load_datasets_basic(FLAGS)
    _, test_datasets_corrupt = load_datasets_corrupted(FLAGS)
    ood_datasets = load_datasets_OOD(FLAGS)
    FLAGS = add_dataset_flags(dataset_builder,FLAGS)
    
    df = pd.DataFrame(columns=['dataset','shift_type','metric']+list(models.keys()))
    
    
    #dataset shift or cifar-10-c

    datasets = test_datasets_corrupt
    #datasets = {'clean':test_datasets_corrupt['clean']}

    for shift_type,dset in datasets.items():
        record = {}
        print('shift_type =',shift_type)
        record['dataset'] = 'cifar10'
        record['shift_type'] = shift_type
        for model_name,model in models.items():
            print('model =',model_name)
            metrics_vals = model.evaluate(dset)
            metrics_names = model.metrics_names

            for metric,metric_val in zip(metrics_names,metrics_vals):

                record['metric'] = metric
                record[model_name] = metric_val

                # save record
                mask_dataset = df['dataset'] == record['dataset']
                mask_shift = df['shift_type'] == record['shift_type']
                mask_metric = df['metric'] == record['metric']
                row_ix = df[(mask_dataset) & (mask_shift) & (mask_metric)].index
    #             print('row_ix=',row_ix)
                if len(row_ix)==0:
    #                 print('new df entry')
                    df = df.append(record,ignore_index=True)
                elif len(row_ix)==1:
                    df_rec = df.at[row_ix[0],model_name]
                    if np.isnan(df_rec):
    #                     print('new record')
                        df.at[row_ix[0],model_name] = metric_val
                    else:
    #                     print('record exists, appending')
                        df.at[row_ix[0],model_name] = metric_val            
                else:
                    print('multiple records, somethings wrong')

                
    df['metric'] = df['metric'].apply(lambda x: certainty_variant+'_'+ x if 'certs' in x else x)
    df.to_csv('summary/cifar10c_results_cert_'+certainty_variant+'.csv')