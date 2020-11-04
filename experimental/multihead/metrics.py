import os.path
import json

from absl import app
from absl import flags
from absl import logging

import numpy as np

import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um
import tensorflow_datasets as tfds

import utils #from baselines/cifar

from tqdm.notebook import tqdm

# class summation(tf.keras.metrics.Mean):
#     def update_state(self, y_true, y_pred, sample_weight=None): 
#         #brier_score = um.brier_score(labels=y_true, probabilities=y_pred)
#         super(summation, self).update_state(y_pred)
     
class BrierScore(tf.keras.metrics.Mean):
    def update_state(self, y_true, y_pred, sample_weight=None):
        brier_score = um.brier_score(labels=y_true, probabilities=y_pred)
        super(BrierScore, self).update_state(brier_score)

        
def rel_diag(model,
             dataset,
             FLAGS,
             prob_output='certs',
             savefig=False,
            ):
    
    #number of classes
    K = FLAGS.no_classes
    
    labels = np.empty(0)
    probs = np.empty((0,K))
    
    for i,(x,y) in enumerate(dataset):
        if i>FLAGS.steps_per_epoch: break
            
        out = model(x)[prob_output]
        if type(out) is tf.python.framework.ops.EagerTensor:
            out = out.numpy()
            
        labels = np.append(labels,y.numpy().astype('int32'))
        probs = np.concatenate((probs,out))
    #     logits = out['logits']
    #     probs = out['probs']
    #     certs = out['certs']
    #     print(logits,probs,certs)
    
    diagram = um.numpy.reliability_diagram(probs=probs,labels=labels.astype('int32'),img=False)
    if savefig:
        diagram_file = os.path.join(os.path.dirname(FLAGS.model_file),'rel_diagram_cv='+FLAGS.certainty_variant+'.jpg')
        diagram.savefig(diagram_file)

# general func
def comp_metrics(model,
                 datasets,
                 metrics,
                 FLAGS,
                 prob_output='certs',
                 save_summary=False,
                 return_quartiles=False,
                ):
    results={}
    for ds_key,ds in tqdm(datasets.items()):
        #logging.info(key)
        for i,(x,y) in enumerate(ds):
            if i>FLAGS.test_steps: break

            out = model(x)[prob_output]
            if type(out) is tf.python.framework.ops.EagerTensor:
                out = out.numpy()
            
            for key,metric in metrics.items():
                metric.update_state(y,out)
    #         metrics['ece'].update_state(y,probs)

        results[ds_key] = [metric.result().numpy().item() for metric in metrics.values()]

        logging.info(f'{key} : {results[ds_key]}')
        for metric in metrics.values():
            metric.reset_states()
            
    if save_summary:
        summary_file = os.path.join(os.path.dirname(FLAGS.model_file),'ece_summary_cv='+FLAGS.certainty_variant+'.json')
        with open(summary_file,'w') as fp:
            json.dump(results,fp)
        logging.info(f'Saving results to... {summary_file}')

    
    qs = quartiles(list(results.values())) if return_quartiles else None
        
    return results, qs
        
# special case of comp_metrics
def ece(model,
        dataset,
        FLAGS,
        prob_output='certs'):
    
    metric = um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece')
    
    results,_ = comp_metrics(model=model,
                           datasets={'ds':dataset},
                           metrics={'metric':metric},
                           FLAGS=FLAGS,
                           prob_output=prob_output
                          )

    return list(results.values())[0][0]


def quartiles(data):
    return np.percentile(data,25),np.percentile(data,50),np.percentile(data,75)


