# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""TF Keras definition for Resnet-20 for CIFAR."""

import os.path
from absl import app
from absl import flags
from absl import logging
from typing import Any, Dict

import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um

import numpy as np
import sklearn.isotonic

from func import one_vs_all_loss_fn




def _activ(activation_type):
    activation = {'relu': tf.keras.layers.ReLU(), 'sin': tf.keras.backend.sin}
    return activation[activation_type]

def _resnet_layer(
    inputs: tf.Tensor,
    num_filters: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    use_activation: bool = True,
    activation_type: str = 'relu', #relu or sin!
    use_norm: bool = True,
    l2_weight: float = 1e-4) -> tf.Tensor:
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: input tensor from input image or previous layer.
    num_filters: Conv2D number of filters.
    kernel_size: Conv2D square kernel dimensions.
    strides: Conv2D square stride dimensions.
    use_activation: whether or not to use a non-linearity.
    use_norm: whether to include normalization.
    l2_weight: the L2 regularization coefficient to use for the convolution
      kernel regularizer.

  Returns:
      Tensor output of this layer.
  """
  kernel_regularizer = None
  if l2_weight:
    kernel_regularizer = tf.keras.regularizers.l2(l2_weight)
  conv_layer = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=kernel_regularizer)

  x = conv_layer(inputs)
  x = tf.keras.layers.BatchNormalization()(x) if use_norm else x
  
  x = _activ(activation_type)(x) if use_activation is not None else x
  return x


def create_model(
    batch_size: int,
    l2_weight: float = 0.0,
    certainty_variant: str = 'partial', # total, partial or normalized
    activation_type: str = 'relu', #relu or sine
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Resnet-20 v1, takes (32, 32, 3) input and returns logits of shape (10,)."""
  # TODO(znado): support NCHW data format.
  input_layer = tf.keras.layers.Input(
      shape=(32, 32, 3), batch_size=batch_size)
  depth = 20
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  x = _resnet_layer(
      inputs=input_layer,
      num_filters=num_filters,
      l2_weight=l2_weight,
      activation_type=activation_type)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:
        strides = 2
      y = _resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          l2_weight=l2_weight,
          activation_type=activation_type)
      y = _resnet_layer(
          inputs=y,
          num_filters=num_filters,
          use_activation=False,
          l2_weight=l2_weight)
      if stack > 0 and res_block == 0:
        x = _resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            use_activation=False,
            use_norm=False,
            l2_weight=l2_weight)
      x = tf.keras.layers.add([x, y])
      x = _activ(activation_type)(x)
    num_filters *= 2

  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  logits = tf.keras.layers.Dense(10, kernel_initializer='he_normal')(x)
    
  #form Ci's
  probs = tf.math.sigmoid(logits)
  probs_comp = 1-probs
  K = probs.shape[1]
  cert_list = []
  for i in range(K):
    proj_vec = np.zeros(K)
    proj_vec[i]=1
    proj_mat = np.outer(proj_vec,proj_vec)
    proj_mat_comp = np.identity(K)-np.outer(proj_vec,proj_vec)
    tproj_mat = tf.constant(proj_mat,dtype=tf.float32)
    tproj_mat_comp = tf.constant(proj_mat_comp,dtype=tf.float32)
    out = tf.tensordot(probs,tproj_mat,axes=1) + tf.tensordot(probs_comp,tproj_mat_comp,axes=1)
    cert_list+=[tf.reduce_prod(out,axis=1)]
    
  if certainty_variant == 'partial':
    certs = tf.stack(cert_list,axis=1)
    
  elif certainty_variant == 'total':
    certs = tf.stack(cert_list,axis=1)
    certs_argmax = tf.one_hot(tf.argmax(certs,axis=1),depth=K)
    certs_reduce = tf.tile(tf.reduce_sum(certs,axis=1,keepdims=True),[1,K])
    certs = tf.math.multiply(certs_argmax,certs_reduce)
    
  elif certainty_variant == 'normalized':
    certs = tf.stack(cert_list,axis=1)
    certs_norm = tf.tile(tf.reduce_sum(certs,axis=1,keepdims=True),[1,K])
    certs = tf.math.divide(certs,certs_norm)
    
  else:
    raise ValueError('unknown certainty_variant')

  #logits_from_certs
  eps = 1e-6
  logcerts = tf.math.log(certs+eps)
  rs = tf.tile(logcerts[:,:1],[1,K])-logcerts #set first logit to zero (an arbitrary choice)
  logits_from_certs = -rs

  return tf.keras.models.Model(
      inputs=input_layer, 
      outputs={'logits':logits,'probs':probs,'certs':certs,'logits_from_certs':logits_from_certs}, 
      name='resnet20-multihead')



def train_model(model,
                train_dataset,
                val_dataset,
                optimizer,
                FLAGS,
                tensorboard_callback,
                ):
    
    logging.info('Saving to dir: %s', FLAGS.output_dir)
    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
        
    loss_funcs = {'logits':one_vs_all_loss_fn(from_logits=True),
                  'probs':None,
                  'certs':None,
                  'logits_from_certs':None}
#     loss_funcs = {'logits':one_vs_all_loss_fn(from_logits=True)}
    
    metrics = {'logits':[tf.keras.metrics.SparseCategoricalAccuracy()],
               'probs':[um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece')],
               'certs':[um.ExpectedCalibrationError(num_bins=FLAGS.num_bins, name='ece')],
               'logits_from_certs':[tf.keras.metrics.SparseCategoricalAccuracy()]}
    
    model.compile(optimizer=optimizer,
                  loss=loss_funcs,
                  metrics=metrics)
    
    history = model.fit(x=train_dataset,
                        batch_size=FLAGS.batch_size,
                        epochs=FLAGS.epochs,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        validation_data=val_dataset,
                        validation_steps=FLAGS.validation_steps,
                        validation_freq=FLAGS.eval_frequency,
                        callbacks=[tensorboard_callback],
                        shuffle=False)
    
    #logging.info(history)
    model_dir = os.path.join(FLAGS.output_dir, 'model.ckpt-{}'.format(FLAGS.epochs))
    logging.info('Saving model to '+model_dir)
    model.save_weights(model_dir)

def load_model(model,
               FLAGS
              ):
    
    load = model.load_weights(FLAGS.model_file).expect_partial()
    logging.info(f'Loaded model...{FLAGS.model_file}')
    

# temperature calibration
def calibrate_model_temp(model,
                    dataset,
                    FLAGS,
                    logit_output='logits', #or 'logits_from_certs'
                    epochs=20,
                    preset_temp=None,
                    ): 
    
    K = FLAGS.no_classes
    
    model_score = tf.keras.models.Model(model.input, model.output[logit_output])
    
    if preset_temp is not None:
        #model.output['probs_cal'] = cal(model.output[logit_output],preset_temp)
        return cal(model_score,preset_temp), preset_temp
    
    T0=3
    T = tf.Variable(T0*tf.ones(shape=(1,)))
    history = []
    optimizer = ub.optimizers.get(optimizer_name='adam',
                                  learning_rate_schedule='constant',
                                  learning_rate=0.01,
                                  weight_decay=None)
    def cost(T, x, y):

        scaled_logits = tf.multiply(x=x, y=1.0 / T)

        cost_value = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits, labels=y)
        )

        return cost_value

    def grad(T, x, y):

        with tf.GradientTape() as tape:
            cost_value = cost(T, x, y)

        return cost_value, tape.gradient(cost_value, T)

    for epoch in range(epochs):
        for i,(x,y) in enumerate(dataset):
            if i>FLAGS.steps_per_epoch: break
                
            X_train = model_score.predict(x)
            y_train = tf.one_hot(y,depth=K)
            train_cost, grads = grad(T, X_train, y_train)
#             print(grads.shape,T.shape)
            optimizer.apply_gradients(zip([grads], [T]))

        history.append([train_cost, T.numpy()[0]])
        print(epoch,history[-1])

    temperature = history[-1][1]
    logging.info(f'Calibration temp={temperature}')

    
    def cal(tensor,temp):
        return tf.nn.softmax(tensor/temp,axis=-1)
    
    cal_model = tf.keras.models.Model(inputs=model.input, 
                                      outputs={logit_output+'_cal': cal(model.output[logit_output],temperature)})

    return cal_model,temperature

#linear, does not work
# def calibrate_model_linear(model,
#                     dataset,
#                     FLAGS,
#                     prob_output='certs', #or 'probs'
#                     epochs=30,
#                     ): 
    
#     #number of classes
#     K = FLAGS.no_classes
    
#     model_score = tf.keras.models.Model(model.input, model.output[prob_output])
    
#     #W0=0.1
#     #W = tf.Variable(W0*tf.ones(shape=(1,K)))
#     W = tf.Variable(tf.random.normal(shape=(1,K)))
#     history = []
#     optimizer = ub.optimizers.get(optimizer_name='adam',
#                                   learning_rate_schedule='constant',
#                                   learning_rate=0.01,
#                                   weight_decay=None)
#     def cost(W, x, y):

#         #certs_W = tf.linalg.matvec(x,W)
#         W_tile = tf.tile(W,[x.shape[0],1])
#         probs_W = x*W_tile
        
#         cc_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
#                                                        label_smoothing=0,
#                                                        name='categorical_crossentropy')
#         cost_value = cc_loss(y,probs_W)

#         return cost_value

#     def grad(W, x, y):

#         with tf.GradientTape() as tape:
#             cost_value = cost(W, x, y)

#         return cost_value, tape.gradient(cost_value, W)


#     for epoch in range(epochs):
#         for i,(x,y) in enumerate(dataset):
#             if i>FLAGS.steps_per_epoch: break
                
#             X_train = model_score.predict(x)
#             y = tf.cast(y,dtype=tf.int32)
#             y_train = tf.one_hot(y,depth=K)
#             train_cost, grads = grad(W, X_train, y_train)
#             optimizer.apply_gradients(zip(grads[0], W[0]))
        
#         history.append([train_cost, W.numpy()[0]])
#         print(epoch,history[-1])

#     W0 = history[-1][1]
#     logging.info(f'Calibration W={W0}')

#     def cal(tensor,W0):
#         W0_tile = tf.tile(W0,[tensor.shape[0],1])
#         return tensor*W0_tile
    
#     cal_model = tf.keras.models.Model(inputs=model.input, 
#                                       outputs={prob_output+'_cal': cal(model.output[prob_output],W0)})

#     return cal_model, W0

#isotonic, uses scikit-learn
def calibrate_model_isotonic(model,
                             dataset,
                             FLAGS,
                             prob_output='certs', #or 'probs'
                             ):
    #number of classes
    K = FLAGS.no_classes
    
    labels = np.empty(0)
    probs = np.empty((0,K))
    
    for i,(x,y) in enumerate(dataset):
        if i>FLAGS.steps_per_epoch: break

        out = model(x)[prob_output].numpy()
        labels = np.append(labels,y.numpy().astype('int32'))
        probs = np.concatenate((probs,out))

    confidences, accuracies = _extract_conf_acc(probs=probs,labels=labels.astype('int32'))
    
    isotonic_model = sklearn.isotonic.IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='nan')
    isotonic_model.fit(X=np.array(confidences),y=np.array(accuracies))
    

    def cal_model(x):
        out = tf.keras.models.Model(model.input, model.output[prob_output])(x)
        out_shape = out.shape
        out_calibr = isotonic_model.predict(out.numpy().reshape(-1))
        return {prob_output+'_cal':out_calibr.reshape(out_shape)}
    
    return cal_model, None



# based on um.numpy.plot_diagram, um.numpy.reliability_diagram
def _extract_conf_acc(probs,labels):

    probs = np.array(probs)
    labels = np.array(labels)
    labels_matrix = um.numpy.visualization.one_hot_encode(labels, probs.shape[1])

    # plot_diagram(probs.flatten(), labels_matrix.flatten(), y_axis))

    probs = probs.flatten()
    labels = labels_matrix.flatten()

    probs_labels = [(prob, labels[i]) for i, prob in enumerate(probs)]
    probs_labels = np.array(sorted(probs_labels, key=lambda x: x[0]))
    window_len = int(len(labels)/100.)
    calibration_errors = []
    confidences = []
    accuracies = []
    # More interesting than length of the window (which is specific to this
    # window) is average distance between datapoints. This normalizes by dividing
    # by the window length.
    distances = []
    for i in range(len(probs_labels)-window_len):
        distances.append((
            probs_labels[i+window_len, 0] - probs_labels[i, 0])/float(window_len))
        # It's pretty sketchy to look for the 100 datapoints around this one.
        # They could be anywhere in the probability simplex. This introduces bias.
        mean_confidences = um.numpy.visualization.mean(probs_labels[i:i + window_len, 0])
        confidences.append(mean_confidences)
        class_accuracies = um.numpy.visualization.mean(probs_labels[i:i + window_len, 1])
        accuracies.append(class_accuracies)
        calibration_error = class_accuracies-mean_confidences
        calibration_errors.append(calibration_error)
    return confidences, accuracies