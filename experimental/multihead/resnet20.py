
import os.path
from absl import app
from absl import flags
from absl import logging
from typing import Any, Dict

import tensorflow as tf
import tensorflow.keras as keras

import uncertainty_baselines as ub
import uncertainty_metrics as um

import numpy as np
# import sklearn.isotonic
# import sklearn.neural_network

from metrics import BrierScore
from metrics import MMC
from metrics import nll

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
        #eps = tf.keras.backend.epsilon()
        eps = 1e-6
        #eps = 1e-10
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


def _activ(activation_type: str = 'relu'):
    activation = {'relu': tf.keras.layers.ReLU(), 'sin': tf.keras.backend.sin}
    if activation_type in activation.keys():
        return activation[activation_type]
    else:
        return activation['relu']

class resnetLayer(tf.keras.layers.Layer):
    def __init__(self,
        num_filters: int = 16,
        kernel_size: int = 3,
        strides: int = 1,
        use_activation: bool = True,
        activation_type: str = 'relu', #relu or sin!
        use_norm: bool = True,
        l2_weight: float = 1e-4):
        
        super(resnetLayer,self).__init__()
        
        self.use_activation = use_activation
        self.use_norm = use_norm
        
        self.kernel_regularizer = None
        if l2_weight:
            self.kernel_regularizer = tf.keras.regularizers.l2(l2_weight)    
        
        self.conv_layer = tf.keras.layers.Conv2D(num_filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding='same',
                                                 kernel_initializer='he_normal',
                                                 kernel_regularizer=self.kernel_regularizer)
        
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = _activ(activation_type)
        
    def call(self,
             inputs: tf.Tensor) -> tf.Tensor:

        x = self.conv_layer(inputs)
        if self.use_norm:
            x = self.batch_norm(x)
        if self.use_activation:
            x = self.activation(x)
            
        return x       
    
class resnetBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                stack: int,
                res_block: int,
                num_filters: int = 16,
                activation_type: str = 'relu', #relu or sin!
                l2_weight: float = 1e-4):
        
        super(resnetBlock,self).__init__()
        
        self.stack = stack
        self.res_block = res_block
        self.num_filters = num_filters
        self.activation_type = activation_type
        self.l2_weight = l2_weight
        
        strides = 1
        if self.stack > 0 and self.res_block == 0:
            strides = 2

        self.l_1 = resnetLayer(num_filters=self.num_filters,
                               strides=strides,
                               l2_weight=self.l2_weight,
                               activation_type=self.activation_type)
        
        self.l_2 = resnetLayer(num_filters=self.num_filters,
                        l2_weight=self.l2_weight,
                        use_activation=False)


        self.l_3 = resnetLayer(num_filters=self.num_filters,
                               kernel_size=1,
                               strides=strides,
                               l2_weight=self.l2_weight,
                               use_activation=False,
                               use_norm=False)

        self.l_add = tf.keras.layers.Add()
        self.l_activation = _activ(self.activation_type)
        
    def call(self,inputs: tf.Tensor) -> tf.Tensor:
        y = self.l_1(inputs)
        y = self.l_2(y)
        x = self.l_3(inputs) if self.stack > 0 and self.res_block == 0 else inputs
        x = self.l_add([x, y])
        x = self.l_activation(x)
        return x

class DMLayer(tf.keras.layers.Layer):
  def __init__(self, units: int = 10, **kwargs):
    super(DMLayer, self).__init__(**kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(name='DMLayer_weight',
          shape=(input_shape[-1], self.units),
          initializer="he_normal",
          trainable=True)

  def get_config(self):
    return {"units": self.units}
    
  def call(self, inputs):

    #tf.tile(inputs)
#     w_tiled = tf.tile(tf.reshape(self.w,shape=(1,)+self.w.shape),[inputs.shape[0],1,1])
#     inputs_tiled = tf.tile(tf.reshape(inputs,shape=inputs.shape+(1,)),[1,1,self.units])
#     out = tf.math.sqrt(tf.math.reduce_euclidean_norm(inputs_tiled-w_tiled,axis=1))

#     a = tf.random.normal(shape=(128,64))
#     b = tf.random.normal(shape=(64,10))
    be=tf.expand_dims(self.w,0)
    ae=tf.expand_dims(inputs,-1)
    out = -tf.math.sqrt(tf.math.reduce_euclidean_norm(be-ae,axis=1))
    #out = tf.math.sqrt(tf.math.reduce_euclidean_norm(be-ae,axis=1))
    
    return out


class resnet20(tf.keras.Model):
    def __init__(self,
                 batch_size: int = 128,
                 l2_weight: float = 0.0,
                 activation_type: str = 'relu', #relu or sin
                 certainty_variant: str = 'partial', #partial, total or normalized
                 model_variant: str = '1vsall', #1vsall or vanilla
                 logit_variant: str = 'affine', #affine or dm
                 **params):
        super(resnet20,self).__init__()
        
#         self.batch_size = params['batch_size'] if 'batch_size' in params.keys() else 128
#         self.activation_type = params['activation_type'] if 'activation_type' in params.keys() else 'relu'
#         self.l2_weight = params['l2_weight'] if 'l2_weight' in params.keys() else 0.0
#         self.certainty_variant = params['certainty_variant'] if 'certainty_variant' in params.keys() else 'partial'
#         self.model_variant = params['model_variant'] if 'model_variant' in params.keys() else '1vsall'
        self.batch_size = batch_size
        self.l2_weight = l2_weight
        self.activation_type = activation_type
        
        if certainty_variant in ['partial','total','normalized']:
            self.certainty_variant = certainty_variant
        else:
            raise ValueError('unknown certainty_variant')
            
        self.model_variant = model_variant
        self.logit_variant = logit_variant
        
        self.depth = 20
        self.num_res_blocks = int((self.depth - 2) / 6)
        num_filters = 16
        
        
        self.layer_init_1 = tf.keras.layers.InputLayer(input_shape=(32, 32, 3),
                                                       batch_size=self.batch_size)
        
        self.layer_init_2 = resnetLayer(num_filters=num_filters,
                                        l2_weight=self.l2_weight,
                                        activation_type=self.activation_type)
        
        self.res_blocks = [[0 for stack in range(3)] for res_block in range(self.num_res_blocks)]

        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                self.res_blocks[stack][res_block] = resnetBlock(stack = stack,
                                                                res_block = res_block,
                                                                num_filters = num_filters,
                                                                activation_type = self.activation_type,
                                                                l2_weight = self.l2_weight)
            num_filters *= 2
        
        self.layer_final_1 = tf.keras.layers.AveragePooling2D(pool_size=8)
        self.layer_final_2 = tf.keras.layers.Flatten()
        
        if self.logit_variant == 'dm':
            self.layer_final_3 = DMLayer(units=10)
        elif self.logit_variant == 'affine':
            self.layer_final_3 = tf.keras.layers.Dense(10, kernel_initializer='he_normal')
        else:
            raise ValueError('unknown logit_variant')   

    def _calc_certs(self,
                    probs: tf.Tensor,
                    certainty_variant: str = 'partial') -> tf.Tensor:
        
        #form Ci's
        #probs = tf.math.sigmoid(logits)
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
            
        return certs
    
    def _calc_logits_from_certs(self, 
                                certs: tf.Tensor, 
                                eps: float = 1e-6) -> tf.Tensor:
        #logits_from_certs
        K = certs.shape[1]
        
        logcerts = tf.math.log(certs+eps)
        rs = tf.tile(logcerts[:,:1],[1,K])-logcerts #set first logit to zero (an arbitrary choice)
        logits_from_certs = -rs    
    
        return logits_from_certs
    
#     def load_augmentation(self,idg):
#         self.idg = idg
    
    def call(self, 
             inputs: tf.Tensor, 
             trainable: bool = False) -> dict:

        x = self.layer_init_1(inputs)
        x = self.layer_init_2(x)
        
        for stack in range(3):
            for res_block in range(self.num_res_blocks):
                x = self.res_blocks[stack][res_block](x)

        x = self.layer_final_1(x)
        x = self.layer_final_2(x)
        
        logits = self.layer_final_3(x)
        
        if self.model_variant == '1vsall':
            probs = tf.math.sigmoid(logits)
            if self.logit_variant == 'dm':
               probs = 2*probs
        elif self.model_variant == 'vanilla':
            probs = tf.math.softmax(logits,axis=-1)
        else:
            raise ValueError('unknown model_variant')
        
        certs = self._calc_certs(probs, certainty_variant = self.certainty_variant)
        logits_from_certs = self._calc_logits_from_certs(certs = certs)
        
        return {'logits':logits,'probs':probs,'certs':certs,'logits_from_certs':logits_from_certs}


  
#    def train_step(self, data):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         x, y = data
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             # Compute the loss value
#             # (the loss function is configured in `compile()`)
#             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, y_pred)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}
    

def create_model(batch_size: int,
                 l2_weight: float = 0.0,
                 activation_type: str = 'relu', #relu or sine
                 certainty_variant: str = 'partial', # total, partial or normalized
                 model_variant: str = '1vsall', #1vsall or vanilla
                 logit_variant: str = 'affine', #affine or dm
                 **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
    
    return resnet20(batch_size=batch_size,
                    l2_weight=l2_weight,
                    activation_type=activation_type,
                    certainty_variant=certainty_variant,
                    model_variant=model_variant,
                    logit_variant=logit_variant)

def load_model(model,
               FLAGS,
               verbose=False
              ):
    
    load = model.load_weights(FLAGS.model_file).expect_partial()
    if verbose: logging.info(f'Loaded model...{FLAGS.model_file}')

def save_model(model,
               FLAGS,
               verbose=False
              ):
    
    model_dir = os.path.join(FLAGS.output_dir, 'model.ckpt-{}'.format(FLAGS.epochs))
    if verbose: logging.info('Saving model to '+model_dir)
    model.save_weights(model_dir)

def configure_model(FLAGS):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(FLAGS.output_dir,'logs'))
    #filepath= os.path.join(FLAGS.output_dir,"weights-improvement-{epoch:03d}-{val_accuracy:.2f}.hdf5")
    filepath = FLAGS.model_file
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, 
                                                                 monitor='val_probs_acc', 
                                                                 verbose=1,
                                                                 save_best_only=True,
                                                                 save_weights_only=True,
                                                                 mode='max')

    callbacks = [checkpoint_callback]
    #callbacks = [tensorboard_callback]
    
    boundaries = [32000, 48000]
    values = [0.1, 0.01, 0.001]
    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

    optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler,
                                        momentum=0.9)

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32)
        return lr

    lr_metric = get_lr_metric(optimizer)

    metrics_basic = {}
    metrics_basic['logits'] = []
    metrics_basic['probs'] = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins,name='ece'),
                              #tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False,name='nll'),
                              nll(name='nll'),
                              BrierScore(name='brier')]
    metrics_basic['certs'] = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins,name='ece'),
                              #tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False,name='nll'),
                              nll(name='nll'),
                              BrierScore(name='brier')]
    metrics_basic['logits_from_certs'] = []    

    
    if FLAGS.model_variant=='1vsall':
        loss_funcs = {'logits':one_vs_all_loss_fn(from_logits=True),
                  'probs':None,
                  'certs':None,
                  'logits_from_certs':None}
        
        metrics = metrics_basic
          
    elif FLAGS.model_variant=='vanilla':        
        loss_funcs = {'logits':tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      'probs':None,
                      'certs':None,
                      'logits_from_certs':None}

        metrics = metrics_basic
        
    else:
        raise ValueError('unknown model_variant')
        
    return callbacks, optimizer, loss_funcs, metrics    

    
# based on um.numpy.plot_diagram, um.numpy.reliability_diagram
def _extract_conf_acc(probs,labels,bins=0):

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
    if bins>0:
        delta = int((len(probs_labels)-window_len)/bins)
        return confidences[::delta],accuracies[::delta]
    else:
        return confidences, accuracies
    
# nonlinear calibration

def calibrate_model_nonlin(model,
                    dataset,
                    FLAGS,
                    output='certs',
                    epochs=10000,
                    verbose=False,
                     bins=4000,
                     basis_type='uniform', # or list
                     basis_params={-10,10,100},
                     basis_list = [-2,-1,0,1,2]
                    ): 
    
    def feature_create(x,basis_exponents):
        x_feat = tf.tile(tf.reshape(x,shape=(-1,1)),[1,len(basis_exponents)])
        be_tf = tf.convert_to_tensor(basis_exponents,dtype=tf.float32)
        return tf.pow(x_feat,tf.exp(be_tf))
    
    def cal_out(W1,x,basis_exponents):
        
        size = len(basis_exponents)
        x_shape = tf.shape(x)   
#         print(x_shape)
        xr = tf.reshape(x,shape=(-1,))
#         print(xr.shape)
        W1_tile = tf.tile(tf.reshape(tf.nn.softmax(W1),[1,size]),[tf.shape(xr)[0],1])
        x_feat = feature_create(xr,basis_exponents)
    #         print(W1_tile.shape)
    #         print(x_feat.shape)
        out = tf.reduce_sum(W1_tile*x_feat,axis=-1)
        return tf.reshape(out,shape=x_shape)


    def cost(W1, x, y):

        yhats = cal_out(W1,x,basis_exponents)
#         print(yhats.shape)
#         print(y.shape)
        cost_value = tf.keras.losses.MSE(y_true=y,
                                         y_pred=yhats)

        return cost_value

    def grad(W1, x, y):

        with tf.GradientTape() as tape:
            cost_value = cost(W1, x, y)

        return cost_value, tape.gradient(cost_value, W1)


    if basis_type=='uniform':
        basis_exponents = np.linspace(*basis_params)
    else:
        basis_exponents = basis_list
    
    
    W1 = tf.Variable(tf.random.normal(shape=(len(basis_exponents),)))
    
    optimizer = ub.optimizers.get(optimizer_name='adam',
                                  learning_rate_schedule='constant',
                                  learning_rate=0.1,
                                  weight_decay=None)    
    
    #number of classes
    K = FLAGS.no_classes
    
    labels = np.empty(0)
    probs = np.empty((0,K))
    
    for i,(x,y) in enumerate(dataset):
        if i>FLAGS.validation_steps: break

        out = model(x)[output].numpy()
        labels = np.append(labels,y.numpy().astype('int32'))
        probs = np.concatenate((probs,out))

    confidences, accuracies = _extract_conf_acc(probs=probs,labels=labels.astype('int32'),bins=bins)
    
    X_train = tf.convert_to_tensor(confidences,dtype=tf.float32)
    y_train = tf.convert_to_tensor(accuracies,dtype=tf.float32)
    
    for i in range(epochs):
        train_cost, grads = grad(W1,X_train,y_train)
        optimizer.apply_gradients(zip([grads], [W1]))
#         if i % 50 == 0: 
#             print(train_cost.numpy())

    def model_return():
        inp = tf.keras.layers.Input(shape=(32,32,3))
        out_model = model(inp)
        out_calibr = cal_out(W1,out_model[output],basis_exponents=basis_exponents)
        out_model[output+'_cal'] = out_calibr
        
        return tf.keras.Model(inputs=inp,outputs=out_model)
    
#     def cal_model(x):
        
#         #out = tf.keras.models.Model(model.layers[0].input, model.output[output])(x)
#         out = model(x)[output]
#         out_shape = out.shape
#         out_calibr = cal_out(W1,out,basis_exponents=basis_exponents)
        
#         return {output+'_cal':out_calibr}
    
    return model_return(),W1