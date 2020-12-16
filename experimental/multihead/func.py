#import os.path

#from absl import app
#from absl import flags
#from absl import logging

import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import uncertainty_metrics as um
import tensorflow_datasets as tfds

import json
import os.path

import utils #from baselines/cifar


def augment_dataset(FLAGS,dataset):
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
    
    #del xs,ys
    for i,ds in enumerate(dataset):
        x,y = ds
        if i>FLAGS.steps_per_epoch: break
        if 'xs' not in locals() and 'ys' not in locals():
            xs = x
            ys = y
        else:
            xs = tf.concat([xs,x],axis=0)
            ys = tf.concat([ys,y],axis=0)
    
    datagen.fit(xs)
    return datagen.flow(x=xs,y=ys,batch_size=FLAGS.batch_size)

def load_datasets_basic(FLAGS):
    
    strategy = ub.strategy_utils.get_strategy(None, False)
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    
    dataset_builder = ub.datasets.Cifar10Dataset(batch_size=FLAGS.batch_size,
                                                 eval_batch_size=FLAGS.eval_batch_size,
                                                 validation_percent=FLAGS.validation_percent)
    
    
    train_dataset = ub.utils.build_dataset(dataset_builder, 
                                           strategy, 
                                           'train', 
                                           as_tuple=True)
    
    
    if FLAGS.augment_train:
        FLAGS = add_dataset_flags(dataset_builder,FLAGS)
        train_dataset = augment_dataset(FLAGS,train_dataset)
        
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

# load cifar100 and svhn datasets
def load_datasets_OOD(FLAGS):
    ood_datasets = {}
    strategy = ub.strategy_utils.get_strategy(None, False)
    dataset_builder = ub.datasets.SvhnDataset(batch_size=FLAGS.batch_size,eval_batch_size=FLAGS.eval_batch_size)
    train_dset = ub.utils.build_dataset(dataset_builder,strategy,'train',as_tuple=True)
    ood_datasets['svhn'] = train_dset
    dataset_builder = ub.datasets.Cifar100Dataset(batch_size=FLAGS.batch_size,eval_batch_size=FLAGS.eval_batch_size)
    train_dset = ub.utils.build_dataset(dataset_builder,strategy,'train',as_tuple=True)
    ood_datasets['cifar100'] = train_dset

    return ood_datasets



def add_dataset_flags(dataset_builder,FLAGS):
    FLAGS.steps_per_epoch = dataset_builder.info['num_train_examples'] // FLAGS.batch_size
    FLAGS.validation_steps = dataset_builder.info['num_validation_examples'] // FLAGS.eval_batch_size
    FLAGS.test_steps = dataset_builder.info['num_test_examples'] // FLAGS.eval_batch_size
    FLAGS.no_classes = 10 # awful but no way to infer from dataset...
    
    return FLAGS


def save_flags(path,FLAGS):
    with open(path,'w') as fp:
        json.dump(FLAGS,fp,indent=4)
        
def load_flags(path):
    assert os.path.exists(path), f'file {path} does not exist'
    with open(path,'r') as fp:
        FLAGS = json.load(fp)
        
    return AttrDict(FLAGS)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    def __copy__(self):
        return AttrDict(self.__dict__.copy())