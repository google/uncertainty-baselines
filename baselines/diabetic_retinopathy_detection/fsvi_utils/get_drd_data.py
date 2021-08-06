import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from absl import flags


DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_NUM_EPOCHS = 1 # 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/deterministic',
    'The directory where the model weights and training/evaluation summaries '
    'are stored. If you aim to use these as trained models for ensemble.py, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')

# Learning rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 4e-4, 'Base learning rate.')
flags.DEFINE_float('final_decay_factor', 1e-3, 'How much to decay the LR by.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_string('lr_schedule', 'step', 'Type of LR schedule.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 1,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '60'],
                  'Epochs to decay learning rate by.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string(
    'class_reweight_mode', None,
    'Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, val,'
    'test respectively). `None` (default) will not perform any loss reweighting. '
    '`constant` will use the train proportions to reweight the binary cross '
    'entropy loss. `minibatch` will use the proportions of each minibatch to '
    'reweight the loss.')
flags.DEFINE_float('l2', 5e-5, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('train_batch_size', DEFAULT_TRAIN_BATCH_SIZE,
                     'The per-core training batch size.')
flags.DEFINE_integer('eval_batch_size', 2,
                     'The per-core validation/test batch size.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.')
FLAGS = flags.FLAGS


def get_drd_data():
    strategy = utils.init_distribution_strategy(FLAGS.force_use_cpu,
                                                FLAGS.use_gpu, FLAGS.tpu)
    use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

    train_batch_size = FLAGS.train_batch_size * FLAGS.num_cores
    eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores

    # Reweighting loss for class imbalance
    class_reweight_mode = FLAGS.class_reweight_mode
    if class_reweight_mode == 'constant':
        class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
    else:
        class_weights = None


    ds_info = tfds.builder('diabetic_retinopathy_detection').info
    steps_per_epoch = ds_info.splits['train'].num_examples // train_batch_size
    steps_per_validation_eval = (
            ds_info.splits['validation'].num_examples // eval_batch_size)
    steps_per_test_eval = ds_info.splits['test'].num_examples // eval_batch_size

    data_dir = FLAGS.data_dir

    dataset_train_builder = ub.datasets.get(
        'diabetic_retinopathy_detection', split='train', data_dir=data_dir)
    dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

    dataset_validation_builder = ub.datasets.get(
        'diabetic_retinopathy_detection',
        split='validation',
        data_dir=data_dir,
        is_training=not FLAGS.use_validation)
    validation_batch_size = (
        eval_batch_size if FLAGS.use_validation else train_batch_size)
    dataset_validation = dataset_validation_builder.load(
        batch_size=validation_batch_size)
    if FLAGS.use_validation:
        dataset_validation = strategy.experimental_distribute_dataset(
            dataset_validation)
    else:
        # Note that this will not create any mixed batches of train and validation
        # images.
        dataset_train = dataset_train.concatenate(dataset_validation)

    dataset_train = strategy.experimental_distribute_dataset(dataset_train)

    dataset_test_builder = ub.datasets.get(
        'diabetic_retinopathy_detection', split='test', data_dir=data_dir)
    dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)
    dataset_test = strategy.experimental_distribute_dataset(dataset_test)
