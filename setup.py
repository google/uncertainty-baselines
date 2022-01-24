"""Uncertainty Baselines.

See more details in the
[`README.md`](https://github.com/google/uncertainty-baselines).
"""

import os
import sys

from setuptools import find_packages
from setuptools import setup

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'uncertainty_baselines')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setup(
    name='uncertainty_baselines',
    version=__version__,
    description='Uncertainty Baselines',
    author='Uncertainty Baselines Team',
    author_email='znado@google.com',
    url='http://github.com/google/uncertainty-baselines',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.8.1',
        'astunparse',
        'chardet',
        'flatbuffers',
        'google-cloud-storage',  # Required for the GCS launcher.
        'idna',
        'ml_collections',
        'numpy>=1.7',
        'opt_einsum',
        ('robustness_metrics @ '
         'git+https://github.com/google-research/robustness_metrics.git'
         '#egg=robustness_metrics'),
        # Required because RM does not do lazy loading and RM requires TFP.
        'tensorflow_probability',
        'tfds-nightly==4.4.0.dev202111160106',
        'urllib3',
        'zipp',
    ],
    extras_require={
        'experimental': [],
        'models': [
            'edward2 @ git+https://github.com/google/edward2.git#egg=edward2',
            'pandas',
            'scipy',
        ],
        'datasets': [
            'librosa',  # Needed for speech_commands dataset
            'scipy',  # Needed for speech_commands dataset
            # TODO(dusenberrymw): Add these without causing a dependency
            # resolution issue.
            # 'seqio',  # Needed for smcalflow and multiwoz datasets
            # 't5',  # Needed for smcalflow and multiwoz datasets
        ],
        'jax': [
            'clu',
            'flax',
            'jax',
            'jaxlib',
        ],
        'tensorflow': [
            'tensorboard',
            'tensorflow>=2.6',
            'tensorflow_addons',  # Required for optimizers.py.
            'tf-models-official',  # Needed for BERT.
        ],
        'tf-nightly': [
            'tb-nightly',
            'tf-nightly',
            'tfa-nightly',
            'tfp-nightly',
            'tf-models-nightly',  # Needed for BERT, depends on tf-nightly.
        ],
        'tests': ['pylint>=1.9.0'],
        'torch': [
            'torch',
            'torchvision',
        ],
        'retinopathy': ['wandb', 'seaborn', 'dm-haiku'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='probabilistic programming tensorflow machine learning',
)
