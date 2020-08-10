# Uncertainty Baselines

The goal of Uncertainty Baselines is to provide a template for researchers to build on. The baselines can be a starting point for any new ideas, applications, and/or for communicating with other uncertainty researchers. This is done in three ways:

1. Provide high-quality implementations of standard and state-of-the-art methods on standard tasks.
2. Have minimal dependencies on other files in the codebase. Baselines should be easily forkable without relying on other baselines and generic modules.
3. Prescribe best practices for training and evaluating uncertainty models.

## Motivation

There are many uncertainty implementations across GitHub. However, they are typically one-off experiments for a specific paper (many papers don't even have code). This raises three problems. First, there are no clear examples that uncertainty researchers can build on to quickly prototype their work. Everyone must implement their own baseline. Second, even on standard tasks such as CIFAR-10, projects differ slightly in their experiment setup, whether it be architectures, hyperparameters, or data preprocessing. This makes it difficult to compare properly across methods. Third, there is no clear guidance on which ideas and tricks necessarily contribute to getting best performance and/or are generally robust to hyperparameters.

Non-goals:

* Provide a new benchmark for uncertainty methods. Uncertainty Baselines implements many methods on already-used tasks. It does not propose new tasks. See [OpenAI Baselines](https://github.com/openai/baselines) for a work in similar spirit for RL. For new benchmarks, see [Riquelme et al. (2018)](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits), [Hendrycks and Dietterich (2019)](https://arxiv.org/abs/1903.12261), [Ovadia et al. (2019)](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019), [`OATML/bdl-benchmarks`](https://github.com/OATML/bdl-benchmarks).

## Metrics

We define metrics used across datasets below. All results are reported by roughly 3 significant digits and averaged over 10 runs.

1. __# Parameters.__ Number of parameters in the model to make predictions after training.
2. __Train/Test Accuracy.__ Accuracy over the train and test sets respectively. For a dataset of `N` input-output pairs `(xn, yn)` where the label `yn` takes on 1 of `K` values, the accuracy is

    ```sh
    1/N \sum_{n=1}^N 1[ \argmax{ p(yn | xn) } = yn ],
    ```

    where `1` is the indicator function that is 1 when the model's predicted class is equal to the label and 0 otherwise.
3. __Train/Test Cal. Error.__ Expected calibration error (ECE) over the train and test sets respectively ([Naeini et al., 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090)). ECE discretizes the probability interval `[0, 1]` under equally spaced bins and assigns each predicted probability to the bin that encompasses it. The calibration error is the difference between the fraction of predictions in the bin that are correct (accuracy) and the mean of the probabilities in the bin (confidence). The expected calibration error averages across bins.

    For a dataset of `N` input-output pairs `(xn, yn)` where the label `yn` takes on 1 of `K` values, ECE computes a weighted average

    ```sh
    \sum_{b=1}^B n_b / N | acc(b) - conf(b) |,
    ```

    where `B` is the number of bins, `n_b` is the number of predictions in bin `b`, and `acc(b)` and `conf(b)` is the accuracy and confidence of bin `b` respectively.
4. __Train/Test NLL.__ Negative log-likelihood over the train and test sets respectively (measured in nats). For a dataset of `N` input-output pairs `(xn, yn)`, the negative log-likelihood is

    ```sh
    -1/N \sum_{n=1}^N \log p(yn | xn).
    ```

    It is equivalent up to a constant to the KL divergence from the true data distribution to the model, therefore capturing the overall goodness of fit to the true distribution ([Murphy, 2012](https://www.cs.ubc.ca/~murphyk/MLbook/)). It can also be intepreted as the amount of bits (nats) to explain the data ([Grunwald, 2004](https://arxiv.org/abs/math/0406077)).
5. __Train/Test Runtime.__ Training runtime is the total wall-clock time to train the model, including any intermediate test set evaluations. Test runtime is the total wall-clock make and evaluate predictions on the test set.
