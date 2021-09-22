import jax.numpy as jnp
import optax


DEFAULT_NUM_EPOCHS = 90


class OptimizerInitializer:
    def __init__(
        self,
        optimizer: str,
        base_learning_rate,
        n_batches,
        epochs,
        one_minus_momentum,
        lr_warmup_epochs,
        lr_decay_ratio,
        lr_decay_epochs,
        final_decay_factor,
        lr_schedule,
    ):
        self.optimizer = optimizer
        self.base_learning_rate = base_learning_rate
        self.n_batches = n_batches
        self.epochs = epochs
        self.one_minus_momentum = one_minus_momentum
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_decay_ratio = lr_decay_ratio
        self.lr_decay_epochs = lr_decay_epochs
        self.final_decay_factor = final_decay_factor
        self.lr_schedule = lr_schedule

    def initialize_optimizer(self) -> optax.GradientTransformation:
        if "adam" in self.optimizer:
            opt = optax.adam(self.base_learning_rate)
        elif "sgd" == self.optimizer and self.lr_schedule == "linear":
            lr_schedule = warm_up_polynomial_schedule(
                base_learning_rate=self.base_learning_rate,
                end_learning_rate=self.final_decay_factor * self.base_learning_rate,
                decay_steps=(self.n_batches * (self.epochs - self.lr_warmup_epochs)),
                warmup_steps=self.n_batches * self.lr_warmup_epochs,
                decay_power=1.0
            )
            momentum = 1 - self.one_minus_momentum
            opt = optax.chain(
                optax.trace(decay=momentum, nesterov=True),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1),
            )
        elif "sgd" in self.optimizer and self.lr_schedule == "step":
            lr_decay_epochs = [
                (int(start_epoch_str) * self.epochs) // DEFAULT_NUM_EPOCHS
                for start_epoch_str in self.lr_decay_epochs
            ]
            lr_schedule = warm_up_piecewise_constant_schedule(
                steps_per_epoch=self.n_batches,
                base_learning_rate=self.base_learning_rate,
                decay_ratio=self.lr_decay_ratio,
                decay_epochs=lr_decay_epochs,
                warmup_epochs=self.lr_warmup_epochs)

            momentum = 1 - self.one_minus_momentum
            opt = optax.chain(
                optax.trace(decay=momentum, nesterov=True),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1),
            )
        else:
            raise ValueError("No optimizer specified.")
        return opt


def warm_up_piecewise_constant_schedule(
        steps_per_epoch,
        base_learning_rate,
        warmup_epochs,
        decay_epochs,
        decay_ratio,
    ):
    def schedule(count):
        lr_epoch = jnp.array(count, jnp.float32) / steps_per_epoch
        learning_rate = base_learning_rate
        if warmup_epochs >= 1:
            learning_rate *= lr_epoch / warmup_epochs
        _decay_epochs = [warmup_epochs] + decay_epochs
        for index, start_epoch in enumerate(_decay_epochs):
            learning_rate = jnp.where(
                lr_epoch >= start_epoch,
                base_learning_rate * decay_ratio ** index,
                learning_rate)
        return learning_rate
    return schedule


def warm_up_polynomial_schedule(
    base_learning_rate,
    end_learning_rate,
    decay_steps,
    warmup_steps,
    decay_power,
):
    poly_schedule = optax.polynomial_schedule(
        init_value=base_learning_rate,
        end_value=end_learning_rate,
        power=decay_power,
        transition_steps=decay_steps,
    )

    def schedule(step):
        lr = poly_schedule(step)
        indicator = jnp.maximum(0.0, jnp.sign(warmup_steps - step))
        warmup_lr = base_learning_rate * step / warmup_steps
        lr = warmup_lr * indicator + (1 - indicator) * lr
        return lr

    return schedule
