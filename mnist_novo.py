import argparse
import time
from typing import Iterator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax import linen as nn


NUM_CLASSES = 10


def load_mnist() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load MNIST from TFDS and return JAX arrays."""
    train_ds = tfds.load("mnist", split="train", as_supervised=True)
    test_ds = tfds.load("mnist", split="test", as_supervised=True)

    def ds_to_arrays(ds):
        images = []
        labels = []

        for image, label in tfds.as_numpy(ds):
            image = image.astype(np.float32) / 255.0
            images.append(image)
            labels.append(int(label))

        x = np.stack(images, axis=0)  # [N, 28, 28, 1]
        y = jax.nn.one_hot(np.array(labels), NUM_CLASSES, dtype=jnp.float32)

        return jnp.array(x), jnp.array(y)

    x_train, y_train = ds_to_arrays(train_ds)
    x_test, y_test = ds_to_arrays(test_ds)
    return x_train, y_train, x_test, y_test


def batch_iterator(
    x: jnp.ndarray,
    y: jnp.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield shuffled mini-batches forever."""
    n = x.shape[0]
    while True:
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield x[batch_idx], y[batch_idx]


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(6, 6), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(5, 5), strides=(1, 1), padding="VALID")(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        x = nn.Dense(features=NUM_CLASSES)(x)
        return x


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(labels * log_probs, axis=1))


def accuracy_from_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    pred_class = jnp.argmax(logits, axis=1)
    true_class = jnp.argmax(labels, axis=1)
    return jnp.mean(pred_class == true_class)


def create_train_state(model: CNN, rng_key: jax.Array, learning_rate: float):
    """Initialize model parameters and optimizer state."""
    dummy_input = jnp.ones((1, 28, 28, 1), dtype=jnp.float32)
    params = model.init(rng_key, dummy_input)["params"]

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    return params, optimizer, opt_state


def fgsm_attack(model: CNN, params, x, y, eps: float):
    """FGSM attack."""
    def loss_fn(x_in):
        logits = model.apply({"params": params}, x_in)
        return cross_entropy_loss(logits, y)

    grad_x = jax.grad(loss_fn)(x)
    x_adv = x + eps * jnp.sign(grad_x)
    return jnp.clip(x_adv, 0.0, 1.0)


def pgd_attack(
    model: CNN,
    params,
    x,
    y,
    eps: float,
    eps_iter: float = 0.01,
    nb_iter: int = 40,
):
    """PGD attack under L-infinity constraint."""
    x_orig = x
    x_adv = x

    for _ in range(nb_iter):
        def loss_fn(x_in):
            logits = model.apply({"params": params}, x_in)
            return cross_entropy_loss(logits, y)

        grad_x = jax.grad(loss_fn)(x_adv)
        x_adv = x_adv + eps_iter * jnp.sign(grad_x)

        x_adv = jnp.clip(x_adv, x_orig - eps, x_orig + eps)
        x_adv = jnp.clip(x_adv, 0.0, 1.0)

    return x_adv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eps", type=float, default=0.3, help="FGSM/PGD epsilon")
    parser.add_argument("--pgd_eps_iter", type=float, default=0.01, help="PGD step size")
    parser.add_argument("--pgd_nb_iter", type=int, default=40, help="PGD number of iterations")
    args = parser.parse_args()

    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()

    model = CNN()
    rng_key = jax.random.PRNGKey(0)
    params, optimizer, opt_state = create_train_state(
        model=model,
        rng_key=rng_key,
        learning_rate=args.learning_rate,
    )

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        def loss_fn(current_params):
            logits = model.apply({"params": current_params}, x_batch)
            loss = cross_entropy_loss(logits, y_batch)
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        acc = accuracy_from_logits(logits, y_batch)
        return new_params, new_opt_state, loss, acc

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply({"params": params}, x)
        loss = cross_entropy_loss(logits, y)
        acc = accuracy_from_logits(logits, y)
        return loss, acc

    rng = np.random.default_rng(seed=0)
    batches = batch_iterator(x_train, y_train, args.batch_size, rng)
    num_batches = int(np.ceil(x_train.shape[0] / args.batch_size))

    print("\nStarting training...")
    for epoch in range(args.nb_epochs):
        start_time = time.time()
        epoch_losses = []
        epoch_accs = []

        for _ in range(num_batches):
            x_batch, y_batch = next(batches)
            params, opt_state, loss, acc = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
            epoch_accs.append(acc)

        epoch_time = time.time() - start_time

        _, train_acc = eval_step(params, x_train, y_train)
        _, test_acc = eval_step(params, x_test, y_test)

        x_test_fgm = fgsm_attack(model, params, x_test, y_test, args.eps)
        x_test_pgd = pgd_attack(
            model,
            params,
            x_test,
            y_test,
            args.eps,
            eps_iter=args.pgd_eps_iter,
            nb_iter=args.pgd_nb_iter,
        )

        _, test_acc_fgm = eval_step(params, x_test_fgm, y_test)
        _, test_acc_pgd = eval_step(params, x_test_pgd, y_test)

        print(f"Epoch {epoch + 1} in {epoch_time:.2f} sec")
        print(f"Train minibatch loss (mean): {jnp.mean(jnp.array(epoch_losses)):.4f}")
        print(f"Train minibatch acc  (mean): {jnp.mean(jnp.array(epoch_accs)):.4f}")
        print(f"Training set accuracy: {train_acc:.4f}")
        print(f"Test set accuracy on clean examples: {test_acc:.4f}")
        print(f"Test set accuracy on FGM adversarial examples: {test_acc_fgm:.4f}")
        print(f"Test set accuracy on PGD adversarial examples: {test_acc_pgd:.4f}")


if __name__ == "__main__":
    main()
