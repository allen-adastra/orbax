import os
import tempfile
import time
from dataclasses import dataclass

import chex
import epath
import equinox as eqx
import jax

import orbax.checkpoint as ocp


class EquinoxCheckpointHandler(ocp.CheckpointHandler):
    """
    TODO(allenw): not working yet.
    https://github.com/google/orbax/issues/741
    """

    def save(
        self,
        directory: epath.Path,
        args: "EquinoxStateSave",
    ):
        full_path = directory / "model.eqx"
        eqx.tree_serialise_leaves(full_path, args.item, is_leaf=eqx.is_array_like)

    def restore(
        self,
        directory: epath.Path,
        args: "EquinoxStateRestore",
    ) -> eqx.Module:
        loaded = eqx.tree_deserialise_leaves(
            directory / "model.eqx", args.item, is_leaf=eqx.is_array_like
        )
        return loaded


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_save=True)
@dataclass
class EquinoxStateSave(ocp.args.CheckpointArgs):
    item: eqx.Module


@ocp.args.register_with_handler(EquinoxCheckpointHandler, for_restore=True)
@dataclass
class EquinoxStateRestore(ocp.args.CheckpointArgs):
    item: eqx.Module


def build_random_nn(key: jax.random.PRNGKey):
    nn = eqx.nn.MLP(in_size=2, out_size=1, width_size=64, depth=2, key=key)
    return nn


TEMP_DIR = tempfile.mkdtemp()


def test_eqx_save():
    # Build and save a random NN
    nn = build_random_nn(key=jax.random.PRNGKey(42))
    eqx.tree_serialise_leaves(os.path.join(TEMP_DIR, "model_eqx_save.eqx"), nn)

    # Wait a bit.
    time.sleep(5)

    # Build another NN with a different key and restore the saved NN.
    nn_restore = build_random_nn(key=jax.random.PRNGKey(123))
    nn_restore = eqx.tree_deserialise_leaves(
        os.path.join(TEMP_DIR, "model_eqx_save.eqx"), nn_restore
    )
    chex.assert_trees_all_equal(nn, nn_restore)


def test_ocp_save():
    # Build a manager and save a random NN
    options = ocp.CheckpointManagerOptions(enable_async_checkpointing=False)
    manager = ocp.CheckpointManager(
        directory=TEMP_DIR,
        options=options,
    )
    nn = build_random_nn(jax.random.PRNGKey(42))
    save_id = 10
    manager.save(save_id, args=EquinoxStateSave(nn), metrics={"loss": 0.1})

    # Wait for the save to finish add another 5 seconds for good measure.
    manager.wait_until_finished()
    time.sleep(5)

    nn_restore = build_random_nn(jax.random.PRNGKey(123))

    nn_restore = manager.restore(save_id, args=EquinoxStateRestore(nn_restore))  # noqa: F841
