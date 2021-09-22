from allennlp.training import optimizers as _allennlp_optimizers

from trapper.common import Registrable
from trapper.common.utils import append_parent_docstr


@append_parent_docstr(parent_id=1)
class Optimizer(Registrable, _allennlp_optimizers.Optimizer):
    """
    The base `Registrable` optimizer class that replaces the one from the
    `allennlp` library.
    """


@Optimizer.register("adam")
class AdamOptimizer(Optimizer, _allennlp_optimizers.AdamOptimizer):
    """
    `Adam` optimizer registered with name "adam".
    """


@Optimizer.register("sparse_adam")
class SparseAdamOptimizer(Optimizer, _allennlp_optimizers.SparseAdamOptimizer):
    """
    `SparseAdam` optimizer registered with name "sparse_adam".
    """


@Optimizer.register("adamax")
class AdamaxOptimizer(Optimizer, _allennlp_optimizers.AdamaxOptimizer):
    """
    `Adamax` optimizer registered with name "adamax".
    """


@Optimizer.register("adamw")
class AdamWOptimizer(Optimizer, _allennlp_optimizers.AdamWOptimizer):
    """
    `AdamW` optimizer registered with name "adamw".
    """


@Optimizer.register("huggingface_adamw")
class HuggingfaceAdamWOptimizer(
    Optimizer, _allennlp_optimizers.HuggingfaceAdamWOptimizer
):
    """
    `HuggingfaceAdamW` optimizer registered with name "huggingface_adamw".
    """


@Optimizer.register("adagrad")
class AdagradOptimizer(Optimizer, _allennlp_optimizers.AdagradOptimizer):
    """
    `Adagrad` optimizer registered with name "adagrad".
    """


@Optimizer.register("adadelta")
class AdadeltaOptimizer(Optimizer, _allennlp_optimizers.AdadeltaOptimizer):
    """
    `Adadelta` optimizer registered with name "adadelta".
    """


@Optimizer.register("sgd")
class SgdOptimizer(Optimizer, _allennlp_optimizers.SgdOptimizer):
    """
    `Sgd` optimizer registered with name "sgd".
    """


@Optimizer.register("rmsprop")
class RmsPropOptimizer(Optimizer, _allennlp_optimizers.RmsPropOptimizer):
    """
    `RmsProp` optimizer registered with name "rmsprop".
    """


@Optimizer.register("averaged_sgd")
class AveragedSgdOptimizer(Optimizer, _allennlp_optimizers.AveragedSgdOptimizer):
    """
    `AveragedSgd` optimizer registered with name "averaged_sgd".
    """


@append_parent_docstr(parent_id=1)
@Optimizer.register("dense_sparse_adam")
class DenseSparseAdamOptimizer(
    Optimizer,
    _allennlp_optimizers.DenseSparseAdam,
):
    """
    `DenseSparseAdam` optimizer registered with name "dense_sparse_adam".
    """
