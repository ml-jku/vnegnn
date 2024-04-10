import hashlib
import os
from functools import partial
from typing import Any, Callable, List, Optional, Type

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn.cluster import MeanShift
from tqdm import tqdm


def set_niceness():
    desired_niceness = 10
    os.nice(desired_niceness)


def pmap_multi(
    pickleable_fn,
    data,
    n_jobs: int = None,
    verbose: int = 1,
    desc: str = None,
    **kwargs,
):
    """

    Parallel map using joblib.
    https://github.com/HannesStark/EquiBind

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the equibind_test_predictions to stdout.
    desc: str, optional
        The description for the progress bar.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the equibind_test_predictions of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data), desc=desc)
    )

    return results


def parallel_class_method_execution(
    ClassToInitialize: Type[Any],
    constant_args,
    method_name: str,
    variable_args: List[Any],
    n_jobs: Optional[int] = None,
    verbose: int = 1,
    desc: Optional[str] = None,
) -> List[Any]:
    """Parallel execution of a class method.

    Args:
        ClassToInitialize (Type[Any]): The class to initialize.
        method_name (str): The name of the method to call.
        variable_args (List[Any]): The variable arguments (e.g., complex_names).
        verbose (int, optional): Verbosinty level for joblib Paralell. Defaults to 1.
        desc (Optional[str], optional): Description for tqdm. Defaults to None.
        constant_args (Tuple[Any, ...], optional): The constant arguments. Defaults to ().

    Returns:
        List[Any]: The results of the method calls.
    """

    def wrapper(var_arg):
        # Here we initialize the class with the combined arguments
        instance = ClassToInitialize(var_arg, **constant_args)
        method_to_call = getattr(instance, method_name)
        return method_to_call()

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
        delayed(wrapper)(arg) for arg in tqdm(variable_args, desc=desc, total=len(variable_args))
    )

    return results


def read_strings_from_txt(path: str) -> List[str]:
    """Read strings of a text file line by line.

    Args:
        path (str): The file path.

    Returns:
        List[Any]: The list of strings.
    """
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


try:
    HASH = hashlib._Hash
except AttributeError:
    HASH = Any


class PartialHashError(TypeError):
    """Raised if partial hash cannot hash a given object."""


def partial_hash(obj: Any, hash: Optional[HASH] = None) -> HASH:  # type: ignore
    """Partially hash a given object."""

    if hash is None:
        hash = hashlib.sha256()

    if isinstance(obj, (int, float, complex)):
        obj = repr(obj)

    if isinstance(obj, str):
        hash.update(obj.encode("utf-8"))
        return hash

    if isinstance(obj, partial):
        hash = partial_hash(obj.func, hash)
        hash = partial_hash(obj.args, hash)
        hash = partial_hash(obj.keywords, hash)
        return hash

    if isinstance(obj, Callable):
        hash = partial_hash(obj.__module__, hash)
        hash = partial_hash(obj.__name__, hash)
        return hash

    if isinstance(obj, (list, tuple)):
        for x in obj:
            hash = partial_hash(x, hash)
        return hash

    if isinstance(obj, set):
        for x in sorted(obj):
            hash = partial_hash(x, hash)
        return hash

    if isinstance(obj, dict):
        for x in sorted(obj):
            hash = partial_hash(x, hash)
            hash = partial_hash(obj[x], hash)
        return hash

    raise PartialHashError(f"cannot partially hash {obj!r}")


def get_clusterd_predictions(pos_sample: np.array, rank_sample: np.array) -> np.array:
    labels = MeanShift().fit_predict(np.concatenate([pos_sample, rank_sample], axis=1))
    unique_labels = np.unique(labels)

    pos_means = []
    rank_means = []
    for label in unique_labels:
        pos_means.append(np.mean(pos_sample[labels == label], axis=0))
        rank_means.append(np.mean(rank_sample[labels == label], axis=0))
    pos_sample = np.stack(pos_means)
    rank_sample = np.stack(rank_means)
    return pos_sample, rank_sample


def load_txt_file(file_path: str) -> List[str]:
    file = ""
    with open(file_path, "r") as f:
        file = f.read()
    return file
