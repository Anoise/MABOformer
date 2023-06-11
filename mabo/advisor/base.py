# License: MIT


from ..utils.constants import MAXINT
from ..acq.acquisition import EI
from ..utils.util_funcs import get_types


acq_dict = {
    'ei': EI,
}


def build_acq_func(func_str='ei', model=None, constraint_models=None, **kwargs):
    func_str = func_str.lower()
    acq_func = acq_dict.get(func_str)
    if acq_func is None:
        raise ValueError('Invalid string %s for acquisition function!' % func_str)
    if constraint_models is None:
        return acq_func(model=model, **kwargs)
    else:
        return acq_func(model=model, constraint_models=constraint_models, **kwargs)


def build_optimizer(func_str='local_random', acq_func=None, config_space=None, rng=None):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        from ..acq.ei_optimization import InterleavedLocalAndRandomSearch
        optimizer = InterleavedLocalAndRandomSearch
    elif func_str == 'random_scipy':
        from ..acq.ei_optimization import RandomScipyOptimizer
        optimizer = RandomScipyOptimizer
    elif func_str == 'scipy_global':
        from ..acq.ei_optimization import ScipyGlobalOptimizer
        optimizer = ScipyGlobalOptimizer
    elif func_str == 'batchmc':
        from ..acq.ei_optimization import batchMCOptimizer
        optimizer = batchMCOptimizer
    elif func_str == 'staged_batch_scipy':
        from ..acq.ei_optimization import StagedBatchScipyOptimizer
        optimizer = StagedBatchScipyOptimizer
    else:
        raise ValueError('Invalid string %s for acq_maximizer!' % func_str)

    return optimizer(acquisition_function=acq_func,
                     config_space=config_space,
                     rng=rng)


def build_surrogate(func_str='gp', config_space=None, rng=None, transfer_learning_history=None):
    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    seed = rng.randint(MAXINT)

    if func_str == 'random_forest':
        from ..surrogate.skrf import RandomForestSurrogate
        return RandomForestSurrogate(config_space, types=types, bounds=bounds, seed=seed)

    elif func_str.startswith('gp'):
        from ..surrogate.build_gp import create_gp_model
        return create_gp_model(model_type=func_str,
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)
    else:
        raise ValueError('Invalid string %s for surrogate!' % func_str)
