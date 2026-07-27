import logging

import numpy as np
import cvxpy as cp
import pandas as pd

logger = logging.getLogger(__name__)


def LSwConstr(b, A, C, parameter_names=None, b_val=None, A_val=None):

    assert A.shape[0] == b.shape[0], f"A/b row mismatch: {A.shape[0]} vs {b.shape[0]}"
    if C is not None:
        assert C.shape[1] == A.shape[1], f"C/A column mismatch: {C.shape[1]} vs {A.shape[1]}"

    x = cp.Variable(A.shape[1], name='x')
    obj = cp.norm2(A @ x - b)
    constr = [] if C is None else [C @ x == 0]  # without constraints will give back same value as OLS (=using only pseudo inverse)
    prob = cp.Problem(cp.Minimize(obj), constr)

    logger.debug('Problem formulation: %s', prob)
    logger.debug('Curvature of the objective function: %s', obj.curvature)
    logger.debug('The problem is well conditioned: %s', prob.is_dcp())

    prob.solve(verbose=False, solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"LSwConstr: solver did not converge, status={prob.status}")

    train_l2_loss = 0.5 * prob.value**2 / A.shape[0]
    df = pd.DataFrame(data={'coef': x.value}, index=parameter_names)

    logger.info('Solved Constrained Least Square problem: objective value=%.6g, l2_loss=%.6g', prob.value, train_l2_loss)

    # Compute validation losses on a held-out set, if provided
    val_l2_loss = None
    if A_val is not None and b_val is not None:
        val_residual = A_val @ x.value - b_val
        val_residual_norm = np.linalg.norm(val_residual, ord=2)
        val_l2_loss = 0.5 * val_residual_norm**2 / A_val.shape[0]

        logger.info('Validation performance: residual norm=%.6g, l2_loss=%.6g' % (val_residual_norm, val_l2_loss))

    return df, x.value, prob.value, train_l2_loss, val_l2_loss
