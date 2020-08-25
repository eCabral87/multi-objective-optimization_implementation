# multi-objective-optimization_implementation
A multi-objective bayesian optimization algorithm was implemented

This project implements a Multi-objective Bayesian Global Optimization (MOGBO) algorithm based on the paper: Multi-Objective Bayesian Global Optimization using expected hypervolume improvement gradient (2019).
This algorithm works either for two-objective and one-objective scenarios, and use:
1) Surrogate function: Gaussian Process-based.
2) Acquisition function: Expected Hypervolume Improvement (So far only 2D is implemented), an Probability of Improvement in 1D (1 objective function).

Please observe in the init function (in MOGBO class) the main variables used for configuring the optimization experiment. Additionally, some objective function examples are provided in the function objective. Note that some examples were multiplied by -1 for maximizing instead of minimizing.
Setting up the optimization experiment can be done after __main__. An example was provided to be used as a reference.

