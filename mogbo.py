# Multi-objective Bayesian Global optimization implementation (MOBGO)
# as presented in paper (2019): <Multi-Objective Bayesian Global Optimization using expected hypervolume improvement gradient>
# Surrogate function: Gaussian Process based
# Acquisition function: Expected Hypervolume Improvement (So far only 2D is implemented), an Probability of Improvement in 1D (1 objective fuunction)
# Author: eCabral87 (gcabral@u2py.mx)

from math import sin
from math import pi, inf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import norm
from scipy.special import erf
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor


class trials:
    def __init__(self):
        self.params = np.array([])
        self.scores = np.array([])
        self.vals = {}

def append_val_results(dict, suggest):
    if not dict:
        for k in [key for key in suggest]:
            dict[k] = [suggest[k]]
    else:
        for k in dict:
            dict[k].append(suggest[k])
    return dict

def objective(x, random_state, dimension=1, noise=0.0):
    # Please observe objective function examples in https://en.wikipedia.org/wiki/Test_functions_for_optimization as those shown below
    noise = random_state.normal(loc=0, scale=noise)

    if dimension == 1:  # > 1D
        return ((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2) + noise
    else:
        #return [(-x[0] ** 2 + -x[1] ** 2 + 6) + noise, (x[0] ** 2 + x[1] ** 2 + 6) + noise]
        #return [((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2) + noise, (-x[0]**2 + -x[1]**2 + 6) + noise]
        return [((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2) + noise, 0.26*(x[0] ** 2 + x[1] ** 2)-0.48*x[0]*x[1] + noise]
        # return [(-(x[0] ** 2 + x[1] - 11) ** 2 - (x[0] + x[1] ** 2 - 7) ** 2) - noise,
        #       -0.26*(x[0] ** 2 + x[1] ** 2)+0.48*x[0]*x[1] - noise]
        #return [(x[0]**2 + x[1]**2) + noise, 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1] + noise]


class MOGBO:
    def __init__(self, funct, eval, domain, goal_loss, constraint = None, rstate=None, maximize=False):
        self.default_n_initial_samples = 20  # Set the number of initial samples to build a gaussian process model
        self.default_n_draw_samples = 100  # Set the number of samples to construct the acquisition function
        self.trials = trials()  # Variable which saves the params, scores of every iteration
        self.funct = funct  # Name of the objective function used
        self.eval = eval  # Number of iterations or objective functions evaluations during the optimization experiment
        self.domain = domain  # Search space; variables and their range
        self.goal_loss = goal_loss  # Use goal_loss = [f1_goal, ... , fn_goal]. Otherwise when only f1: goal_loss = f1_goal
        self.maximize = maximize  # by default all fn are sought to be minimized (better score). Otherwise,
                        #use the next configuration: maximize = [True, False, True, ...] or maximize = True for all fn.
        self.pareto = {}  # Pareto front: those points in which none of its metrics can be improved without degrading the other one
        self.pareto_best = {}  # Best Pareto front
        self.constraint = constraint # [(f1_const), (f2_const)]. f1_const = (min, max); f2_const = (min, max)
        if rstate is None:  # Set an integer random state for replicability issues
            self.rstate = np.random.RandomState()
        else:
            self.rstate = rstate


    def surrogate(self, model, X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return model.predict(X, return_std=True)


    def opt_acquisition(self, X, model):
        # optimize the acquisition function
        # X: independent parameters history
        # model: GP regresor model

        # random search, generate random samples
        Xsamples = np.zeros((self.default_n_draw_samples, len(self.domain)))
        for i, key in enumerate(self.domain):
            Xsamples[:, i] = [self.rstate.uniform(self.domain[key][0], self.domain[key][1]) for _ in range(self.default_n_draw_samples)]

        # 7) and 8) Train GP models and Find promising points based on the surrogate
        if type(self.goal_loss) is list:  # > 1D
            # calculate the acquisition function for each sample.
            ix = self.acquisition_nd(Xsamples, model)
        else: # 1D
            # calculate the acquisition function for each sample.
            probs = self.acquisition_1d(X, Xsamples, model[0])
            # locate the index of the largest scores
            ix = np.argmax(probs)

        # Save into trial.val variable next suggestion
        for i, key in enumerate(self.domain):
            self.trials.vals[key] = np.hstack((self.trials.vals[key], Xsamples[ix, i]))
        return Xsamples[ix]

    def get_mu_std(self, X, model):
        mu, std = self.surrogate(model, X)
        return [mu, std]

    def estimate_reference(self):
        # Based on GPflow library: https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/acquisition/hvpoi.py
        pf = self.pareto_best['y']
        f = np.max(pf, axis=0, keepdims=True) - np.min(pf, axis=0, keepdims=True)
        return np.max(pf, axis=0, keepdims=True) + 2 * f / pf.shape[0]


    def acquisition_nd(self, Xsamples, model):
        # Expected Hypervolume Improvement calculation
        mu = []
        std = []
        for j in range(len(self.goal_loss)):
            mu.append(self.get_mu_std(Xsamples, model[j])[0])
            std.append(self.get_mu_std(Xsamples, model[j])[1])

        # Expected Hypervolume Improvement calculation
        ExI = np.zeros(len(mu[0]))
        for i in range(len(mu[0])):
            ExI[i] = self.eihv_2d(self.pareto_best['y'], self.estimate_reference().tolist()[0], [mu[0][i], mu[1][i]], [std[0][i], std[1][i]])
        return np.argmax(ExI)


    def acquisition_1d(self, X, Xsamples, model):
        # probability of improvement acquisition function

        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(model, X)
        best = min(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, Xsamples)
        mu = mu[:, 0]
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std + 1E-9))
        return probs

    def initial_mogbo(self):

        # 1) Initialize mu points {x1, x2, ..., xmu}
        x = np.zeros((self.default_n_initial_samples, len(self.domain)))
        if type(self.goal_loss) is list:
            n_loss = len(self.goal_loss)
            y = np.zeros((self.default_n_initial_samples, len(self.goal_loss)))

        else:
            n_loss = 1
            y = np.zeros((self.default_n_initial_samples, 1))

        for i, key in enumerate(self.domain):
            x[:, i] = [self.rstate.uniform(self.domain[key][0], self.domain[key][1]) for _ in
                              range(self.default_n_initial_samples)]

        # 2) Evaluate the initial mu points mu point {y1=f(x1), y2=f(x2), ..., yxmu=f(xmu)}
        if n_loss > 1:
            for i in range(n_loss):
                y[:, i] = np.asarray([self.funct(k, self.rstate, dimension=n_loss)[i] for k in x])  # remove self.state in future
        else:
            y[:, 0] = np.asarray([self.funct(k, self.rstate, dimension=n_loss) for k in x])  # remove self.state in future

        # 3) Store x and y in D
        init_param = {}
        for i, key in enumerate(self.domain):
            init_param[key] = x[:,i]
        self.trials.vals = init_param

        self.trials.params = np.zeros((self.default_n_initial_samples, len(self.domain)))
        self.trials.scores = np.zeros((self.default_n_initial_samples, n_loss))
        self.trials.params = x
        self.trials.scores = y

    def is_pareto_efficient(self, costs, maximize = False, return_mask = True):
        """ 4) Compute the non-dominated subset of D
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param maximize: by default all n_cost are sought to be minimized (better score). Otherwise,
                        use the next configuration: maximize = [True, False, True, ...] or maximize = True for all n_cost.
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            if type(maximize) is list:
                tmp_cost = np.array([maximize for i in range(len(costs))])
                for i, m in enumerate(maximize):
                    if m:
                        tmp_cost[:, i] = costs[:, i] > costs[next_point_index, i]
                    else:
                        tmp_cost[:, i] = costs[:, i] < costs[next_point_index, i]
                nondominated_point_mask = np.any(tmp_cost, axis=1)
            elif not maximize:
                nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            else:
                nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)

            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def pareto_constrained(self, p):
        # Only 2D case addressed
        # p: pareto front. p = [[f10, f20], [f11, f21], ..., [f1n, f2n]]
        # constraint format: [(f1_const), (f2_const)]

        # Find the point in p that satisfy the constraints
        f1_mask = np.zeros(len(p))
        f2_mask = np.zeros(len(p))
        for i in range(len(p)):
            f1_mask[i] = (p[i,0] <= self.constraint[0][1]) and (p[i, 0] >= self.constraint[0][0])
            f2_mask[i] = (p[i, 1] <= self.constraint[1][1]) and (p[i, 1] >= self.constraint[1][0])
        f_mask = np.all(np.vstack((f1_mask,f2_mask)), axis=0)

        # Check if none of the points satisfy the constraints
        if not any(f_mask):
            #-------------p == np.median(p, axis=0) --------------------
            f_mask = np.zeros(p.shape, dtype=bool) # False by default
            idx = (np.abs(p - np.median(p, axis=0))).argmin(axis=0)
            for i in range(f_mask.shape[1]):
                for j in range(f_mask.shape[0]):
                    if j == idx[i]:
                        f_mask[idx[i], i] = True
            #-----------------------------------------------------------
            self.pareto_best['x'] = self.pareto['x'][np.any(f_mask, axis=1)]
            self.pareto_best['y'] = p[np.any(f_mask, axis=1)]  # Return the points closer to the median of both metrics
        else:  # Return pareto points that satisfy constraints
            self.pareto_best['x'] = self.pareto['x'][f_mask]
            self.pareto_best['y'] = p[f_mask]

    def eihv_2d(self, p, r, mu, s):
        # P: approximation set 2xK, r: reference point, mu: mean vector, s:stdev
        # Example: exi2d(np.array([[0,2], [1,1], [2,0]]),[3,3],[0,0],[0.1,0.1]) should approximately
        # result in 3

        # Obtained in Matlab format from http://liacs.leidenuniv.nl/~csmoda/index.php?page=code for
        # Hypervolume Based Expected Improvement (based in paper: The computation of the expected improvement
        # indominated hypervolume of Pareto frontapproximation)

        # determine all lower left corner cell coordinates
        S = p[np.lexsort(np.fliplr(p).T)]  # sort (ascendingly) rows by columns, just as sortrows Matlab function
        k = len(S)  # len of columns
        c1 = np.sort(S[:, 0])
        c2 = np.sort(S[:, 1])

        c = np.zeros((k+1, k+1))
        for i in range(k+1):
            for j in range(k-i+1):
                # c1(i), c2(j) are now the cell coordinates according
                # For coordinate j determine hight fMax2
                if (j==0):
                    fMax2 = r[1]
                else:
                    fMax2 = c2[k-j]
                # For coordinate i determine the width of the staircase fMax1
                if (i==0):
                    fMax1 = r[0]
                else:
                    fMax1 = c1[k-i]
                # get cell coordinates
                if (j==0):
                    cL1 = -inf
                else:
                    cL1 = c1[j-1]
                if (i==0):
                    cL2 = -inf
                else:
                    cL2 = c2[i-1]
                if (j == k):
                    cU1 = r[0]
                else:
                    cU1 = c1[j]
                if (i == k):
                    cU2 = r[1]
                else:
                    cU2 = c2[i]
                # SM = points that are dominated or equal to upper cell bound
                SM = np.zeros((1,2))
                n_assert = 0
                for m in range(k):
                    if (cU1 <= S[m,0] and cU2 <= S[m,1]):
                        if n_assert == 0:
                            SM[0,:] = [S[m,0], S[m, 1]]
                        else:
                            SM = np.vstack(([S[m,0], S[m, 1]], SM)) # first in first row
                        n_assert += 1

                sPlus = self.hvolume2d(SM, [fMax1, fMax2], n_assert)
                # Marginal integration over the length of a cell
                Psi1 = self.exipsi(fMax1, cU1, mu[0], s[0]) - self.exipsi(fMax1, cL1, mu[0], s[0])
                # Marginal integration over the height of a cell
                Psi2 = self.exipsi(fMax2, cU2, mu[1], s[1]) - self.exipsi(fMax2, cL2, mu[1], s[1])
                # Cumulative Gaussian over length for correction constant
                GaussCDF1 = self.gausscdf((cU1 - mu[0]) / (s[0]+1E-9)) - self.gausscdf((cL1 - mu[0]) / (s[0]+1E-9))
                # Cumulative Gaussian over length for correction constant
                GaussCDF2 = self.gausscdf((cU2 - mu[1]) / (s[1]+1E-9)) - self.gausscdf((cL2 - mu[1]) / (s[1]+1))
                c[i, j] = Psi1*Psi2-sPlus*GaussCDF1*GaussCDF2

        return np.sum(np.sum(c))


    def hvolume2d(self, P, x, empti):
        S = P[np.lexsort(np.fliplr(P).T)]  # sort (ascendingly) rows by columns, just as sortrows Matlab function
        h = 0
        if not(empti) == 0:
            k = len(S[:, 0])
            for i in range(k):
                if (i == 0):
                    h = h + (x[0]-S[i, 0]) * (x[1]-S[i, 1])
                else:
                    h = h + (x[0] - S[i, 0]) * (S[i-1, 1] - S[i, 1])
        return h

    def exipsi(self, a, b, m, s):
        return s*self.gausspdf((b-m)/(s+1E-9)) + (a-m)*self.gausscdf((b-m)/(s+1E-9))

    def gausspdf(self, x):
        return 1/np.sqrt(2*pi)*np.exp(-x**2/2)

    def gausscdf(self, x):
        return 0.5*(1 + erf(x/np.sqrt(2)))


    def fmin(self, g):

        MODEL = {}
        if type(self.goal_loss) is list: # > 1D
            n_loss = len(self.goal_loss)
            for j in range(len(self.goal_loss)):
                MODEL[j] = GaussianProcessRegressor(random_state=self.rstate)
                MODEL[j].fit(self.trials.params, self.trials.scores[:,j])
        else:  # 1D
            n_loss = 1
            # define the model
            MODEL[0] = GaussianProcessRegressor(random_state=self.rstate)
            # fit the model
            MODEL[0].fit(self.trials.params, self.trials.scores)

        # perform the optimization process
        while g < (self.eval):  # 5) and 6) steps

            # select the next point to sample 7 -> 8)
            x = self.opt_acquisition(self.trials.params, MODEL)
            # sample the point
            actual = self.funct(x, self.rstate, dimension=n_loss)

            if type(self.goal_loss) is list:  # > 1D
                est = []
                for j in range(len(self.goal_loss)):
                    # summarize the finding
                    est.append(self.surrogate(MODEL[j], [x])[0].tolist()[0])
                    # 9) add the data to the dataset
                    if j == 0:
                        self.trials.params = np.vstack((self.trials.params, [x]))
                        self.trials.scores = np.vstack((self.trials.scores, [actual]))
                    # update the model
                    MODEL[j].fit(self.trials.params, self.trials.scores[:,j])
                print('%d>x=%s, f()=%s, actual=%s' % (g + 1, x, est, actual))

                # 10) Update Pareto front
                mapp = self.is_pareto_efficient(self.trials.scores, self.maximize)
                self.pareto['x'] = self.trials.params[mapp, :]
                self.pareto['y'] = self.trials.scores[mapp, :]
                if self.constraint is not None and type(self.goal_loss) is list:
                    self.pareto_constrained(self.pareto['y'])
                else:
                    self.pareto_best = self.pareto.copy()

                if actual[0] < self.goal_loss[0] and actual[1] < self.goal_loss[1]:
                    print('Goal loss: %s reached; actual loss = %s' % (self.goal_loss, actual))
                    break

            else:
                # summarize the finding
                est, _ = self.surrogate(MODEL[0], [x])
                print('%d>x=%s, f()=%3f, actual=%.3f' % (g+1, x, est, actual))
                # add the data to the dataset
                self.trials.params = np.vstack((self.trials.params, [x]))
                self.trials.scores = np.vstack((self.trials.scores, [[actual]]))
                # update the model
                MODEL[0].fit(self.trials.params, self.trials.scores)

                if actual < self.goal_loss:
                    print('Goal loss: %.3f reached; actual loss = %.3f' % (self.goal_loss, actual))
                    break

            # 11)
            g += 1

        # best result
        if type(self.goal_loss) is list:  # > 1D
            print('Best Result: x=%s\n\ny=%s' % (self.pareto_best['x'], self.pareto_best['y']))
            plt.scatter(opt.trials.scores[:, 0], opt.trials.scores[:, 1])
            plt.scatter(self.pareto['y'][:, 0], self.pareto['y'][:, 1], cmap='Greens')
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.legend(['samples', 'pareto-optimal'])
            plt.show()
        else:
            ix = np.argmin(self.trials.scores)
            print('Best Result: x=%s, y=%.3f' % (self.trials.params[ix], self.trials.scores[ix]))

    def main(self):
        # 1) -> 3)
        self.initial_mogbo()
        if type(self.goal_loss) is list:
            # 4) Compute pareto front
            mapp = self.is_pareto_efficient(self.trials.scores, self.maximize)
            self.pareto['x'] = self.trials.params[mapp, :]
            self.pareto['y'] = self.trials.scores[mapp, :]
            self.pareto_best = self.pareto.copy()
            if self.constraint is not None and type(self.goal_loss) is list:
                self.pareto_constrained(self.pareto['y'])
        # 5 -> 6)
        self.fmin(self.default_n_initial_samples)
        print('Done')



if __name__ == '__main__':
    seed = np.random.RandomState(int(10))
    funct = objective
    eval = 100
    goal_loss = [0, 0]
    constraint = [(0, 2), (0, 2)]
    maximize = False
    domain = {}
    domain['x1'] = [-4, 4]
    domain['x2'] = [-4, 4]

    opt = MOGBO(funct, eval, domain, goal_loss, constraint, seed, maximize)
    opt.main()












