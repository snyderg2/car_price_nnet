import copy
import numpy as np
import time
import math
import sys

floatPrecision = sys.float_info.epsilon

class Optimizers():

    def __init__(self, all_weights):
        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''
        
        self.all_weights = all_weights

        self.scg_initialized = False

        # The following initializations are only used by adam.
        # Only initializing m, v, beta1t and beta2t here allows multiple calls to adam to handle training
        # with multiple subsets (batches) of training data.
        self.mt = np.zeros_like(all_weights)
        self.vt = np.zeros_like(all_weights)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1t = 1
        self.beta2t = 1


    def sgd(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001, save_wtrace=False,
            verbose=True, error_convert_f=None):
        '''
error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
            with respect to each weight.
error_convert_f: function that converts the standardized error from error_f to original T units.
        '''

        error_trace = []
        weights_trace = []
        if save_wtrace:
            weights_trace = [self.all_weights.copy()]
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(*fargs)
            grad = gradient_f(*fargs)

            # Update all weights using -= to modify their values in-place.
            self.all_weights -= learning_rate * grad

            if error_convert_f:
                error = error_convert_f(error)
            error_trace.append(error)

            if save_wtrace:
                weights_trace.append(self.all_weights.copy())

            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):
                error_scalar = np.asscalar(error) if isinstance(error, np.ndarray) else error
                print(f'sgd: Epoch {epoch+1:d} Error={error_scalar:.5f}')

        return (error_trace, np.array(weights_trace)) if save_wtrace else error_trace

    def adam(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=0.001,
             save_wtrace=False, verbose=True, error_convert_f=None):
        '''
error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
            with respect to each weight.
error_convert_f: function that converts the standardized error from error_f to original T units.
        '''

        alpha = learning_rate  # learning rate called alpha in original paper on adam
        epsilon = 1e-8
        error_trace = []
        weights_trace = []
        if save_wtrace:
            weights_trace = [self.all_weights.copy()]
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(*fargs)
            grad = gradient_f(*fargs)

            self.mt[:] = self.beta1 * self.mt + (1 - self.beta1) * grad
            self.vt[:] = self.beta2 * self.vt + (1 - self.beta2) * grad * grad
            self.beta1t *= self.beta1
            self.beta2t *= self.beta2

            m_hat = self.mt / (1 - self.beta1t)
            v_hat = self.vt / (1 - self.beta2t)

            # Update all weights using -= to modify their values in-place.
            self.all_weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    
            if error_convert_f:
                error = error_convert_f(error)
            error_trace.append(error)

            if save_wtrace:
                weights_trace.append(self.all_weights.copy())

            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):
                error_scalar = np.asscalar(error) if isinstance(error, np.ndarray) else error
                print(f'Adam: Epoch {epoch+1:d} Error={error_scalar:.5f}')

        return (error_trace, np.array(weights_trace)) if save_wtrace else error_trace

    def scg(self, error_f, gradient_f, fargs=[], n_epochs=100, learning_rate=None,
            save_wtrace=False, error_convert_f=lambda x: x, verbose=True):
        '''learning_rate not used in scg'''

        if not self.scg_initialized:
            shape = self.all_weights.shape
            self.w_new = np.zeros(shape)
            self.w_temp = np.zeros(shape)
            self.g_new = np.zeros(shape)
            self.g_old = np.zeros(shape)
            self.g_smallstep = np.zeros(shape)
            self.search_dir = np.zeros(shape)
            self.scg_initialized = True

        sigma0 = 1.0e-6
        fold = error_f(*fargs)
        error = fold
        self.g_new[:] = gradient_f(*fargs)
        # print('scg g\n', self.g_new)
        self.g_old[:] = copy.deepcopy(self.g_new)
        self.search_dir[:] = -self.g_new
        success = True				# Force calculation of directional derivs.
        nsuccess = 0				# nsuccess counts number of successes.
        beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
        betamin = 1.0e-15 			# Lower bound on scale.
        betamax = 1.0e20			# Upper bound on scale.
        nvars = len(self.all_weights)
        iteration = 1				# j counts number of iterations
        error_trace = []
        weights_trace = []
        if save_wtrace:
            weights_trace = [self.all_weights.copy()]
        
        error_trace.append(error_convert_f(error))

        thisIteration = 1
        startTime = time.time()
        startTimeLastVerbose = startTime

        # Main optimization loop.
        while thisIteration <= n_epochs:

            # Calculate first and second directional derivatives.
            if success:
                mu = self.search_dir @ self.g_new
                if mu >= 0:
                    self.search_dir[:] = - self.g_new
                    mu = self.search_dir.T @ self.g_new
                kappa = self.search_dir.T @ self.search_dir
                if math.isnan(kappa):
                    print('kappa', kappa)

                if kappa < floatPrecision:
                    return (error_trace, np.array(weights_trace)) if save_wtrace else error_trace

                sigma = sigma0 / math.sqrt(kappa)

                self.w_temp[:] = self.all_weights
                self.all_weights += sigma * self.search_dir
                # error_f(*fargs)  # forward pass through model for intermediate variable values for gradient
                # gradient_f assumed to do forward pass to save hidden layer outputs
                self.g_smallstep[:] = gradient_f(*fargs)
                # print('scg smallstep g\n', self.g_smallstep)
                self.all_weights[:] = self.w_temp

                theta = self.search_dir @ (self.g_smallstep - self.g_new) / sigma
                if math.isnan(theta):
                    print('theta', theta, 'sigma', sigma, 'search_dir[0]', self.search_dir[0],
                          'g_smallstep[0]', self.g_smallstep[0]) #, 'gradnew[0]', gradnew[0])

            ## Increase effective curvature and evaluate step size alpha.

            delta = theta + beta * kappa
            # if math.isnan(scalarv(delta)):
            if math.isnan(delta):
                print('delta is NaN', 'theta', theta, 'beta', beta, 'kappa', kappa)
            elif delta <= 0:
                delta = beta * kappa
                beta = beta - theta / kappa

            if delta == 0:
                success = False
                fnow = fold
            else:
                alpha = -mu / delta
                ## Calculate the comparison ratio Delta
                self.w_temp[:] = self.all_weights
                self.all_weights += alpha * self.search_dir
                fnew = error_f(*fargs)
                Delta = 2 * (fnew - fold) / (alpha * mu)
                if not math.isnan(Delta) and Delta  >= 0:
                    success = True
                    nsuccess += 1
                    fnow = fnew
                else:
                    success = False
                    fnow = fold
                    self.all_weights[:] = self.w_temp

            iterationsPerPrint = math.ceil(n_epochs/10)

            # print('fnow', fnow, 'converted', error_convert_f(fnow))
            error_trace.append(error_convert_f(fnow))

            if verbose and ((thisIteration + 1) % max(1, iterationsPerPrint) == 0):
                error = error_trace[-1]
                error_scalar = np.asscalar(error) if isinstance(error, np.ndarray) else error
                print(f'SCG: Epoch {thisIteration:d} Error={error_scalar:.5f}')
            
            # if verbose and thisIteration % max(1, iterationsPerPrint) == 0:
            #     print('SCG: Iteration {:d} ObjectiveF={:.5f} Scale={:.3e} Seconds={:.3f}'.format(iteration,
            #                     error_convert_f(fnow), beta, (time.time()-startTimeLastVerbose)))


                startTimeLastVerbose = time.time()

            if save_wtrace:
                weights_trace.append(self.all_weights.copy())

            if success:

                fold = fnew
                self.g_old[:] = self.g_new
                self.g_new[:] = gradient_f(*fargs)
                # print('scg gnew\n', self.g_new)

                # If the gradient is zero then we are done.
                gg = self.g_new @ self.g_new  # dot(gradnew, gradnew)
                if gg == 0:
                    return (error_trace, np.array(weights_trace)) if save_wtrace else error_trace

            if math.isnan(Delta) or Delta < 0.25:
                beta = min(4.0 * beta, betamax)
            elif Delta > 0.75:
                beta = max(0.5 * beta, betamin)

            # Update search direction using Polak-Ribiere formula, or re-start
            # in direction of negative gradient after nparams steps.
            if nsuccess == nvars:
                self.search_dir[:] = -self.g_new
                nsuccess = 0
            elif success:
                gamma = (self.g_old - self.g_new) @ (self.g_new / mu)
                #self.search_dir[:] = gamma * self.search_dir - self.g_new
                self.search_dir *= gamma
                self.search_dir -= self.g_new

            thisIteration += 1
            iteration += 1

            # If we get here, then we haven't terminated in the given number of
            # iterations.

        return (error_trace, np.array(weights_trace)) if save_wtrace else error_trace


if __name__ == '__main__':

    def parabola(wmin):
        return ((w - wmin) ** 2)[0]

    def parabola_gradient(wmin):
        return 2 * (w - wmin)

    wmin = 5

    print()
    w = np.array([0.0])
    optimizer = Optimizers(w)
    optimizer.sgd(parabola, parabola_gradient, [wmin],
                  n_epochs=50, learning_rate=0.1)
    print(f'sgd: Minimum of parabola is at {wmin}. Value found is {w}')

    print()
    w = np.array([0.0])
    optimizer = Optimizers(w)
    optimizer.adam(parabola, parabola_gradient, [wmin],
                   n_epochs=50, learning_rate=0.1)
    print(f'adam: Minimum of parabola is at {wmin}. Value found is {w}')

    print()
    w = np.array([0.0])
    optimizer = Optimizers(w)
    optimizer.scg(parabola, parabola_gradient, [wmin],
                  n_epochs=50, learning_rate=0.1)
    print(f'scg: Minimum of parabola is at {wmin}. Value found is {w}')

    
