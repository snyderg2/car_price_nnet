import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optimizers

class NeuralNetwork():
    def __init__(self, n_inputs, n_hiddens_list, n_outputs):
        self.n_inputs = n_inputs
        self.n_hiddens_list = n_hiddens_list
        self.n_outputs = n_outputs
        self.n_layers = len(n_hiddens_list)
        self.all_weights, self.Ws = self.make_weights()
        self.all_gradients, self.Gs = self.make_weights()
        self.initialize_weights()
    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, {self.n_hiddens_list}, {self.n_outputs})'
    
    def make_weights(self):
        n_Ws = []
        n_Ws.append((1 + self.n_inputs) * self.n_hiddens_list[0])
        if (len(self.n_hiddens_list) > 1):
            for i in range(len(self.n_hiddens_list)-1):
                n_Ws.append(n_Ws[i]+(1 + self.n_hiddens_list[i]) * self.n_hiddens_list[i+1])
        n_Ws.append(n_Ws[len(n_Ws)-1]+(1 + self.n_hiddens_list[len(self.n_hiddens_list)-1]) * self.n_outputs)
        n_weights = 0
        for i in n_Ws:
            n_weights = n_Ws[len(n_Ws)-1]
        all_weights = np.zeros(n_weights)
        Ws = []
        Ws.append(all_weights[:n_Ws[0]].reshape(1 + self.n_inputs, self.n_hiddens_list[0]))
        if (len(self.n_hiddens_list)!=1):
            for i in range(len(self.n_hiddens_list)-1):
                Ws.append(all_weights[n_Ws[i]:n_Ws[i+1]].reshape(1 + self.n_hiddens_list[i], self.n_hiddens_list[i+1])) 
        Ws.append(all_weights[n_Ws[len(n_Ws)-2]:].reshape(1 + self.n_hiddens_list[len(self.n_hiddens_list)-1], self.n_outputs))
        return all_weights, Ws
    
    def initialize_weights(self):
        self.Ws[0][:] = np.random.uniform(-1, 1, size = (1 + self.n_inputs, self.n_hiddens_list[0])) / np.sqrt(self.n_inputs + 1)
        if (len(self.n_hiddens_list)!=1):
            for i in range(len(self.n_hiddens_list)-1):
                self.Ws[i+1][:] = np.random.uniform(-1, 1, size = (1 + self.n_hiddens_list[i], self.n_hiddens_list[i+1])) / np.sqrt(self.n_hiddens_list[i] + 1)
        self.Ws[len(self.Ws)-1][:] = np.random.uniform(-1, 1, size=(1 + self.n_hiddens_list[len(self.n_hiddens_list)-1], self.n_outputs)) / np.sqrt(self.n_hiddens_list[len(self.n_hiddens_list)-1] + 1)

  
    def train(self, X, T, n_epochs, learning_rate=0, method = 'adam', verbose=True):
        self.stand_params = calc_standardize_parameters(X, T)
        Xst = standardize_X(X, self.stand_params)
        Tst = standardize_T(T, self.stand_params)
        optimizer = optimizers.Optimizers(self.all_weights)
        def error_convert(mse_st):
            if T.shape[1] == 1:
                return np.sqrt(mse_st) * self.stand_params['Tstds'][0]
            else:
                return np.sqrt(mse_st)
        if method == 'sgd':
            self.error_trace = optimizer.sgd(self.mse, self.backward, [Xst, Tst], n_epochs, learning_rate, error_convert_f=error_convert)
        elif method == 'adam':
            self.error_trace = optimizer.adam(self.mse, self.backward, [Xst, Tst], n_epochs, learning_rate, error_convert_f=error_convert)
        elif method == 'scg':
            learning_rate = None
            self.error_trace = optimizer.scg(self.mse, self.backward, [Xst, Tst], n_epochs, learning_rate)
        else:
            print('method must be ''sgd'', ''adam'', or ''scg''.')
    
    def use(self, X, return_hidden_layer_outputs=False):
        Xst = standardize_X(X, self.stand_params)
        Outs = self.forward(Xst) 
        if return_hidden_layer_outputs:
            return unstandardize_T(Outs[len(Outs)-1], self.stand_params), Outs[:-1]
        else:
            return unstandardize_T(Outs[len(Outs)-1], self.stand_params)
    
    def get_error_trace(self):
        return self.error_trace
    
    def forward(self, Xst):
        Z = np.tanh(add_ones(Xst) @ self.Ws[0])
        Outs = []
        Outs.append(Z)
        for i in range(1,len(self.Ws)-1):
            Outs.append(np.tanh(add_ones(Outs[i-1]) @ self.Ws[i]))
        Outs.append(add_ones(Outs[len(Outs)-1]) @ self.Ws[len(self.Ws)-1])
        return Outs

    def backward(self, Xst, Tst):
        n_samples = Xst.shape[0]
        n_outputs = Tst.shape[1]
        Outs = self.forward(Xst)
        gradient_ms = []
        gradient_vs = []
        delta = -2 * (Tst - Outs[len(Outs)-1]) /  (n_samples * n_outputs)
        gradient_w = add_ones(Outs[len(Outs)-2]).T @ delta
        lenH = len(Outs)
        for i in range(2, lenH):
            delta = (delta @ self.Ws[lenH-i+1][1:, :].T) * (1 - Outs[lenH-i] ** 2)
            gradient_vs.append(add_ones(Outs[lenH-i-1]).T @ delta)
        delta = (delta @ self.Ws[1][1:, :].T) * (1 - Outs[0] ** 2)
        self.Gs[0][:] = add_ones(Xst).T @ delta
        for i in reversed(range(len(gradient_vs))):
            self.Gs[len(self.Gs)-i-2][:] = gradient_vs[i]
        self.Gs[len(self.Gs)-1][:] = gradient_w
        return self.all_gradients
    
    def mse(self, Xst, Tst):
        Outs = self.forward(Xst)
        return np.mean((Tst - Outs[len(Outs)-1])**2)


def add_ones(X):
    return np.insert(X, 0, 1, axis=1)

def calc_standardize_parameters(X, T):
    Xmeans = X.mean(axis=0)
    Xstds = X.std(axis=0)
    Tmeans = T.mean(axis=0)
    Tstds = T.std(axis=0)
    return {'Xmeans': Xmeans, 'Xstds': Xstds,
            'Tmeans': Tmeans, 'Tstds': Tstds}

def standardize_X(X, stand_parms):
    return (X - stand_parms['Xmeans']) / stand_parms['Xstds']


def unstandardize_X(Xst, stand_parms):
    return Xst * stand_parms['Xstds'] + stand_parms['Xmeans']


def standardize_T(T, stand_parms):
    return (T - stand_parms['Tmeans']) / stand_parms['Tstds']


def unstandardize_T(Tst, stand_parms):
    return Tst * stand_parms['Tstds'] + stand_parms['Tmeans']



def run(Xtrain, Ttrain, Xtest, Ttest, method, n_epochs, learning_rate, hidden_unit_list=[50, 50, 50, 50, 50]):
    
    # n_samples = 30
    # Xtrain = np.linspace(0., 20.0, n_samples).reshape((n_samples, 1))
    # Ttrain = 0.2 + 0.05 * (Xtrain) + 0.4 * np.sin(Xtrain / 2) + 0.2 * np.random.normal(size=(n_samples, 1))

    # Xtest = Xtrain + 0.1 * np.random.normal(size=(n_samples, 1))
    # Ttest = 0.2 + 0.05 * (Xtest) + 0.4 * np.sin(Xtest / 2) + 0.2 * np.random.normal(size=(n_samples, 1))
    # print(Xtrain)
    n_inputs = Xtrain.shape[1]
    n_hiddens_list = hidden_unit_list
    n_outputs = Ttrain.shape[1]

    nnet = NeuralNetwork(n_inputs, n_hiddens_list, n_outputs)
    nnet.train(Xtrain, Ttrain, n_epochs, learning_rate, method=method, verbose=False)

    def rmse(Y, T):
        error = T - Y
        return np.sqrt(np.mean(error ** 2))

    Ytrain = nnet.use(Xtrain)
    rmse_train = rmse(Ytrain, Ttrain)
    Ytest = nnet.use(Xtest)
    rmse_test = rmse(Ytest, Ttest)

    print(f'Method: {method}, RMSE: Train {rmse_train:.2f} Test {rmse_test:.2f}')

    # plt.figure(1, figsize=(10, 10))
    # plt.clf()

    # n_plot_rows = nnet.n_layers + 1
    # ploti = 0

    # ploti += 1
    # plt.subplot(n_plot_rows, 1, ploti)
    # plt.plot(nnet.get_error_trace())
    # plt.xlabel('Epoch')
    # plt.ylabel('RMSE')

    # ploti += 1
    # plt.subplot(n_plot_rows, 1, ploti)
    # plt.plot(Xtrain, Ttrain, 'o', label='Training Data')
    # plt.plot(Xtest, Ttest, 'o', label='Testing Data')
    # X_for_plot = np.linspace(0, 20, 20).reshape(-1, 1)
    # Y, Zs = nnet.use(X_for_plot, return_hidden_layer_outputs=True)
    # # print(X_for_plot)
    # # print(Y)
    # plt.plot(X_for_plot, Y, label='Neural Net Output')
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # # Y= nnet.use(np.array([year]).reshape(-1, 1))
    # # print(Y)
    # for layeri in range(nnet.n_layers - 2, -1, -1):
    #     ploti += 1
    #     plt.subplot(n_plot_rows, 1, ploti)
    #     plt.plot(X_for_plot, Zs[layeri])
    #     plt.xlabel('X')
    #     plt.ylabel(f'Outputs from Layer {layeri}')
        
    return nnet
# n_samples=30
# Xtrain = np.linspace(0., 20.0, n_samples).reshape((n_samples, 1))
# print(Xtrain)
# run('sgd', 4000, 0.1)
