import torch
import torchcde

#------------------------ CDE MODEL ---------------------------
# Remember z_t = z_0 + int_0^t f_\theta(z_s) dX_s
# https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py
# https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
#--------------------------------------------------------------

class F(torch.nn.Module):
    '''
    Defines the neural network denoted f_\theta in our neural CDE model
    '''
    def __init__(self, input_channels, hidden_channels):
        '''
        :param input_channels: the number of input channels in the data X. (Determined by the data.)
        :param hidden_channels: the number of channels for z_t. (Determined by you!)
        '''
        super(F, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, input_channels * hidden_channels)

        self.dropout = torch.nn.Dropout(0.25)
    def forward(self, t, z):
        '''
        :param t: t is usually embedded in the data (if at all) but we can use it explicitly here also
        :param z: input to the network & has shape (batch, hidden_channels)
        :return: F(z)
        '''
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)

        # Tip from authors - best results tend to be obtained by adding a final tanh nonlinearity.
        z = z.tanh()

        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDEModel(torch.nn.Module):
    '''
    A class that packages up our function f into a model we can use and train
    '''
    def __init__(self,input_channels, hidden_channels, output_channels, interpolation_method = 'cubic'):
        super(NeuralCDEModel, self).__init__()
        self.func = F(input_channels,hidden_channels)
        self.interpolation_method = interpolation_method


        # Init the initial and final transformations
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.output_channels = output_channels

    def forward(self, coeffs):
        '''
        Performs the integral (via adjoint method)
        :param coeffs: The interpolation coefficients, obtained by fitting a method to the data
        :return:
        '''
        # 0. Get callable data interpolation function
        if self.interpolation_method == 'cubic':
            X = torchcde.CubicSpline(coeffs)

        # 1. Get the initial hidden state
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        # 2. Perform the integral, ie actually solve the CDE
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
        zT = zt[..., -1, :]  # get the terminal value of the CDE [NB the cdeint returns both initial & final values]

        # 3. Get predicted value, from final hidden state
        pred_y = self.readout(zT)
        return pred_y

    def forward_return_all_times(self, coeffs):
        '''
        Performs the integral (via adjoint method) but returns the trajectory of the end result
        :param coeffs: The interpolation coefficients, obtained by fitting a method to the data
        :return:
        '''
        # 0. Get callable data interpolation function
        if self.interpolation_method == 'cubic':
            X = torchcde.CubicSpline(coeffs)

        # 1. Get the initial hidden state
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        # 2. Perform the integral, ie actually solve the CDE with all the time steps
        t = torch.linspace(X.interval[0],X.interval[1],int(X.interval[1]-X.interval[0]+1))
        zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=t)

        pred_y_vector = self.readout(zt)
        return pred_y_vector






'''
  class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.initial = torch.nn.Linear(input_channels, hidden_channels)
            self.func = F()
            self.readout = torch.nn.Linear(hidden_channels, output_channels)

        def forward(self, coeffs):
            X = torchcde.CubicSpline(coeffs)
            X0 = X.evaluate(X.interval[0])
            z0 = self.initial(X0)
            zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval)
            zT = zt[..., -1, :]  # get the terminal value of the CDE
            return self.readout(zT)
'''