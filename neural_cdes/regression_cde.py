import sys
sys.path.append('../')
import torch,torchcde
from neural_cdes.solver import cdeint as my_cdeint
import numpy as np

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
        torch.manual_seed(3)
        super(F, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, input_channels * hidden_channels)

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


class CDEModel(torch.nn.Module):
    '''
    A class that packages up our function f into a model we can use and train
    '''
    def __init__(self,input_channels, hidden_channels, output_channels, interpolation_method = 'cubic', sde = False, adjoint = True, dropout_p = 0.5, **kwargs):
        super(CDEModel, self).__init__()
        torch.manual_seed(3)
        self.func = F(input_channels,hidden_channels)
        self.interpolation_method = interpolation_method
        self.kwargs = kwargs
        self.adjoint = adjoint

        # Init the initial and final transformations
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.output_channels = output_channels

        self.is_sde = sde
        self.backend = 'torchsde' if sde else 'torchdiffeq'
        self.dropout_p = dropout_p

    def forward(self, coeffs, return_path = False, num_stochastic_samples = None):
        '''
        Performs the integral (via adjoint method)
        :param coeffs: The interpolation coefficients, obtained by fitting a method to the data
        :param return_path: True if we want the full path output, false else
        :return:
        '''
        coeffs_initial_size = coeffs.shape # Make a record of the input size, before we do any reshaping ect

        # 0. Get callable data interpolation function
        if (self.backend == 'torchsde') and (num_stochastic_samples is not None):
            if coeffs.ndim == 2: # (time, feature)
                coeffs = torch.unsqueeze(coeffs, 0)  # so now have (new_axis, time, feature)
            if coeffs.ndim == 3: # (batch, time, feature)
                coeffs = torch.repeat_interleave(coeffs, num_stochastic_samples, 0) # Repeat along batch dimension
            else:
                raise ValueError(f"The coeffs you passed in has {coeffs.ndim} dims, it needs to have 2 or 3")

        if self.interpolation_method == 'cubic':
            X = torchcde.CubicSpline(coeffs)

        # 1. Get the initial hidden state
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        # 2. Perform the integral, ie actually solve the CDE
        t = X.interval if (not return_path) else torch.arange(X.interval[0],X.interval[1]) # Either start and end times, or all times
        zt = my_cdeint(X=X, func=self.func, z0=z0, t=t, backend = self.backend, adjoint=self.adjoint, **self.kwargs)

        if (not return_path):
            zt = zt[..., -1, :]  # get the terminal value of the CDE

        # 3. Get predicted value, from final hidden state
        pred_y = self.readout(zt)

        # Added a final non-linearity here to squish output to [0,1]
        #################################################
        pred_y = pred_y.sigmoid()
        #################################################

        # 4. Reshape if needed
        if (self.backend == 'torchsde') and (num_stochastic_samples is not None):
            pred_y = pred_y.reshape(-1,num_stochastic_samples,*pred_y.shape[1:])
        return pred_y








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