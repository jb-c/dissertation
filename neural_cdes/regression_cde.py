import sys
sys.path.append('../')
import torch,torchcde
from neural_cdes.solver import cdeint as my_cdeint
import numpy as np
from neural_cdes.F import F

#------------------------ CDE MODEL ---------------------------
# Remember z_t = z_0 + int_0^t f_\theta(z_s) dX_s
# https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py
# https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
#--------------------------------------------------------------


class CDEModel(torch.nn.Module):
    '''
    A class that packages up our function f into a model we can use and train
    '''
    def __init__(self,input_channels, hidden_channels, output_channels, interpolation_method = 'cubic', sde = False, adjoint = True, dropout_p = 0.5, output_uncertainty_bars = False, **kwargs):
        super(CDEModel, self).__init__()
        #torch.manual_seed(3)
        self.func = F(input_channels,hidden_channels)
        self.interpolation_method = interpolation_method
        self.kwargs = kwargs
        self.adjoint = adjoint
        self.output_uncertainty_bars = output_uncertainty_bars

        if output_uncertainty_bars:
            output_channels = 3*output_channels # Additional 2 ouput per channel for upper & lower uncertainty

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
        elif self.interpolation_method == 'linear':
            X = torchcde.LinearInterpolation(coeffs)

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
        if self.output_uncertainty_bars:
            pred_y = pred_y.reshape(*pred_y.shape[:-1],-1,3)
        return pred_y
