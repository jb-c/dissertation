import torch

class F(torch.nn.Module):
    '''
    Defines the neural network denoted f_{\theta} in our neural CDE model
    '''
    def __init__(self, input_channels, hidden_channels, width = 128):
        '''
        :param input_channels: the number of input channels in the data X.
        :param hidden_channels: the number of channels for z_t. (We use h = 32)
        '''
        #torch.manual_seed(3)
        super(F, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, width)
        self.linear2 = torch.nn.Linear(width, width)
        self.linear3 = torch.nn.Linear(width, input_channels * hidden_channels)

    def forward(self, t, z):
        '''
        :param t: t is normally embedded in the data
        :param z: input to the network & has shape (batch, hidden_channels)
        :return: F(z)
        '''
        z = self.linear1(z)
        z = z.tanh()
        z = self.linear2(z)
        z = z.tanh()
        z = self.linear3(z)

        # A final tanh non-linearity.
        z = z.tanh()

        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z

