import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
from catboost import CatBoostRegressor as cat
from django.shortcuts import render
from plotly.offline import plot
from torch import nn
from plotly.offline import plot
from plotly.graph_objs import Scatter


class LSTM(nn.Module):
    r"""
            Model class for LSTM module using torch.nn.LSTM module
            This will be trained for time-series forecasting.
            """
    def __init__(self, input_dim = 2, hidden_dim = 100, num_layers = 2, output_dim = 2 ) :
        """
        -input_dim =     Input dimension for LSTM network.
                        Default 2 (For Speed and Direction)
        -hidden_dim =     Hidden dimension for LSTM network.
                        Default 100.
        -output_dim =     Output dimension for LSTM network.
                        Default 2 (For Speed and Direction).
        -num)layers =     No. of iteration that will be performed on each example.
        -device =         Torch device for storing model class and related variables.
                        Default device will be enabled by system according to availability.
        """
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Number of hidden layers
        
        self.num_layers = num_layers
        self.device = device
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
def get_model(checkpoint = 'weights.pth'):
    r"""
    -checkpoint :     Path to model's state_dict(weights)
                    Default - 'weights.pth'
    """
    model = LSTM(input_dim = 2, hidden_dim = 100, num_layers = 2, output_dim = 2 )
    if checkpoint is not None:
        try:
            model.load_state_dict(torch.load(checkpoint))
            return model
        except FileNotFoundError:
            print(f'Please give a valid path to the weigths.\nFile {checkpoint} does not exist')
            return 0
        except:
            print('Unknown exception happened\nExit....')
            return 0


class Predictor(nn.Module):
    r"""
    This class instantiates an model instance and predicts upcoming speed and direction.
    """
    def __init__(self, model):
        r"""
        -model =     Model object which will forecast speed and direction.
        """
        super(Predictor, self).__init__()
        self.lstm = model.lstm
        self.fc = model.fc
        self.device = next(model.parameters()).device
        self.model = model.cpu() 
    
    def predict(self, x, steps:int = 10, reduce:bool = True):
        r"""
        -x = torch.tensor
        -steps =     No. of future predictions to be made.
                    int
                    Default 10.
        -reduce =    Whether to reduce inputs
                    (Divide by constants -> speed(25)
                                         -> direction(360))
                                         
        """
        try:
            x = x.view(1, -1, 2)
        except:
            print('Invalid input')
            return 
        fc = self.model.fc
        speed = []
        direction = []
        h_0 = torch.zeros(self.model.num_layers, x.size(0), self.model.hidden_dim).to(self.device)
        c_0 = h_0
        if reduce:
            x[:, :, 0] /= 25
            x[:, :, 1] /= 360 
        for i in range(steps):
            if i == 0:
                out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
            else:
                out = out.unsqueeze(1)
                out, (h_n, c_n) = self.lstm(out, (h_n, c_n))
            out = self.fc(out[:, -1, :])
            speed.append(out.cpu()[0, 0]*25)
            direction.append(out.cpu()[0, 1]*360)
        return speed, direction

class Tree(object):
    def __init__(self, checkpoint = 'regressor.cbm'):
        r"""
        Tree class which will predict energy output from wind speed and direction.
        -checkpoint =     Path for saved model.
                        type:     str
                        Default: 'regressor.cbm'
        """
        self.model = cat()
        if checkpoint is not None:
            try:
                self.model.load_model(checkpoint)
            except FileNotFoundError:
                print(f'Please give a valid path to the weigths.\nFile {checkpoint} does not exist')
            
            except:
                print('Unknown exception happened\nExit....')
    def forward(self, speed, direction):
        df = pd.DataFrame({'Wind Speed (m/s)':speed, 'Wind Direction (°)': direction})
        power = self.model.predict(df).tolist()
        return power


def get_speed():
    from pyowm.owm import OWM
    owm = OWM('8f1b9a3225495a9c8a89cb7ff7848c08') 
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place('New Delhi,IN')
    wind_dict_in_meters_per_sec = observation.weather.wind()   # Default unit: 'meters_sec'
    speed = wind_dict_in_meters_per_sec['speed']
    speed = float(speed)
    return speed

    
    
    
    
def graph(speed: float = 10.0, direction:float = 180.0, n_steps:int = 40):
    r"""
    -speed = initial wind speed(gust) in m/s.
    -direction = initial Wind Direction (°).
    -n_steps = no of predictions to be made.
    """
    from plotly.offline import plot
    import plotly.graph_objs as go

    fig = go.Figure()

    lstm = get_model()
    predictor = Predictor(lstm)
    x = torch.Tensor([speed/25, direction/360])
    speed, direction = predictor.predict(x, steps = n_steps)
    tree = Tree()
    power = tree.forward(speed, direction)
    if type(power) != list:
        power = [ i for i in power]
    scatter = go.Scatter(x=np.arange(len(power)), y=power,
                         mode='lines', name='Power Forecast',
                         opacity=1.0, marker_color='red') 
    
    fig.add_trace(scatter)
    fig.update_layout(    title={
                                'text': f'Power output(kWh) forecast for next {n_steps//6} hrs',
                                'y':0.9,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'}, 
                        xaxis_title="Time",
                        yaxis_title="Power(kWh)",
                        font=dict(
                            family="Roboto, monospace",
                            size=10,
                            color="#000000"
                        )
                    )

    plt_div = plot(fig, output_type='div')

    return plt_div, max(power), np.argmax(power)
  
    
    
########################################################################################
#####This function will accept input from webpage as int and return respective grap#####
########################################################################################
def in_out(nsteps: int = 10):
    speed = get_speed()
    direction = 180.0
    grph = graph(speed, direction, n_steps = nsteps)
    return grph
