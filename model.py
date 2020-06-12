import torch 
from torch import nn
import numpy as np
import pandas as pd
import os
from catboost import CatBoostRegressor as cat
from plotly.offline import plot
import plotly.graph_objs as go
from django.shortcuts import render


class LSTM(nn.Module):
    def __init__(self, input_dim = 2, 
                 hidden_dim = 100, num_layers = 2, 
                 output_dim = 2 , 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

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
    
    def predict()

def get_model(checkpoint = 'weights.pth'):
    r"""
    -checkpoint : Path to model's state_dict(weights)
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
    
    def __init__(self, model):
        super(predictor, self).__init__()
        self.lstm = model.lstm
        self.fc = model.fc
        self.device = next(model.parameters()).device
        self.model = model.cpu() 
    
    def predict(self, x, steps:int = 10, reduce:bool = True):
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
    observation = mgr.weather_at_place('Tokyo,JP')
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
    lstm = get_model()
    predictor = Predictor(lstm)
    x = torch.Tensor([speed/25, direction/360])
    speed, direction = predictor.predict(x, steps = n_steps)
    tree = Tree()
    power = tree.forward(speed, direction)
    if type(power) != list:
        power = [ i for i in power]
    plot_div = plot([Scatter(x=np.arange(len(power)), y=power,
                        mode='lines', 
                        opacity=0.8, marker_color='red')],
               output_type='div')
 
    
    
    return plot_div
  
    
    
########################################################################################
#####This function will accept input from webpage as int and return respective grap#####
########################################################################################
def in_out(nsteps: int = 10):
    speed = get_speed()
    direction = 180.0
    grph = graph(speed, direction, n_steps = nsteps)
    return grph


    
    