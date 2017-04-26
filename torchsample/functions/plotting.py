"""
Live plotting of various metrics
"""

import torch
from torch.autograd import Variable
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from functools import wraps

def plot(bokeh_fig):
  def decorator(train_fn):
    """
    Takes in a fitting function that performs one step and outputs
    two values that correspond to an x and y on a plot. Generally
    x will be the training step, and y will be the loss or some other
    metric.
    """
    @wraps(train_fn)
    def wrapper(*fn_args, **fn_kwargs):
      ds = bokeh_fig.data_source
      new_data = {}
      new_x, new_y = train_fn(*fn_args, **fn_kwargs)
      new_data['x'] = ds.data['x'] + [new_x]
      new_data['y'] = ds.data['y'] + [new_y]
      ds.data = new_data
    return wrapper
  return decorator