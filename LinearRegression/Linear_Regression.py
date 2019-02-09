import numpy as np
import matplotlib.pyplot as plt


class Linear_Regression():
   def __init__(self):
       self.W = None
       self.bias = None
       self.learning_rate = None
       self.n_iters = None