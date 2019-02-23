#!/usr/bin/env python3
# -*- coding: utf-8 -*-

SET_NICE_INITIAL_WEIGHTS=True
EPOCHS=500 #0 #1
LR=1e-4

import pylab

import numpy as np

import math

from scipy.special import logit

from matplotlib.pyplot import savefig

import torch.nn.functional as F



import torch    

c=-20
x=np.arange(1, 120, 0.1, dtype=np.float32).reshape(-1,1)
y=(0.01*(x+c)**3)-(1.25*(x+c)**2)+(7*(x+c))+2500

# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = len(x), 1, 1, 1


z = torch.randn(N, D_in)

x = torch.from_numpy(x)
y = torch.from_numpy(y)



class MyModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        negative_slope=0.0
        
        h_relu1 = F.leaky_relu(self.linear1(x), negative_slope=negative_slope)
        h_relu2 = F.leaky_relu(self.linear2(h_relu1), negative_slope=negative_slope)
        h_relu3 = F.leaky_relu(self.linear3(h_relu2), negative_slope=negative_slope)
        y_pred = self.linear4(h_relu3)
        return y_pred

# Construct our model by instantiating the class defined above
model = MyModel(D_in, H, D_out)


my_weights=[
            (1.0, -35.0),
            (-1.8, 100),
            (20.0, 00),
            (1, 400),
            ]

if SET_NICE_INITIAL_WEIGHTS:
    from torch import nn
    i=0
    model.linear1.weight=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][0]]], dtype=np.float32 )))
    model.linear1.bias=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][1]]], dtype=np.float32 )))
    i+=1
    model.linear2.weight=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][0]]], dtype=np.float32 )))
    model.linear2.bias=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][1]]], dtype=np.float32 )))
    i+=1
    model.linear3.weight=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][0]]], dtype=np.float32 )))
    model.linear3.bias=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][1]]], dtype=np.float32 )))
    i+=1
    model.linear4.weight=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][0]]], dtype=np.float32 )))
    model.linear4.bias=nn.Parameter(torch.from_numpy(np.array([[my_weights[i][1]]], dtype=np.float32 )))


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
for t in range(EPOCHS):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_pred = model(x)





print("w1: ", model.linear1.weight.detach().numpy().reshape(-1))
print("b1: ", model.linear1.bias.detach().numpy().reshape(-1))
print("w2: ", model.linear2.weight.detach().numpy().reshape(-1))
print("b2: ", model.linear2.bias.detach().numpy().reshape(-1))
print("w3: ", model.linear3.weight.detach().numpy().reshape(-1))
print("b3: ", model.linear3.bias.detach().numpy().reshape(-1))
print("w4: ", model.linear4.weight.detach().numpy().reshape(-1))
print("b4: ", model.linear4.bias.detach().numpy().reshape(-1))




x=x.numpy()
y=y.numpy()
y_pred=y_pred.detach().numpy()

y_pred2=x
for j,(w,b) in enumerate(my_weights):
    if j<len(my_weights)-1:
        y_pred2=np.clip(y_pred2*w + b, 0, None)
    else:
        y_pred2=y_pred2*w + b

pylab.plot(x,y, label='ground truth')
pylab.plot(x, y_pred, label='y_pred')
pylab.plot(x, y_pred2, label='desired output')
pylab.legend(loc='upper right')
pylab.title("title")
pylab.savefig("simpleNN.png")
pylab.show()
