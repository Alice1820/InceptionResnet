import torch
from inceptionv4.pytorch_load import inceptionv4
net = inceptionv4()
input = torch.autograd.Variable(torch.ones(1,3,299,299))
output = net.forward(input)
