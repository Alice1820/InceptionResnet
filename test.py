import torch
from inceptionv4.pytorch_load import inceptionv4
from inceptionresnetv2.pytorch_load import inceptionresnetv2

# net = inceptionv4()
net = inceptionresnetv2()
input = torch.autograd.Variable(torch.ones(1,3,299,299))
output = net.forward(input)
print (output)
