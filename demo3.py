import torch
import numpy
a = numpy.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
print(a)
t = torch.from_numpy(a)
e = t.view(1,9,1)
print(t)
print(e)