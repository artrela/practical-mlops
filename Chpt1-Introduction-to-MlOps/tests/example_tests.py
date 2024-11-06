import pytest
from src.main import *

def test_add():
    assert torch.equal(scale_tensor(torch.FloatTensor([1., 2., 3.]), 2), torch.FloatTensor([2., 4., 6.]))
    
def test_relu():
    assert torch.equal(relu(torch.FloatTensor([1. , -1, 4.])), torch.FloatTensor([1., 0., 4.]))
    
def test_transpose():
    x = torch.FloatTensor([[1., 1., 1.]])
    assert torch.equal(transpose(x), x.T)