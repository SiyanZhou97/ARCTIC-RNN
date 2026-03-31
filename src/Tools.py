try:
    import cupy as np
    a=np.array([0])
except:
    import numpy as np

#modified tanh parameters
a=500
r0 = 0.0001
r1 = 4

def phi(x, name='tanh'):
    """
    activation function
    """
    if name == 'tanh':
        return np.tanh(x)
    if name == 'modifiedtanh':
        output=np.zeros(x.shape)
        output[x<=0]= r0 * np.tanh(x[x<=0]/(1-a*x[x<=0]) / r0 / r1)
        output[x>0]=np.tanh(x[x>0] / r1)
        return output

def reverse_phi(x, name='tanh'):
    if name == 'tanh':
        h = 0.5 * (np.log(1 + x + 1e-14) - np.log(1 - x + 1e-14))
        h[np.logical_and(np.isnan(h), x > 0)] = 0.5 * (np.log(1 + 1 + 1e-14) - np.log(1 - 1 + 1e-14))
        h[np.logical_and(np.isnan(h), x <= 0)] = 0.5 * (np.log(1 - 1 + 1e-14) - np.log(1 + 1 + 1e-14))
        return h
    if name == 'modifiedtanh':
        output=np.zeros(x.shape)
        y=reverse_phi(x[x<=0] / r0, 'tanh') * r0 * r1
        output[x<=0]= y/(1+a*y)
        output[x>0]=reverse_phi(x[x>0], 'tanh') * r1 
        return output

def derivative(x, name='logistic'):
    if name == 'tanh':
        return 1 - np.tanh(x) ** 2
    if name == 'modifiedtanh':
        output=np.zeros(x.shape)
        output[x<=0]= 1/((1-a*x[x<=0])**2) / r1 * derivative(x[x<=0]/(1-a*x[x<=0])/r0/r1, 'tanh')
        output[x>0]=1 / r1 * derivative(x[x>0]/r1, 'tanh')
        return output
