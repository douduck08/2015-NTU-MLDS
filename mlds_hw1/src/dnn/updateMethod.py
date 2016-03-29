import math
import theano
import theano.tensor as T
import numpy
import globalParam

# Momentum        
def Momentum(grads, params, m):
    if(globalParam.flag):
        globalParam.velocitys = [ -globalParam.lr * grad for grad in grads ]
        globalParam.flag = False
    else:
        globalParam.velocitys = [ m * velocity - globalParam.lr * (1 - m) * grad for velocity, grad in zip(globalParam.velocitys, grads) ]
    paramsUpdate = [ (param, param + velocity) for param, velocity in zip(params, globalParam.velocitys) ]
    return paramsUpdate

# RMSProp        
def RMSProp(grads, params):
    alpha = 0.9
    epsilon = 1e-6
    if(globalParam.flag):
        globalParam.sigmaSqrs = [ g * g for g in grads ]
        globalParam.flag = False
    else:
        globalParam.sigmaSqrs = [ ( ( alpha * s ) + ( (1 - alpha) * (g * g) ) ) for s, g in zip(globalParam.sigmaSqrs, grads) ]
    paramsUpdate = [( p, p - ( globalParam.lr * g ) / ( T.sqrt(s) + epsilon ) ) for p, g, s in zip(params, grads, globalParam.sigmaSqrs)]
    return paramsUpdate

# Adagrad
def Adagrad(grads, params):
    epsilon = 1e-6
    if(globalParam.flag):
        globalParam.gradSqrs = [ g * g for g in grads ]
        globalParam.flag = False
    else:
        globalParam.gradSqrs = [ s + g * g for s, g in globalParam.gradSqrs, grads ]
    paramsUpdate = [ (p, p - ( globalParam.lr * g ) / (T.sqrt(s) + epsilon) ) for p, g, s in zip(params, grads, globalParam.gradSqrs) ]
    return paramsUpdate
