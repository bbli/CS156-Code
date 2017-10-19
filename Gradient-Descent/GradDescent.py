import numpy as np
from sympy import Matrix
from sympy.abc import x,y
from sympy import diff
from sympy import exp
from sympy import lambdify


def test_gradient():
    '''
    This function calculates the 2-D gradient of the given error function.
    '''
    part_x=diff((x*exp(y)-2*y*exp(-x))**2,x)
    part_y=diff((x*exp(y)-2*y*exp(-x))**2,y)
    
    return part_x,part_y
#####################################################################
def gradient(u,v):
    '''
    This function calculates the 2-D gradient of the given error function given the state
    '''
    assert type(u)==np.float128
    assert type(v)==np.float128
    part_x=diff((x*exp(y)-2*y*exp(-x))**2,x)
    part_y=diff((x*exp(y)-2*y*exp(-x))**2,y)
    px=lambdify([x,y],part_x)
    py=lambdify([x,y],part_y)
    grad_x=px(u,v)
    grad_y=py(u,v)
    return np.array([grad_x,grad_y],dtype=np.float128)

def error(u,v):
    '''
    This function returns the error value given the state
    '''
    assert type(u)==np.float128
    assert type(v)==np.float128
    return (u*np.exp(v)-2*v*np.exp(-u))**2

def grad_descent():
    state=np.array([1,1],dtype=np.float128)
    count=0
    error_value=np.array([1],dtype=np.float128)
    while error_value[0]>1e-14 :
        count=count+1
        step=gradient(*state)
        state=state-0.1*step
        error_value[0]=error(*state)
    return state,count

def coord_descent():
    state=np.array([1,1],dtype=np.float128)
    count=0
    error_value=np.array([1],dtype=np.float128)
    while count<15:
        count=count+1

        step1x,step1y=gradient(*state)
        state=state-0.1*np.array([step1x,0],dtype=np.float128)
        step2x,step2y=gradient(*state)
        state=state-0.1*np.array([0,step2y],dtype=np.float128)
        error_value[0]=error(*state)
    return error_value    
