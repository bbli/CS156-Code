import sympy as sp
import numpy as np

def random_points_f(number_of_points):
    '''
    This function returns random points on the plane in the form of one numpy array
    '''
    return 2*np.random.random(size=(2,number_of_points))-1

#############################################################
def my_line(x):
    return 1.2*x+0.3
    
def points_assigner(random_points,random_line):
    '''
    This function takes a set of random points and a line an return a numpy array with the classfications 1 or -1
    '''
    random_points_dimensions,random_points_number=random_points.shape
    assert random_points_dimensions==2, "Input is not 2d!"
    ## To take this assert out, I should write a proper test using my pipeline
    result_vector=np.zeros(random_points_number)
    
    x_values=random_points[0,:]
    y_values=random_points[1,:]
    result_vector[:]= (random_line(x_values)<y_values)
    np.place(result_vector,result_vector==0,-1)
    return result_vector

############################################################

def input_matrix_f(random_points):
    '''
    Returns the input matrix with 1's appended to each row
    '''
    random_points_dimensions,random_points_number=random_points.shape
    input_matrix=np.ones((3,random_points_number))
    input_matrix[1:3,:]=random_points
    input_matrix[0,:]=1

    return input_matrix.T    
####################################################################
w1, w2, w3=sp.symbols('w1 w2 w3')

def s_innerproduct(x_vec):
    '''
    This function returns the symbolic inner product with the weight_vector. Assumes x_vec is of length 3
    '''
    return w1*x_vec[0]+w2*x_vec[1]+w3*x_vec[2]

def ln_creator(x_vec, y_value):
    '''
    This function creates the ln sympy function from the x vec and the y value
    '''
    s=s_innerproduct(x_vec)
    return sp.log(1+sp.exp(-1*y_value*s))

def gradient_list_f(input_matrix,result_vector):
    '''
    This function returns a list of the gradient functions for each of the 100 datapoints
    '''
    gradient_list=[ 0 for i in range(len(result_vector)) ]
    for i in range(len(result_vector)):
        error_func=ln_creator(input_matrix[i],result_vector[i])
        grad_error_w1=sp.diff(error_func,w1)
        grad_error_w2=sp.diff(error_func,w2)
        grad_error_w3=sp.diff(error_func,w3)
        grad_error=[grad_error_w1, grad_error_w2, grad_error_w3]
        gradient_list[i]=grad_error
    return gradient_list
##############################################################
def error(old_state,new_state):
    '''
    This function returns the norm of the two's state difference vector
    '''
    diff=old_state-new_state
    return np.linalg.norm(diff)

def eval_gradient(gradient, state):
    '''
    This function takes a gradient in symbolic form and evaluates it for a state. Returns a numpy numerical array of length 3
    '''
    f1=sp.lambdify([w1,w2,w3],gradient[0])
    f2=sp.lambdify([w1,w2,w3],gradient[1])
    f3=sp.lambdify([w1,w2,w3],gradient[2])
    grad_1=f1(*state)
    grad_2=f2(*state)
    grad_3=f3(*state)
    return np.array([grad_1, grad_2, grad_3], dtype=np.float128)

def LogRegression(gradient_list):
    '''
    Returns "Logistic Regression weight vector" and "epoch count". This function assumes there are 100 training points
    '''
    ## Initalizing the state vector.
    old_state=np.array([1,0,0],dtype=np.float128)
    new_state=np.array([0,0,0],dtype=np.float128)
    count=0

    while error(old_state,new_state)>0.01:
        old_state=new_state
        count=count+1
        permutation_index=np.random.permutation(100)
        for i in permutation_index:
            step=eval_gradient(gradient_list[i],new_state)## the gradient for the ith datapoint
            new_state=new_state-0.01*step
        
    return new_state, count
##########################################################

def error_function_list_f(input_matrix,result_vector):
    '''
    This function returns a list of the error functions for each of the datapoints
    '''
    error_func_list= [ln_creator(input_matrix[i],result_vector[i]) for i in range(len(result_vector))]
    return error_func_list

###############################################################
def eval_error(error_func,state):
    '''
    This function takes a error function in symbolic form and evaluates it on the state, returning a scalar.
    '''
    f=sp.lambdify([w1,w2,w3],error_func)
    error=f(*state)
    return error
