import numpy as np

def generated_points_f():
    '''
    This function returns two random points on the sin(pi*x) function, in the interval [-1,1]
    '''
    random_points=2*np.random.rand(1,2)-1
    #random_points=np.arange(2)/2 ##Testing
    result_vector=np.sin(np.pi*random_points).reshape(2,1)
    return random_points, result_vector
    
######################################################
#### y=ax hypothesis

def input_matrix_f(random_points):
    '''
    Returns the input matrix transposed. Assume random points is a row vector with two entries
    '''

    input_matrix=random_points.reshape(2,1)
    return input_matrix

def Lin_weight_vector_f(input_matrix, result_vector):
    '''
    This function computes the linear regression scalar. Assumes input_matrix is a column vector
    '''
    transpose=input_matrix.T
    inverse_part=1/(np.dot(transpose,input_matrix))
    #print(np.dot(transpose,input_matrix))
    pseudo_inverse=np.dot(inverse_part,transpose)
    #print(pseudo_inverse)

    return np.dot(pseudo_inverse,result_vector)

#################################################
## y=b hypothesis
def constant_vector_f(result_vector):
    '''
    This function takes the two y values and returns their average, which represents their midpoint
    '''
    return result_vector.mean()

##########################################################
### y=ax+b hypothesis
def appended_input_matrix_f(random_points):
    '''
    This function takes the random x values and puts them in column form, along with appending a 1 at the beginning.
    '''

    input_matrix=np.ones((2,random_points.size))
    #print(random_points.size)
    input_matrix[0,:]=1
    input_matrix[1]=random_points

    return input_matrix.T    

def LinReg_weight_vector_f(input_matrix, result_vector):
    '''
    This function computes the linear regression vector. Assumes rows are the different vectors and columns is the dimension of the vector
    '''
    transpose=input_matrix.T
    inverse_part=np.linalg.inv(np.dot(transpose,input_matrix))
    pseudo_inverse=np.dot(inverse_part,transpose)

    return np.dot(pseudo_inverse,result_vector)

##############################################################
### y=ax^2 hypothesis
def nonlinear_input_matrix_f(random_points):
    '''
    This function takes the x values in row form, squares them, and puts them in column form
    '''
    square=random_points**2
    return square.reshape(2,1)
