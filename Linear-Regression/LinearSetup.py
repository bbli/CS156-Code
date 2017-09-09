import numpy as np

def random_points_f(number_of_points):
    '''
    This function returns random points on the plane in the form of one numpy array
    '''
    return 2*np.random.random(size=(2,number_of_points))-1

#########################################################

## I seem to be worried that 
def random_line_f():
    '''
    This function creates a random line function y(x)
    '''
    x2,y2= random_points_f(1)
    x1,y1= random_points_f(1)
    slope= (y2-y1)/(x2-x1)
    def line(x):
        return slope*(x-x1)+y1
    return line

#############################################################
def points_assigner(random_points,random_line):
    '''
    This function takes a set of random points and a line an return a numpy array with the classfications
    '''
    random_points_dimensions,random_points_number=random_points.shape
    assert random_points_dimensions==2, "Input is not 2d!"
    ## To take this assert out, I should write a proper test using my pipeline
    result_vector=np.zeros(random_points_number)
    
    x_values=random_points[0,:]
    y_values=random_points[1,:]
    result_vector[:]= (random_line(x_values)<y_values)
    return result_vector

###########################################################################

def input_matrix_f(random_points):
    '''
    Returns the input matrix with 1's appended to each row
    '''
    random_points_dimensions,random_points_number=random_points.shape
    input_matrix=np.ones((3,random_points_number))
    input_matrix[1:3,:]=random_points
    input_matrix[0,:]=1

    return input_matrix.T    

#################################################################
def LinReg_weight_vector_f(input_matrix, result_vector):
    '''
    This function computes the linear regression vector. Assumes rows are different vectors and columns is the dimension of the vector
    '''
    transpose=input_matrix.T
    inverse_part=np.linalg.inv(np.dot(transpose,input_matrix))
    pseudo_inverse=np.dot(inverse_part,transpose)

    return np.dot(pseudo_inverse,result_vector)

#########################################################################
def reg_line_f(weight_vector):
    '''This function returns the line for a weight vector'''
    try:
        b=-float(weight_vector[0])/float(weight_vector[2])
        m=-float(weight_vector[1])/float(weight_vector[2])
    except ZeroDivisionError:
        b=0
        m=0
    def line(x):
        return m*x+b
    return line

#######################################################################

def frequency_of_mismatch_f(reg_line, random_points, result_vector):
    '''
    This function returns the percentage of the time that the Linear Regression weight vector misclassifies the dataset
    '''
    reg_result_vector=points_assigner(random_points,reg_line)
    comparsion=(reg_result_vector==result_vector)
    return np.sum(comparsion)/comparsion.size

#########################################################################


