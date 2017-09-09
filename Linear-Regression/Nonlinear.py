import random as r
import LinearSetup as L
import numpy as np

##########################################################
def nonlinear_function(x1,x2):
    return x1**2+x2**2>0.6
def random_flips_f(result_vector):
    '''
    This function takes the result_vector and flips 10% of the bits
    '''
    n=result_vector.size
    #########################################################
    int_vector1=np.zeros(n)
    int_vector1[:]=result_vector
    int_vector2=np.absolute(int_vector1-1)
    ##################################################
    integers=r.sample(range(1,n),10)
    replacement=int_vector2[integers]
    np.put(int_vector1,integers,replacement)
    return int_vector1


def noisy_vector_f(random_points):
    '''
    This function takes the random_points and returns a noisy result_vector
    '''
    result_vector=nonlinear_function(*random_points)
    return random_flips_f(result_vector)

##############################################################
def higher_input_matrix_f(random_points):
    '''
    This function takes the random points and generates the higher dimensonal map
    '''
    x1=random_points[0]
    x2=random_points[1]
    x1x2=x1*x2
    s1=x1**2
    s2=x2**2

    ab,n=random_points.shape
    final_input_matrix=np.zeros((6,n))
    final_input_matrix[0,:]=np.ones(n)
    final_input_matrix[1,:]=x1
    final_input_matrix[2,:]=x2
    final_input_matrix[3,:]=x1x2
    final_input_matrix[4,:]=s1
    final_input_matrix[5,:]=s2

    return final_input_matrix.T

#####################################################################

def higher_classification_error_f(weight_vector,higher_input_matrix,result_vector):
    '''
    This function takes LR weight_vector and transformed data set and returns the classification error.
    '''
    assert weight_vector.size == 6, "Not a 6 dimensonal vector!"
    w0=weight_vector[0]
    w1=weight_vector[1]
    w2=weight_vector[2]
    w3=weight_vector[3]
    w4=weight_vector[4]
    w5=weight_vector[5]
    

    tranposeback=higher_input_matrix.T
    x0=tranposeback[0]
    x1=tranposeback[1]
    x2=tranposeback[2]
    x3=tranposeback[3]
    x4=tranposeback[4]
    x5=tranposeback[5]

    sign_list= (w0*x0+w1*x1+w2*x2+w3*x3+w4*x4+w5*x5 >0)
    
    assert sign_list.size == result_vector.size
    comparsion= (sign_list == result_vector)
    return np.sum(comparsion)/comparsion.size
