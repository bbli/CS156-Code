import numpy as np
import pandas as pd

def datapoints_f(path):
    '''
    This function takes the raw data and changes it into a numpy array and a result_vector, with the array's columns being each datapoint. Assumes path is a string
    '''
    df=pd.read_csv(path,delim_whitespace=True,names=['x','y','result'])
    data=df.as_matrix()
    datapoints=data[:,0:2]
    result_vector=data[:,2]
    return datapoints,result_vector
###########################################################
def input_matrix_f(datapoints):
    '''
    Returns the transformed inputs. Assumes transformed space is 8 dimensional. Also assumes the datapoints' columns are each point in R^2
    '''
    # Seperating out the x and y values
    transpose=datapoints.T
    x1=transpose[0]
    x2=transpose[1]

    # Creating the output matrix
    datapoints_dimensions,datapoints_number=transpose.shape
    input_matrix=np.ones((8,datapoints_number))
    input_matrix[0]=1
    input_matrix[1]=x1
    input_matrix[2]=x2
    input_matrix[3]=x1**2
    input_matrix[4]=x2**2
    input_matrix[5]=x1*x2
    input_matrix[6]=np.abs(x1-x2)
    input_matrix[7]=np.abs(x1+x2)

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

################################################################
def squared_error_f(weight_vector, datapoints, result_vector):
    '''
    This function returns the squared error
    '''
    pred_vector=np.dot(datapoints,weight_vector)
    norm_vector=np.subtract(pred_vector,result_vector)
    norm=np.linalg.norm(norm_vector)
    samples_size=len(result_vector)
    return (norm**2)/samples_size

