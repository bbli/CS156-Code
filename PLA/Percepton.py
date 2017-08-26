import numpy as np
import random as rand
import PerceptonSetup as P
import pdb
import pickle

with open("mypickle.pickle", "rb") as f:
    b,m,x_list,y_list,output_list=pickle.load(f)


########################################
def sign(value):
    '''
    This function return the sign of a value
    '''
    if value > 0:
        return 1
    elif value < 0 :
        return -1
    elif value == 0:
        return 0
    else:
        return "error"
def innerproduct(weight_vector, x, y):
    '''
    This function takes the weight_vector and the coordinates of a point and computes their inner product
    '''
    return weight_vector[0]+weight_vector[1]*x+weight_vector[2]*y

def comparer(weight_vector, x, y, z):
    '''
    This function takes a point and tells me if my hypothesis agrees with it or not
    '''
    sign_hyp=sign(innerproduct(weight_vector, x, y))
    if sign_hyp == z:
        return True
    else:
        return False
#######################################################################

def adjust(weight_vector,x,y, z):
    '''
    Returns the adjusted weight vector
    '''
    if z ==1:
        return [ x1+x2 for (x1,x2) in zip(weight_vector,[1,x,y]) ]
    else:
        return [ x1-x2 for (x1,x2) in zip(weight_vector,[1,x,y]) ]
##########################################################################

def list_classifier(x,y,line):
    '''
    This function takes in two numpy lists and a line function, and produces a list of True False values
    '''
    return line(x)<y

########################################################################
def vec_to_lin(weight_vector):
    '''This function returns the line for a weight vector'''
    b=-weight_vector[0]/weight_vector[2]
    m=-weight_vector[1]/weight_vector[2]
    def line(x):
        return m*x+b
    return line
#########################################################################


def PLA(inital_weight_vector, datapoints):
    '''This function returns the final weight vector. The points should be in tuple form inside a list, like (x1,x2,y1)
    '''
    shuffled_list=datapoints
    rand.shuffle(shuffled_list)
    weight_vector=inital_weight_vector
    converge = False
    counter=0  ###To get the average number of iterations
    #print(weight_vector[0]) ## Testing my theory that the constant doesn't change that much
    while converge == False:
        counter=counter+1
        converge = True
        for x,y,z in shuffled_list:
            compare_result=comparer(weight_vector,x,y,z)
            if compare_result:
                pass
            else:
                converge = False
                weight_vector=adjust(weight_vector,x,y,z)
                #print(weight_vector[0]) ## Testing my theory...etc
                break
    return weight_vector,counter

    ## Remember to profile your code

def Probability(weight_vector):
    '''
    This function returns the probability_of_mismatch
    '''
    x_intermediate, y_intermediate=P.random_points(8000)
    x_sample_points = np.array(x_intermediate)
    y_sample_points = np.array(y_intermediate)
    final_weight_line=vec_to_lin(weight_vector)
    final_vec_classified_list=list_classifier(x_sample_points,y_sample_points,final_weight_line)
    target_classified_list=list_classifier(x_sample_points,y_sample_points,lambda x: m*x+b)
    #pdb.set_trace()
    matching_list=(final_vec_classified_list==target_classified_list)
    sum_matching_list=np.sum(matching_list)
    probability_of_mismatch=1-sum_matching_list/matching_list.size
    return probability_of_mismatch


