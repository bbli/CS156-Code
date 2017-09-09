import numpy as np

def ten_coins_freq():
    '''
    This function returns the frequency of heads in 10 tosses
    '''
    return np.sum(np.random.randint(2,size=10))/10



def one_thousand_trials():
    '''
    This function returns the frequency of the first toss, a random toss, and the min frequency
    '''
    trials=np.zeros(1000)
    for i in range(1000):
        trials[i]=ten_coins_freq()
    return trials[0],trials[np.random.randint(1000)],np.amin(trials)

#def ten_thousand_trials():
#    '''
#    This function returns the frequency of the first toss, a random toss, and the min frequency. This version is slower because put allows you more functionality than regular slicing
#    '''
#    trials=np.zeros(100000)
#    for i in range(100000):
#        trials.put(i,ten_coins_freq())
#    return trials
