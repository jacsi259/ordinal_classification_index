'''
Ordinal Classification Index

# Please, cite the following reference journal paper if you use this code:
# @article{CardosoJaimeS.Sousa2011,
#  author = {Cardoso, Jaime S. and Sousa, Ricardo},
#  journal = {International Journal of Pattern Recognition and Artificial Intelligence},
#  keywords = {classification accuracy,evaluation measures,machine learning},
#  title = {{Measuring the performance of ordinal classification}},
#  year = {2011},
#  doi = {10.1142/S0218001411009093},
#  volume = {25},
#  number = {8},
#  pages = {1173--1195}
# }
    
    # Jaime S. Cardoso :: jaime.cardoso <at> inescporto <dot> pt
# Ricardo Sousa :: ricardo.j.sousa <at> inescporto <dot> pt
# INESC TEC, Faculdade de Engenharia, Universidade do Porto, Portugal
    
    # input: confusion matrix and number of classes
# size(cMatrix) must be [K K]

'''
import numpy as np
from sklearn.metrics import confusion_matrix

def add_offset(n, error):
    offset = np.random.randint(low=-error, high=error, size=1)
    val = n + offset
    if val > 6:
        val = 6
    if val < 0:
        val = 0
    return int(val)

def OrdinalClassificationIndex(cMatrix):
    '''
    Calculate the Ordinal Classification Index from the confusion matrix
    
    Input: confusion matrix as produced by sklearn.metrics.confusion_matrix
    
    Output: an Ordinal Classification Index score
    
    '''
    K = len(cMatrix)
    N = np.sum(cMatrix)

    ggamma=1

    bbeta=0.75 / (N * ((K - 1) ** ggamma))
    helperM2 = np.zeros([K,K])
    errMatrix = np.zeros([K,K])
    for r in range(K):
        for c in  range(K):
            helperM2[r,c]=  cMatrix[r,c] * ((abs(r - c)) ** ggamma)
    
    TotalDispersion = (np.sum(helperM2)) ** (1 / ggamma)
    helperM1 = cMatrix / (TotalDispersion + N)
    errMatrix[0,0] = 1 - helperM1[0,0] +  (bbeta * helperM2[0,0])
    for r in  range(1,K-1):
        c=1
        errMatrix[r,c] = errMatrix[r - 1,c] - helperM1[r,c] +  (bbeta * helperM2[r,c])
    
    for c in  range(1, K-1):
        r=1
        errMatrix[r,c] = errMatrix[r,c - 1] - helperM1[r,c] +  (bbeta * helperM2[r,c])
    
    for c in  range(1,K-1):
        for r in  range(1,K-1):
            costup = errMatrix[r - 1,c]
            costleft = errMatrix[r,c - 1]
            lefttopcost = errMatrix[r - 1,c - 1]
            aux = min(costup,costleft,lefttopcost)
            errMatrix[r,c]= aux - helperM1[r,c] +  (bbeta * helperM2[r,c])
    
    oc = errMatrix[-1,-1]
    return oc


if __name__ == '__main__':
    #Generate a confusion matrix
    y_true = [[0]*1000,[1]*1000,[2]*1000,[3]*1000, [4]*1000, [5]*1000, [6]*1000]
    y_true = [item for sublist in y_true for item in sublist]
    
    #generate a sequence of predictions where the class if off by up to 2 adjacent classes on ordinal scale
    y_pred_2 = [add_offset(n, 2) for n in y_true]
    
    #generate a sequence of predictions where the class if off by up to 3 adjacent classes on ordinal scale
    y_pred_3 = [add_offset(n, 3) for n in y_true]
    
    conf_2 = confusion_matrix(y_true, y_pred_2)
    conf_3 = confusion_matrix(y_true, y_pred_3)
    
    #The code is obviously broken because it returns 0.0 no matter what.
    OrdinalClassificationIndex(conf_2)
    OrdinalClassificationIndex(conf_3) 
