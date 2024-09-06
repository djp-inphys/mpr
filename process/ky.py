# Libraries needed for ky (pcacov and mean_scatter)
import numpy as np
import sklearn.decomposition as sk
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#scores=scores.T
import sys

'''
    principal component analysis 
    input covariance matrix

PCA(n_components=None, copy=True, whiten=False, svd_solver=’auto’, tol=0.0, iterated_power=’auto’, random_state=None)[source]¶
'''

def princomp(descs, n_components='mle'):
    scores = sk.PCA(n_components).fit_transform( descs )
    return scores


'''
    principal component analysis 
    input covariance matrix
'''

def pcacov(cv):
    # covariance matrix
    coeff, latent, s = np.linalg.svd(cv, full_matrices=True)
    #coeff=s
    totalvar = np.sum(latent)
    explained = 100 * latent/totalvar
    p,d = np.shape(coeff)
    maxind = np.zeros(p)
    for findingmaxindex in range (p):
      maxabscoeff = np.max(np.abs(coeff[:,findingmaxindex]))
      m = np.max(np.abs(coeff[:, findingmaxindex]))
      maxindex=[i for i, j in enumerate(np.abs(coeff[:,findingmaxindex])) if j == m]
      maxind[findingmaxindex]=maxindex[0]

    ind = np.zeros(p)
    sign = np.zeros(p)
    reshapedcoeff = np.reshape(coeff, p*p,1)
    #reshapedcoeff = np.transpose(reshapedcoeff)
    for pdaccumlation in range(p):
        ind[pdaccumlation] = p*pdaccumlation
        indd=maxind[pdaccumlation] + ind[pdaccumlation]
        sign[pdaccumlation] = np.sign(reshapedcoeff[int(indd)])

    coefff= coeff*sign
    #coeff=coefff
    #coeff=coeff*-1
    return coeff, latent



def mean_scatter(priors,mean_vecs):
    ndim, ncls = np.shape(mean_vecs)

    mean_vecss = np.stack(mean_vecs)
    mean_vecsss = np.matrix(mean_vecss)
    mean_vecssss = np.transpose(mean_vecss)
    mean_vecsssss = np.matrix(mean_vecssss)

    mn = np.matrix.mean(mean_vecsssss)
    Sb= np.zeros((ndim,ndim))
    mn= np.zeros((ndim))

    for meanfinding in range(ndim):
        mn[meanfinding] =np.matrix.mean(mean_vecsssss[:, meanfinding])

    for cls_no in range(ncls):
        mn_vec = mean_vecs[:,cls_no]
        mnminusvec_mn = mn_vec - mn
        mnminusvec_mnT = (mnminusvec_mn[:,None].T)

        multi=np.dot(mnminusvec_mn[:,None],mnminusvec_mn[:,None].T)
        Sb = Sb + (priors[0] * (multi))

    return Sb


'''
 Main function for ky 
( the function needs the A matrix which is the output of pca or any ( samples X observation ) matrix
  and the number of classes, i.e 5 for our IsoId work)
  the last element on the input arrays is the 
'''
def ky(data,cls):
    # cls
    num_cls = int( np.max( cls ) + 1)

    # sort out priors
    no_cls = np.zeros(num_cls)
    for clsno in range( num_cls ):
        no_cls[clsno] = np.count_nonzero(cls==clsno)

    nomeas = np.sum( no_cls )
    priors  = np.zeros(num_cls)
    for idx in range(num_cls):
        priors[idx] = no_cls[idx]/nomeas

    no_meas = int( data.shape[0]/2 )
    no_dim  = data.shape[1]
    Sw      = np.zeros( (no_dim, no_dim) )
    mns     = np.zeros( (no_dim, num_cls) )
    for clsno in range(num_cls):
        #meas_idx=0
        sample = data[cls==clsno,:]

        Sw = Sw + priors[clsno] * np.cov(sample.T)

        ave = sample.mean(0)
        mns[:,clsno] = ave

    U,lembda = pcacov(Sw)
    d=np.diag(np.sqrt(1/lembda))
    B=np.matmul(U,d)

    Sb = mean_scatter(priors,mns)
    # output
    multi = np.matmul(Sb,B)
    finalTransformMatrix = np.matmul(B.T,multi)
    V, lambdaa = pcacov(finalTransformMatrix)


    coeffs = np.matmul(B,V)

    scores = np.matmul(data,coeffs)
    #scores =  dta * coeffs

    #scores = scores.T
    #coeffs = coeffs.T
    return coeffs, scores


# # If u applied PCA first then just run this code below (score the output score of PCA):
#
# A = score.T
#
# coeffs,scores = ky(A,5)
#
# ## Untill her the Ky function finished, the code lines below  are just to save the output or plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(scores[1:768,0],scores[1:768,1],scores[1:768,2])
# ax.scatter(scores[769:768*2,0],scores[769:768*2,1],scores[769:768*2,2])
#
# ax.scatter(scores[768*2:768*3,0],scores[768*2:768*3,1],scores[768*2:768*3,2])
# ax.scatter(scores[768*3:768*4,0],scores[768*3:768*4,1],scores[768*3:768*4,2])
# ax.scatter(scores[768*4:768*5,0],scores[768*4:768*5,1],scores[768*4:768*5,2])
#
# #plt.show()
# #plt.close(fig)
# #plt.clf()
# #fig.gcf()
# plt.draw()
# plt.pause(1) # <-------
# input("<Hit Enter To Close>")
# plt.close(fig)
#
# np.savetxt("scores.csv", scores, delimiter=",")
# np.savetxt("coeffs.csv", coeffs, delimiter=",")
# np.savetxt("Sw.csv", Sw, delimiter=",")
# np.savetxt("coeff.csv", coeff, delimiter=",")
# np.savetxt("sign.csv", sign, delimiter=",")
# np.savetxt("U.csv", U, delimiter=",")
# np.savetxt("d.csv", d, delimiter=",")
# np.savetxt("B.csv", B, delimiter=",")
# np.savetxt("Sb.csv", Sb, delimiter=",")
# np.savetxt("V.csv", V, delimiter=",")
# np.savetxt("finalTransformMatrix.csv", finalTransformMatrix, delimiter=",")
# ##
#
#
# AA = genfromtxt('C:/Users/David/PycharmProjects/pcaandky/scorePCApython.csv', delimiter=',')
#
# A=AA