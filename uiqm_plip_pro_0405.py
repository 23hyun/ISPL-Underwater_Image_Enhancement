#!/usr/bin/env python
"""
# > Modules for computing the Underwater Image Quality Measure (UIQM)
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from scipy import ndimage
import numpy as np
import os


def plip_g(x,mu=1026.0):
    return mu-x


def plip_theta(g1, g2, k):
#    g1 = plip_g(g1)
#    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))


def plip_cross(g1, g2, gamma):
#    g1 = plip_g(g1)
#    g2 = plip_g(g2)
 
    return  g1+g2-((g1*g2)/(gamma))


def plip_diag(c, g, gamma):
#    g = plip_g(g)
    return gamma - (gamma * np.power((1 - (g/gamma) ), c) )


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - np.power(np.exp(-g/ plip_lambda), 1/plip_beta));


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   =1.0
    return -plip_lambda * np.power(np.log(1 - g / plip_lambda), plip_beta)


def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = np.ceil(alpha_L*K)
    T_a_R = np.floor(alpha_R*K)

    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val


def s_a(x, mu):
    xx = x-mu
    ave = sum(xx*xx)
    return ave/len(x)


def _uicm(x):

    R = x[:, :, 2].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 0].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = np.sqrt(np.power(mu_a_RG,2)+np.power(mu_a_YB,2)) 
    r = np.sqrt(s_a_RG+s_a_YB)
    # print( mu_a_RG,mu_a_YB,np.sqrt(s_a_RG),np.sqrt(s_a_YB))
    return (-0.0268*l)+(0.1586*r)


def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0
  
    return mag


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
 #    
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    # weight
    w = 2.0/(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size

    x = x[:blocksize_y*k2, :blocksize_x*k1]    
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
  
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if not (min_ == 0.0 or max == 0):
                val += np.log(max_/min_)
        
    return w*val


def _uism(x, ws):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:,:,2]
    G = x[:,:,1]
    B = x[:,:,0]
    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    
 
    # get eme for each channel
    r_eme = eme(R_edge_map, ws)
    g_eme = eme(G_edge_map, ws)
    b_eme = eme(B_edge_map, ws)
    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.114
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    xx = (x[:,:,0]+x[:,:,1]+x[:,:,2])/3.0

    # weight
    w = -1.0/(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    xx = xx[:blocksize_y*k2, :blocksize_x*k1]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = xx[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            
            max_ = np.max(block)
            min_ = np.min(block)
            
            if not (max_== min_):
                top = plip_theta(max_, min_,plip_k)
                bot = plip_cross(max_, min_,plip_mu) 
                v1 = top/bot
                val += plip_multiplication(v1, np.log(v1)) 
                
    out =  plip_diag(w, val, plip_gamma)

    return out


def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    k1 = int(x.shape[1])
    k2 = int(x.shape[0])
    w_size = 10
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
 #   c1 = 0.0371281; c2 = 0.33345; c3 = -0.34144
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
 
    uicm   = _uicm(x)
    uism   = _uism(x,w_size)
    uiconm = _uiconm(x,w_size)
    
    uiqm = c1*uicm+c2*uism+c3*uiconm
    # print(' uiqm= %f3.2, uicm= %f3.2, uism = %f3.2 uiconm = %f3.2' % (uiqm, uicm, uism, uiconm))
    # print('UIQM_plip version out')
    return uiqm / (c1 + c2 + c3), uiqm, uicm, uism, uiconm


def i2f(i_image):
    f_image = np.float32(i_image)/255.0
    return f_image


def get_dir_all_files(dir_name):
    total_files = []
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for file in files:
            file_full_path = os.path.join(root, file)
            if os.path.isfile(file_full_path):
                total_files.append(file_full_path)
    return total_files