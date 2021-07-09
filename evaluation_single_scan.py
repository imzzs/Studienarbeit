#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:56:25 2021

@author: axmann
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import cm
#mc=mc_best_rot

# Hack: helper to draw a "crosshair" to a matrix, using NaN's.
def plot_crosshair(m):
    s = m.shape[0]//2
    m[s,:s-20] = np.nan
    m[s,s+20:] = np.nan
    m[:s-20,s] = np.nan
    m[s+20:,s] = np.nan
    return m

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#### Plot Consensus Matrix (best rotation) (without and with conf. ellipse)####
def plot_mc(mc,best_rot=np.nan,savepath="",pc_id="Test",Save=False,vis_conf_ell_mc=False,cov_array_pixel=np.asarray([])):
    plt.figure()
    ax = plt.gca()
    plt.imshow(mc, interpolation="nearest")
    plt.colorbar()
    plt.title("Consensus Matrix (pc: Nr. "+pc_id+", rot. angle: %.2f, %.2f, %.2f)" % (best_rot[0],best_rot[1],best_rot[2] ))
    if Save:
        plt.savefig(savepath+"MC_"+pc_id+"_"+str(best_rot)+".png")

    if vis_conf_ell_mc:    
        confidence_ellipse(cov_array_pixel[:,1], cov_array_pixel[:,0], ax, 3.0, edgecolor='firebrick')
        if Save:
            plt.savefig(savepath+"MC_CONF_ELL"+pc_id+"_"+str(best_rot)+".png")

#### Create height map ####
def create_height_map(mc,savepath="",pc_id="Test",best_rot=np.nan,Save=False):
    x, y = np.meshgrid(range(mc.shape[0]), range(mc.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, mc,cmap=cm.RdYlGn)
    ax.set_zlabel('Consensus')
    plt.title('MC as 3d height map')# (%i points)' %len(h_points))
    if Save:
        plt.savefig(savepath+"HM_"+pc_id+"_"+str(best_rot)+".png")


#### Plot point cloud next to consensus matrix image ####
def create_img_pcl(mc,points_copy_numpy,savepath="",pc_id="Test",img_pcl=False,Save=False):
    if img_pcl:  
        fig, axs = plt.subplots(1,2,figsize=(15,15),gridspec_kw={'width_ratios': [1, 2]})
        plt.axis('off')
        #plt.title("Consensus Matrix and point cloud top view (pc: Nr. "+str(name[name_1:name_2])+", rot. angle: %.2f)" % (best_rot))
    
        axs[0].imshow(mc, interpolation="nearest")
        #axs[1].imshow(np.asarray(image))
        #fig = plt.figure()
        axs[1] = fig.add_subplot(1, 2, 2, projection='3d')
        #fig.add_subplot(111, projection='3d')
    
        axs[1].scatter(points_copy_numpy[:,0], points_copy_numpy[:,1], points_copy_numpy[:,2], c='r', marker='.')        
        axs[1].view_init(azim=0, elev=90)
        figure = plt.gca()
        figure.axes.xaxis.set_ticklabels([])
        figure.axes.yaxis.set_ticklabels([])
        figure.axes.zaxis.set_ticklabels([])
        if Save:
            plt.savefig(savepath+"img_pcl_"+pc_id+".png")

