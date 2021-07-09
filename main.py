#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:06:39 2021

@author: axmann
"""

import numpy as np
import os
from os import path
from sys import argv
from os.path import splitext, basename
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import cKDTree
import open3d as o3d
from plyfile import PlyData, PlyElement#!
import glob
import natsort#!
from mpl_toolkits.mplot3d import Axes3D
import time
from collections import Counter
import copy
import pylab
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from numba import jit, njit
from scipy.spatial.transform import Rotation as R_scipy
import math
from scipy.stats import entropy
#from calculation_functions import *#!
from evaluation_single_scan import *#!
import multiprocessing
from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects

#matplotlib inline

#### Find 80 percent threshhold by reshaping and sorting ####
def sort_with_threshold(mc):  #mc是一个array 
    mc[np.isnan(mc)]=0    #mc上所有的非数都应为0  但是bool的array怎么做index？
    mc_resh=mc.reshape(1,np.size(mc))  #mc_resh为 一行
    mc_resh_sorted=-np.sort(-mc_resh)    #大到小
    mc_thresh=np.floor(mc_resh_sorted[0,0]*0.8)
    return mc_thresh

@njit
def calc_ellipse(mc__,mc_thresh_,max_cons_grid_edge):   
    cov_list=[]

    for i in range(mc__.shape[0]):
        for j in range(mc__.shape[1]):
            if mc__[i,j]>mc_thresh_:
                for k in range(int(np.round(mc__[i,j]))):
                    cov_list.append([i,j])
    
    #mc__[i,j] --> i entspricht Reihe (y-Wert/Koordinate)), j entspricht Spalte (x-Wert/Koordinate)--> für mean und cov vertauschen erforderlich!!
    cov_array_p=np.asarray(cov_list)
    cov_array=(np.asarray(cov_list)*max_cons_grid_edge)-(mc__.shape[0]*max_cons_grid_edge*0.5)
    x_meaN=np.mean(cov_array[:,1])
    y_meaN=np.mean(cov_array[:,0])
    cov=np.cov(cov_array[:,1],cov_array[:,0])
    return cov,cov_array_p,cov_array,x_meaN,y_meaN

def get_map_path(i):   
    switcher={
            "LOD0":"/home/axmann/Dokumente/Data/Mapathon/triangle/Map/cut_E_549090-549242_N_5804040-5804185/adjusted_tiles_E_549090-549242_N_5804040-5804185.ply",
            "LOD1":'',
            "LOD2":"E:/5/documents-export-2021-04-22/Zisen_Zhao/Zisen_Zhao/maps/002.ply",
            "LOD3":"E:/5/documents-export-2021-04-22/Zisen_Zhao/Zisen_Zhao/maps/003.ply",
            "LOD4":"E:/5/documents-export-2021-04-22/Zisen_Zhao/Zisen_Zhao/maps/004.ply",
            "LOD5":"E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\maps\\005.ply",
            "classified":"E:/5/documents-export-2021-04-22/Zisen_Zhao/Zisen_Zhao/maps/173_174_184_187.ply",
            "LM":"/home/axmann/Dokumente/Data/Mapathon/triangle/Map/cut_E_549090-549242_N_5804040-5804185/latent_map_15_E_549090-549242_N_5804040-5804185.ply",
            "Veg_rem":"/home/axmann/Dokumente/Data/Mapathon/triangle/Map/Scanstrips/Ply_withoutGround/Merged_corner.ply",
            "Orig":"/home/axmann/Dokumente/Data/Mapathon/triangle/Map/original_Riegl_point_clouds/Merged/20200825_mapathon_laserdata_etrs_coordinates - Scanner 1and2 - 200825_123323_Scanner_1and2 - originalpoints - Cloud.ply"
            }
            
    return switcher.get(i,"Invalid day of week")   #.get 输入key值get value 如果不存在返回 Invalid day of week

def calculate_normal_vectors(points):
    #### Calculate normal vectors of car sensor scan ####
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points)
    points_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.7, max_nn=100))

    #### Convert Open3D.o3d.geometry.PointCloud to numpy array ####
    points_numpy = np.asarray(points_pcd.points)
    normals_numpy = np.asarray(points_pcd.normals)
    points=np.concatenate((points_numpy, normals_numpy),axis=1)  #  points[:,:3] position   points[:,3:6] normal vector
    return points

def calc_covariance_score(max_cons_matrix_,xy_indices,points_0,idx_0,points_1,idx_1):

    matched_map_points = points_0[idx_0]
    matched_scan_points = points_1[idx_1]
   
    # Compute cosin of normal vectors and use as weights
    # Note cos(theta) = n1*n2/norm(n1)/norm(n2)
    nv_prod = np.sum(matched_map_points[:,3:6] * matched_scan_points[:,3:6], axis=1)
    nv_prod[nv_prod < 0.0] = 0.0
    
    cov_xx = matched_map_points[:,3]*matched_map_points[:,3]*nv_prod
    cov_yy = matched_map_points[:,4]*matched_map_points[:,4]*nv_prod
    cov_xy = matched_map_points[:,3]*matched_map_points[:,4]*nv_prod
    cov_matrix = np.zeros((max_cons_matrix_.shape[0],max_cons_matrix_.shape[1],2,2))
    np.add.at(cov_matrix, (xy_indices[:,0],xy_indices[:,1],0,0), cov_xx)
    np.add.at(cov_matrix, (xy_indices[:,0],xy_indices[:,1],0,1), cov_xy)
    np.add.at(cov_matrix, (xy_indices[:,0],xy_indices[:,1],1,1), cov_yy)
    cov_matrix[:,:,1,0] = cov_matrix[:,:,0,1]
    # Avoid singularity
    cov_matrix[:,:,0,0] += 1e-10
    cov_matrix[:,:,1,1] += 1e-10
    
    # Estimate size of error ellipse by computing a^2 + b^2,
    # where a and b are the semi-major axis of error ellipse, 
    # which is then the eigenvalue of inv(cov).
    # Note a^2 + b^2 = trace(inv(cov)) = trace(cov)/det(cov).
    cov_score_matrix = np.trace(cov_matrix,axis1=cov_matrix.ndim-2,axis2=cov_matrix.ndim-1) / np.linalg.det(cov_matrix)
    cov_score_matrix = 1./cov_score_matrix #np.sqrt(1./cov_score_matrix)
    return cov_score_matrix


@njit
def matmul_jit(A, B):
    """Perform square matrix multiplication of out = A * B
    """
    out = np.empty((A.shape[0],B.shape[1]), B.dtype)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            out[i, j] = tmp
    return out



# Compute max cons alignment for two scans.
def align_two_scans(points_0, points_1,save_backprojection=True):
      
    # Trick for kd tree:
    # - In x and y, we want to find all points within distance
    #   +/- max_cons_radius.
    # - However, in z, we want to find points only within max_cons_z_range.
    # In order to have the same search radius in all 3 dimenstions, we
    # simply scale up z.

    points_1_=points_1
    points_1_[:,2] *= z_scale    #为了统一搜索范围，z的搜索范围需要scale


    # For each point in points_1, find all the neighbors.
    # Note we use Minkowski inf-norm, using p=inf.
    # Neighbors will be an array of len(points_1) lists, each list containing
    # the indices of all neighbors. Lists may be empty.
    neighbors = tree_.query_ball_point(points_1_[:,0:3], max_cons_radius,
                                        p=np.inf)
    
        
    #new array based implementation for numba
    repetitions=np.zeros(len(neighbors), dtype=int)
    i=0
    for e in neighbors:
        repetitions[i]=int(len(e))
        i+=1
    
    idx_1 = np.repeat(np.arange(len(points_1_)), repetitions)

    # Also make a flat list of indices into points_0.
    idx_0 = np.concatenate(neighbors).astype(int)
    #print("pairs: %7d" % idx_0.shape[0])

    # Compute all xy offsets.
    # Here we compute the shift of scan 1, relative to scan 0.
    xy_offsets = points_1_[:,0:2][idx_1] - points_0[:,0:2][idx_0]

    # If required, compute scalar products of normal vectors.
    if use_normal_vectors:
        nv_prod = np.sum(points_0[:,3:6][idx_0] * points_1_[:,3:6][idx_1],
                         axis=1)
        # Zero where it is negative.
        nv_prod[nv_prod < 0.0] = 0.0

    # Convert into bins and matrix indices.
    # The max cons matrix will be from -matrix_radius to +matrix_radius,
    # which will be 2*matrix_radius+1 elements.
    matrix_radius = int(np.ceil(max_cons_radius / max_cons_grid_edge))
    factor = matrix_radius / max_cons_radius
    xy_indices = np.round(xy_offsets * factor).astype(int) + matrix_radius
    
    if single_consideration:      
        single_indices, indicesList = np.unique(np.concatenate((xy_indices,idx_1[:,np.newaxis]),axis=1),axis=0,return_index=True)
        single_indices = np.concatenate((single_indices,idx_0[indicesList,np.newaxis]),axis=1)
        xy_indices=single_indices[:,0:2]
        idx_1=single_indices[:,2]
        idx_0=single_indices[:,3]
           
        
    # Make max cons matrix and accumulate all results.
    max_cons_matrix = np.zeros((2*matrix_radius+1, 2*matrix_radius+1))
    if use_normal_vectors:
        # Use nv product as weight.
        np.add.at(max_cons_matrix, (xy_indices[:,0], xy_indices[:,1]),
                  nv_prod)
    else:
        # Do not use scalar product as weight, instead fix at 1.0.
        np.add.at(max_cons_matrix, (xy_indices[:,0], xy_indices[:,1]), 1.0)
        
    if use_covariance_score:
        max_cons_matrix=calc_covariance_score(max_cons_matrix,xy_indices,points_0,idx_0,points_1_,idx_1)
            
            
    if save_backprojection:
        if single_consideration:
            idx_max1,idx_max2 = np.where(max_cons_matrix == np.amax(max_cons_matrix))
            backproj_ind = np.where((xy_indices[:,0]==idx_max1) & (xy_indices[:,1]==idx_max2))
            best_backproj_points=points_1_[single_indices[backproj_ind[0],2]]
            best_backproj_points[:,2]/=z_scale
        else:
            idx_max1,idx_max2 = np.where(max_cons_matrix == np.amax(max_cons_matrix))
            backproj_ind = np.where((xy_indices[:,0]==idx_max1) & (xy_indices[:,1]==idx_max2))
        
            best_backproj=idx_1[backproj_ind[0]]
            best_backproj_points=points_1_[best_backproj]
            best_backproj_points[:,2]/=z_scale
    else:
        best_backproj_points=np.asarray([])
    
    # Flip the matrix so that "x" is to the right and "y" is up.
    max_cons_matrix = np.flipud(max_cons_matrix.T)

    return max_cons_matrix, best_backproj_points

def apply_small_angle_rotation_o3d(rot,h_points,rot_center):
    h_points_local=h_points-rot_center[0:3]
    
    points_rot = o3d.geometry.PointCloud()
    points_rot.points = o3d.utility.Vector3dVector(h_points_local)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()            
    R = mesh.get_rotation_matrix_from_zyx((2*np.pi+np.pi * rot_center[5] / 180, 2*np.pi+np.pi * rot_center[4] / 180,2*np.pi+ np.pi * rot_center[3] / 180))
    points_copy = copy.deepcopy(points_rot)


    points_copy.rotate(R)
    points_l=copy.deepcopy(points_copy)
    points_l=np.asarray(points_l.points)
    
    R_small_angle = mesh.get_rotation_matrix_from_zyx((2*np.pi+np.pi * rot[2] / 180, 2*np.pi+np.pi * rot[1] / 180,2*np.pi+ np.pi * rot[0] / 180))
    R_local_to_global = matmul_jit(R.transpose(),R_small_angle)#R_small_angle.transpose() hier nehme ich das transpose raus
    points_copy.rotate(R_local_to_global)#,rot_center, da h_points_local=h_points-rot_center[0:3]
    
    #### Estimate normals of rotated points and convert to numpy array ####
    if use_normal_vectors or use_covariance_score:
        points_copy.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.7, max_nn=100)) 
        points_copy_numpy = np.asarray(points_copy.points)
        normals_copy_numpy = np.asarray(points_copy.normals)
        rotated_points=np.concatenate((points_copy_numpy, normals_copy_numpy),axis=1)
    else:
        rotated_points=np.asarray(points_copy.points)
    rotated_points[:,0:3]=rotated_points[:,0:3]+rot_center[0:3]
    return rotated_points



@delayed
@wrap_non_picklable_objects
def do_parallel_6DOF(rot,h_points,rot_center):#
    h_points=h_points.copy()
    h_points[:,2]=h_points[:,2]+rot[3]
    rotated_points=apply_small_angle_rotation_o3d(rot,h_points,rot_center)
   
    mc,backprojection = align_two_scans(h_map_pcl,rotated_points,False) #, 
    return (np.amax(mc),rot[4])



def calculate_mc(name_):   #输入是一个点
    ##### Measure elapsed time ####
    print(name_)

    
    #### Load car sensor scan ####
    points = np.loadtxt(name_, delimiter=",")  
    if len(points)==0:
        return 0
    # if len(points)==3:
        
    points=points[:,:3]
    #points=points.copy()    #for assignment_destination is read-only
    #### Calculate normal vectors of car sensor scan ####
    points=calculate_normal_vectors(points)    # return  point with shape(:,6)
    
    
    #### Remove ground of car sensor scan: Vertical surfaces = horizontal normals. ####
    if remove_ground_car_sensor_scan:
        h_points = points[np.logical_and(
            points[:,5] >= cos_lb, points[:,5] <= cos_ub)]
        print("remaining points on vertical surfaces", len(h_points))
    else:
        h_points=points
    if len(h_points)==0:
        return 0
    
    #### Initialize rotation of car sensor scan ####
    points_rot = o3d.geometry.PointCloud()
    points_rot.points = o3d.utility.Vector3dVector(h_points[:,0:3])   #h_points就是扫描后用来定位的点的法向量，去掉了地面
    points_rot.normals = o3d.utility.Vector3dVector(h_points[:,3:6])
    max_prev=0
    
    #### Downsampling of car sensor scan (still an open task) ####
    if downsampling_carscan:
        points_rot_ = points_rot.voxel_down_sample(voxel_size=0.2)
        h_points=np.concatenate((np.asarray(points_rot_.points), np.asarray(points_rot_.normals)),axis=1) 
        
    
    #### Find rotation center of car sensor scan ####    #
    ind_rot_center=np.where(reference_trajectory[:,0]==int(name_[name_1:name_2]))[0]    #  在这个地方的转弯中心，就是参考轨迹  extractet tractory point的点

    if ind_rot_center.size==0:
        rot_center=np.asarray([])
    else:
        rot_center=reference_trajectory[int(ind_rot_center),2:8]   #[position,pose]
    
    
    start_ = time.time()
    num_cores = multiprocessing.cpu_count()
    list_amax=Parallel(n_jobs=num_cores)(do_parallel_6DOF(rot,h_points[:,0:3],rot_center) for rot in rotation_combinations)#ACHTUNG n_jobs=num_cores #n_jobs=4, verbose=3
    end_ = time.time()
    elapsed_time_Parallel = end_ - start_
    print(elapsed_time_Parallel)

        
    list_amax_array=np.asarray([item for t in list_amax for item in t])
    list_amax_array=list_amax_array.reshape(int(list_amax_array.shape[0]/2),2)

    sorted_list_amax_array = list_amax_array[np.argsort(list_amax_array[:, 1])]
    sorted_rotation_combinations = rotation_combinations[np.argsort(rotation_combinations[:, 4])]
    store_list_amax_array=np.concatenate((sorted_rotation_combinations,sorted_list_amax_array),axis=1)


    idx_best_rot_comb=np.where(list_amax_array==np.amax(list_amax_array[:,0]))
    if idx_best_rot_comb[0].shape[0]>1:
        print("ATTENTION! MULT ROTS==1")
        mult_rots=1
    else:
        mult_rots=0
        
    best_rot_comb=rotation_combinations[np.where(rotation_combinations[:,4]==list_amax_array[idx_best_rot_comb[0][0]][1])[0][0],:]
    

    h_points = points[np.logical_and(
            points[:,5] >= cos_lb, points[:,5] <= cos_ub)]
    h_points[:,2]=h_points[:,2]+best_rot_comb[3]
    rotated_points=apply_small_angle_rotation_o3d(best_rot_comb,h_points[:,0:3],rot_center)
    np.savetxt(savepath_mc+"best_rotated_points_"+str(name_[name_1:name_2])+"_"+str(best_rot_comb[0])+str(best_rot_comb[1])+str(best_rot_comb[2])+".csv",rotated_points, delimiter=",")
   
    #### Call align_two_scans function (CTF still an open task) #### 
    mc, backprojection = align_two_scans(h_map_pcl,rotated_points,True)#, 
    
    #### Find peak of mc matrix ####  
    idx_max1,idx_max2 = np.where(mc == np.amax(mc))

    #### Check for multiple peaks ####
    idx_max1_shape=idx_max1.shape
    if idx_max1_shape[0]>1:
        mult_peaks=idx_max1_shape[0]
    else:
        mult_peaks=1       
    
    #### Store best roation result ####
    if (np.amax(mc) >= max_prev):
        best_idx1=int(idx_max1[0])
        best_idx2=int(idx_max2[0])
        best_rot_x=best_rot_comb[0]
        best_rot_y=best_rot_comb[1]
        best_rot_z=best_rot_comb[2]
        mc_best_rot=mc
        mult_peaks_best=mult_peaks
        max_prev=np.amax(mc)
        best_backprojection = backprojection
        best_z = best_rot_comb[3]

                
    best_important_facts=np.asarray([best_idx1,best_idx2,best_rot_x,best_rot_y,best_rot_z,mult_peaks_best,max_prev,mult_rots,best_z])            
    np.savetxt(savepath_mc+"best_important_facts"+str(name_[name_1:name_2])+".csv",best_important_facts, delimiter=",")                
    np.savetxt(savepath_mc+"best_backprojection_"+str(name_[name_1:name_2])+"_"+str(best_rot_x)+str(best_rot_y)+str(best_rot_z)+".csv",best_backprojection, delimiter=",")
    np.savetxt(savepath_mc+"mc_best_rot_"+str(name_[name_1:name_2])+"_"+str(best_rot_x)+str(best_rot_y)+str(best_rot_z)+".csv",mc_best_rot, delimiter=",")
    np.savetxt(savepath_mc+"list_amax_array_"+str(name_[name_1:name_2])+".csv",store_list_amax_array, delimiter=",")
               
# ------
# Main
# ------
if __name__ == "__main__":
    #### FOR-LOOP to run multiple evaluation cases ####
    Invest=[["Velodyne","classified",False,"no_single_consideration","no_covariance","no_downsampling",0.02,False,"0.05","001","bresenham"],["Velodyne","LOD3",False,"no_single_consideration","covariance","no_downsampling",0.04]]
    #Invest[i]=[sensor,map,use_normal_vectors,single_consideration,covariance,downsampling,max_cons_grid_edge,simulation,voxel_size,scan_data,algorithum]
    #文件名
    for i in range(1):  #i=0
        stepsize=1
        
        #### Whether to weight the counts using the normal vector scalar product. ####
        use_normal_vectors = Invest[i][2]    #法向量是false
        if use_normal_vectors:
            NV="NV"
        else:
            NV="noNV"
        

        #### Read map .ply file ####
        map_ = Invest[i][1]          #地图
        map_path=get_map_path(map_)        #加载地图地址
        pcl_map_load = o3d.io.read_point_cloud(map_path)  #在open3d里面加载地图，得到点云

        
        #### Read car sensor scans and corresponding reference trajectory ####
        sensor=Invest[i][0]
       
        if sensor == "Velodyne":
            if Invest[i][7]==False:
                  if Invest[i][9]=="000":
                       path_ = 'E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\car_sensor_scans\\Test_heading_m045\\*.txt'  #加载sensor scan point cloud
                  if Invest[i][9]=="001":
                       path_ = 'E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\trajectory_utm_adj\\trajectory_utm_adj\\*.txt'
                  # Load the reference trajectory
                  
            else:
                if Invest[i][10]=="slab":
                      path_="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\slab-"+Invest[i][8]+"-"+Invest[i][1]+"\\*.txt"                     
                elif Invest[i][10]=="bresenham":
                      path_="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\bresenham-"+Invest[i][8]+"-"+Invest[i][1]+"\\*.txt"              
                elif Invest[i][10]=="others":
                      path_="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\output_pointcloud_every_epoch\\output_txt\\*.txt"
            reference_trajectory= np.loadtxt('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\extractet_traj_points.txt', delimiter=",")
       
        name_1=len(path_)-5   #Test_heading_m045 的地址字符数
        name_2=len(path_)     #sensor scan point cloud 地址字符数
    
        # Investigate heading range(0,1) if you do not want to investigate heading
        rots_x = np.arange(0,1,1)   #np.arange  形成一个array [0]
        rots_y = np.arange(0,1,1)
        rots_z =np.arange(-1.5,1.75,0.5)   #rots_z=[-1.5,-1.0,-0.5,0,0.5,1.0,1.5]
        z_steps=np.arange(0,1,1)
  
        number_rotations=rots_x.shape[0]*rots_y.shape[0]*rots_z.shape[0]   #旋转的可能数目 只有绕z轴的几个方向 旋转离散的 
        rotation_combinations=np.concatenate((np.repeat(rots_x,rots_y.shape[0]*rots_z.shape[0])[:,None],np.tile(np.repeat(rots_y,rots_z.shape[0]),rots_x.shape[0])[:,None]),axis=1)
        rotation_combinations=np.concatenate((rotation_combinations,np.tile(rots_z,rots_x.shape[0]*rots_y.shape[0])[:,None]),axis=1)
        rotation_combinations=np.concatenate((np.tile(rotation_combinations,(z_steps.shape[0],1)),np.repeat(z_steps,(rotation_combinations.shape[0]))[:,None]),axis=1)
        rotation_combinations=np.concatenate((rotation_combinations,np.arange(number_rotations*z_steps.shape[0])[:,None]),axis=1)
        # np.concatenate 沿着某个轴数组拼接 np.repeat 重复   [:,None] 新建一个维度，就是把里面每个元素再括起来
        #np.tile(a,(3,2)) 构造3*2个copy
        
        
        #### Read file names of sensor scans ####
        files = glob.glob(path_)  #获得所有sensor scans 的文件列表
        files=sorted(files)#files=natsort.natsorted(files)   #files是由小到大排列的sensor scan 文件们
        maxpcl=np.size(files)  #maxpcl 点云文件数目
        files=np.array(files)    #路径数组  
        concatenated =list(range(64,maxpcl,stepsize))  #序列 0到maxpcl
        concatenated=np.asarray(concatenated,dtype=int)   #list->array
        concatenated=concatenated[:,None]   #变成一列 每行之后一个array元素
            
      
        #### Angle to select normal vectors. This is used to delete ground points. ####
        up_vector_angle =10.0
        cos_ub = np.cos(np.deg2rad(up_vector_angle))
        cos_lb = np.cos(np.deg2rad(180.0 - up_vector_angle))
    
        #### Radius of max cons search. Result will be from -radius to +radius. ####
        max_cons_radius = 1.0
    
        
        #### The grid size of the max cons grid. #####
        max_cons_grid_edge = Invest[i][6]    #0.04
        
        
        #### Save files and setup savepath ####
        path_o = os.path.normpath(path_)
        path_split = path_o.split(os.sep)
        folder = path_split[-2]
        Save = True
        savepath_="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao" 
        if Invest[i][7]:    #simulation
             if Invest[i][10]=="slab":
                    savepath=savepath_+'\\slabmc-'+str(Invest[i][8])+'-'+map_+'\\'
                    savepath_mc=savepath+"Zisen_Zhaomc\\"
                    readpath_mc=savepath_mc 
             elif Invest[i][10]=="bresenham":
                    savepath=savepath_+'\\bresenhammc-'+str(Invest[i][8])+'-'+map_+'\\'
                    savepath_mc=savepath+"Zisen_Zhaomc\\"
                    readpath_mc=savepath_mc 
             elif Invest[i][10]=="others":
                    savepath=savepath_+'\\bresenham_others_mc'+'-'+map_+'\\'
                    savepath_mc=savepath+"Zisen_Zhaomc\\"
                    readpath_mc=savepath_mc 
             if os.path.isdir(savepath)==False:
                    os.mkdir(savepath)
                    savepath_mc=savepath+"Zisen_Zhaomc\\"
             if os.path.isdir(savepath_mc)==False:
                    os.mkdir(savepath_mc)
                    
        else:    
                readpath_mc=savepath_+str(i)+sensor+"_"+map_+"_"+NV+"_GRSI"+str(int(max_cons_grid_edge*100))+"_"+Invest[i][3]+"_"+Invest[i][4]+"_"+Invest[i][5]+"_"+Invest[i][9]+"_"+folder+"/"+"mc/"
               
                if Save:
                    savepath=savepath_+str(i)+sensor+"_"+map_+"_"+NV+"_GRSI"+str(int(max_cons_grid_edge*100))+"_"+Invest[i][3]+"_"+Invest[i][4]+"_"+Invest[i][5]+"_"+Invest[i][9]+"_"+folder+"/"
                    if os.path.isdir(savepath)==False:
                        os.mkdir(savepath)
                    savepath_mc=savepath+"mc/"
                    if os.path.isdir(savepath_mc)==False:
                        os.mkdir(savepath_mc)

                
                
        #### Adapt zrange to map resolution ####    
        # In this implementation, z is aligned already. When aligning x/y,
        # we will search only for corresponting points in a range +/- zrange.
        if map_=="LOD0":
            max_cons_z_range = 0.05
        elif map_=="LM":
            max_cons_z_range = 0.05
        elif map_=="Veg_rem":
            max_cons_z_range = 0.10
        elif map_=="LOD2" or "classified":
            max_cons_z_range = 0.10
        elif map_=="LOD3":
            max_cons_z_range = 0.20
        elif map_=="LOD4":
            max_cons_z_range = 0.40
        elif map_=="LOD5":
            max_cons_z_range = 0.80   
        elif map_=="Orig":
            max_cons_z_range = 0.05
    
        
                     
        f = open(savepath_mc+"log.txt", "w")
        f.write(str([rots_x,rots_y,rots_z,z_steps])) 
        f.close()
        
        num_cores = multiprocessing.cpu_count()
        print("NUM_CORES:")
        print(num_cores)           
        
        
        #### Define settings #### 
        # Only use one neighbor (map point) per grid cell per scan point
        if Invest[i][3] =="single_consideration":
            single_consideration=True
        else:
            single_consideration=False
            
        # Remove ground in car sensor scan (in map the ground is always removed)#
        remove_ground_car_sensor_scan = True
        
        # Do coarse-to-fine
        ctf = False
        
        # Save IMG with point cloud
        img_pcl = True
        
        # Downsampling of car sensor scan
        if Invest[i][5] =="downsampling":
            downsampling_carscan = True
        else:
            downsampling_carscan = False
        
        # Visualize confidence ellipse in mc matrix
        vis_conf_ell_mc = True
        
        # Save backprojection
        save_backprojection = True
        
        # ICP on inlier
        ICP = False
        
        # Maximum consensus matrix is filled by 1/trace(Qxx)
        if Invest[i][4] =="covariance":
            use_covariance_score=True
        else:
            use_covariance_score=False
        
        ##### Initialize lists for the loop over all car sensor scans ####
        idx_max_all_list=[]
        idx_max_max_list=[]
        start = 0
        cov_xy=[]
        
        matrix_radius_ = int(np.ceil(max_cons_radius / max_cons_grid_edge))
        distribution_map = np.zeros((2*matrix_radius_+1, 2*matrix_radius_+1))
            
        #Calculate maximum consensus matrices or load those from files
        mode="calculate_mc"#"calculate_mc"#"read_mc"
        
        if mode== "calculate_mc":
            #### map: convert open3d format to numpy array and remove ground ####
            radius_remove_ground=0.4 
            max_nn_=1000
            pcl_map_load.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_remove_ground, max_nn=max_nn_))
            pcl_map_numpy = np.asarray(pcl_map_load.points)
            pcl_map_numpy_normals = np.asarray(pcl_map_load.normals)
            map_pcl=np.concatenate((pcl_map_numpy, pcl_map_numpy_normals),axis=1)
            h_map_pcl = map_pcl[np.logical_and(map_pcl[:,5] >= cos_lb, map_pcl[:,5] <= cos_ub)]
            points_h_map_pcl = o3d.geometry.PointCloud()
            points_h_map_pcl.points = o3d.utility.Vector3dVector(h_map_pcl[:,0:3])#
            points_h_map_pcl.normals = o3d.utility.Vector3dVector(h_map_pcl[:,3:6])
           
            
            #### build the kdtree of the map ####
            z_scale = max_cons_radius / max_cons_z_range
            h_map_pcl[:,2] *= z_scale
            tree_ = cKDTree(h_map_pcl[:,0:3])

            
            ### for-loop over all car sensor scans ####
            for name in files[concatenated]:  #files[concatenated]是点云数据
                calculate_mc(name[0])
             

        elif mode== "read_mc":
            #falls normale mc auch mal eingelesen werden müssen --> dann aber rot schleife noch mit drum
            for name in files[concatenated]:   #在点云中循环  09088global--- 最后一个位置扫描点云
                name=name[0]     # name是个array 所以name[0]代表文件
                best_important_facts=np.loadtxt(readpath_mc+"best_important_facts"+str(name[name_1:name_2])+".csv", delimiter=",")  
                print(str(name[name_1:name_2])+" done!")
                best_idx1 = int(best_important_facts[0])
                best_idx2 = int(best_important_facts[1])
                best_rot_x = best_important_facts[2]
                best_rot_y = best_important_facts[3]
                best_rot_z = best_important_facts[4]
                mult_peaks_best = int(best_important_facts[5])
                max_prev = best_important_facts[6]
                mult_rots = best_important_facts[7]
                best_z = best_important_facts[8]
                if mult_rots==1:
                    print("attention! mult rots==1")
                    #break
                
                best_rotcomb=np.asarray([best_rot_x, best_rot_y, best_rot_z])
                #mc_best_rot=np.loadtxt(readpath_mc+"mc_best_rot_"+str(name[name_1:name_2])+"_"+str(int(best_rot_x))+str(int(best_rot_y))+str(int(best_rot_z))+".csv", delimiter=',')
                mc_best_rot=np.loadtxt(readpath_mc+"mc_best_rot_"+str(name[name_1:name_2])+"_"+str(best_rot_x)+str(best_rot_y)+str(best_rot_z)+".csv", delimiter=',')
    
                if ICP or save_backprojection:
                    #best_backprojection=np.loadtxt(readpath_mc+"best_backprojection_"+str(name[name_1:name_2])+"_"+str(int(best_rot_x))+str(int(best_rot_y))+str(int(best_rot_z))+".csv", delimiter=',')
                    best_backprojection=np.loadtxt(readpath_mc+"best_backprojection_"+str(name[name_1:name_2])+"_"+str(best_rot_x)+str(best_rot_y)+str(best_rot_z)+".csv", delimiter=',')
                #best_rotated_points=np.loadtxt(readpath_mc+"best_rotated_points_"+str(name[name_1:name_2])+"_"+str(int(best_rot_x))+str(int(best_rot_y))+str(int(best_rot_z))+".csv", delimiter=',')
                best_rotated_points=np.loadtxt(readpath_mc+"best_rotated_points_"+str(name[name_1:name_2])+"_"+str(best_rot_x)+str(best_rot_y)+str(best_rot_z)+".csv", delimiter=',')
       
                
                
                #### confidence ellipse ####
                mc_thresh=sort_with_threshold(mc_best_rot)
                cov,cov_array_pixel,cov_array,x_mean,y_mean=calc_ellipse(mc_best_rot,mc_thresh,max_cons_grid_edge)
 
                #### plot consensus matrix (best rotation) (without and with conf. ellipse)####
                plot_mc(mc_best_rot,best_rotcomb,savepath,str(name[name_1:name_2]),Save,vis_conf_ell_mc,cov_array_pixel)
                
                #### create height map ####
                create_height_map(mc_best_rot,savepath,str(name[name_1:name_2]),best_rotcomb,Save)
                
                #### plot point cloud next to consensus matrix image ####
                #best_rotated_points beinhalten normal vectors und daher sind in der create_img_pcl funktion noch anpassungen erforderlich
                create_img_pcl(mc_best_rot,best_rotated_points,savepath,str(name[name_1:name_2]),img_pcl,Save)
                
                #### entropie #### alle gleich-->entopie=1, ein peak--> entopie=0
                #entropie=calculate_entropy(mc_best_rot)
            
                #### evaluation based on confidence ellipse ####
                cov_xy.append([name[name_1:name_2],cov[0,0],cov[1,1],cov[0,1],cov[1,0],x_mean,y_mean,int(best_idx1),int(best_idx2),best_rot_x,best_rot_y,best_rot_z,mult_rots,best_z,cov_array])
                
            # if Save:
            #     cov_xy_array=[]
            #     cov_xy_=[]
            #     cov_xy_save=cov_xy
            #     for i in range(len(cov_xy)):
            #         cov_xy_array.append(cov_xy[i][14])
            #         cov_xy_.append(cov_xy[i][0:15])
            #     # cov_xy=np.asarray(cov_xy_,dtype=float)
            #     np.savetxt(readpath_mc+"cov_xy.csv",cov_xy_, delimiter=",")
                                
