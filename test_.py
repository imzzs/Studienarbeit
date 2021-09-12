# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:18:39 2021

@author: 123
"""

import open3d as o3d
import numpy as np
import math
import glob
import Sensor_total
import time
import numba
from numba import jit,njit
import functools
from functools import lru_cache
from numba.experimental import jitclass
from numba import boolean,float64
from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects
import multiprocessing
from bitarray import bitarray
from bitarray import util

#@jitclass(spec)   数据中包含点云数据类型，无法进行加速
class OctreeNode(object):
    def __init__(self,minbound,maxbound,pointcloud,depth,root=False):    # 每个node包含了自己的position
        self.root=root
        self.node_size=0  #占位
        if root: 
           self.minbound=pointcloud.get_min_bound()    # np.array 最小点坐标        
           self.node_size=max(pointcloud.get_max_bound()-self.minbound)
           self.maxbound=self.minbound+self.node_size
        else:
           self.minbound=minbound
           self.maxbound=maxbound
           self.node_size=max(maxbound-minbound)
        self.pointcloud=pointcloud 
        # self.children=self.set_children()   #包含每个孩子节点  
        if np.asarray(pointcloud.points).size: 
                self.leaf=False
        else:
            self.leaf=True
        
        self.child0=None
        self.child1=None
        self.child2=None
        self.child3=None
        self.child4=None
        self.child5=None
        self.child6=None
        self.child7=None
        self.depth=depth
        # self.node_size=max(self.maxbound-self.minbound)
        # self.center=self.node_size+self.minbound
    # def set_node_size(self):
    #     node_size=max(self.maxbound-self.minbound)
    #     return node_size
    
    def set_center(self):
        center=self.node_size/2+self.minbound
        return center
    
def idx_save_points(tuple_idx,pcd):      
    list=pcd.select_by_index(tuple_idx[0])    #tuple 是 np.where 的结果
    points=np.asarray(list.points)
    return points
    
@functools.lru_cache(None)   
def set_octree(Node):
    array=np.array
    narray=np.asarray
    where=np.where
    if  Node.leaf:
        return None
    elif Node.depth==1:
        return Node   #叶子节点是没有点云的   ，不存在的节点
    else:
        depth_=Node.depth-1

        # if Node.depth==1:
        #     Node.leaf=True    #只有没有pcd 的节点才算叶子节点，depth=1的不算 
        # node_size=self.set_node_size()
        print(depth_)
        center=Node.set_center()
        pcd0=o3d.geometry.PointCloud()
        pcd1=o3d.geometry.PointCloud()
        pcd2=o3d.geometry.PointCloud()
        pcd3=o3d.geometry.PointCloud()
        pcd4=o3d.geometry.PointCloud()
        pcd5=o3d.geometry.PointCloud()
        pcd6=o3d.geometry.PointCloud()
        pcd7=o3d.geometry.PointCloud()
        # pcd_points=self.pointcloud.points      #vec3dvec

            
        # idx0_0 (array([],dtype=int64),)
        # idx0_1 array([],dype=int64)
        # idx0_2 array([],dtype=int64)
        # 八个孩子的点云数据 idx _2 是下标
        pcd=Node.pointcloud
        pcd_array=narray(pcd.points)  
        idx0_0=np.where(pcd_array[:,0]<=center[0])    #孩子节点0 x坐标 条件 1
        pcd0_0=pcd.select_by_index(idx0_0[0])
        pcd0_0_array=narray(pcd0_0.points)
        idx0_0_1=where(pcd0_0_array[:,0]>Node.minbound[0])   #孩子节点0 x坐标 条件2
        
        pcd0_1=pcd0_0.select_by_index(idx0_0_1[0])              #孩子节点0 y坐标
        pcd0_1_array=narray(pcd0_1.points)
        idx0_1=where(pcd0_1_array[:,1]<=center[1])       #0-y-1
        pcd0_1_1=pcd0_1.select_by_index(idx0_1[0])
        pcd0_1_1_array=narray(pcd0_1_1.points)
        idx0_1_1=where(pcd0_1_1_array[:,1]>Node.minbound[1])

        pcd0_2=pcd0_1_1.select_by_index(idx0_1_1[0])              #孩子节点0 z坐标
        pcd0_2_array=narray(pcd0_2.points)
        idx0_2=where(pcd0_2_array[:,2]<=center[2])
        pcd0_2_1=pcd0_2.select_by_index(idx0_2[0])              #孩子节点0 z坐标
        pcd0_2_array=narray(pcd0_2_1.points)    
        idx0_2_1=where(pcd0_2_array[:,2]>Node.minbound[2])



        idx1_0=where(pcd_array[:,0]>center[0])
        pcd1_0=pcd.select_by_index(idx1_0[0])
        pcd1_0_array=narray(pcd1_0.points)
        idx1_0_1=where(pcd1_0_array[:,0]<=Node.maxbound[0])
        
        pcd1_1=pcd1_0.select_by_index(idx1_0_1[0])              #孩子节点1 y坐标
        pcd1_1_array=narray(pcd1_1.points)
        idx1_1=where(pcd1_1_array[:,1]<=center[1])
        
        pcd1_1_1=pcd1_1.select_by_index(idx1_1[0])
        pcd1_1_1_array=narray(pcd1_1_1.points)
        idx1_1_1=where(pcd1_1_1_array[:,1]>Node.minbound[1])
        
        pcd1_2=pcd1_1_1.select_by_index(idx1_1_1[0])              #孩子节点1 z坐标
        pcd1_2_array=narray(pcd1_2.points)
        idx1_2=where(pcd1_2_array[:,2]<=center[2])
        
        pcd1_2_1=pcd1_2.select_by_index(idx1_2[0])              #孩子节点1 z坐标
        pcd1_2_array=narray(pcd1_2_1.points) 
        idx1_2_1=where(pcd1_2_array[:,2]>Node.minbound[2])      #记得最后输出的点云不是pcd 而是 pcd1_2_1


        idx2_0=where(pcd_array[:,0]<=center[0])
        pcd2_0=pcd.select_by_index(idx2_0[0])
        pcd2_0_array=narray(pcd2_0.points)
        idx2_0_1=where(pcd2_0_array[:,0]>Node.minbound[0])
        
        pcd2_1=pcd2_0.select_by_index(idx2_0_1[0])              #孩子节点2 y坐标
        pcd2_1_array=narray(pcd2_1.points)
        idx2_1=where(pcd2_1_array[:,1]>center[1])
        
        pcd2_1_1=pcd2_1.select_by_index(idx2_1[0])
        pcd2_1_1_array=narray(pcd2_1_1.points)
        idx2_1_1=where(pcd2_1_1_array[:,1]<=Node.maxbound[1])
        
        pcd2_2=pcd2_1_1.select_by_index(idx2_1_1[0])              #孩子节点2 z坐标
        pcd2_2_array=narray(pcd2_2.points)
        idx2_2=where(pcd2_2_array[:,2]<=center[2])
        
        pcd2_2_1=pcd2_2.select_by_index(idx2_2[0])              #孩子节点2 z坐标
        pcd2_2_array=narray(pcd2_2_1.points)
        idx2_2_1=where(pcd2_2_array[:,2]>Node.minbound[2])


        idx3_0=where(pcd_array[:,0]>center[0])
        pcd3_0=pcd.select_by_index(idx3_0[0])
        pcd3_0_array=narray(pcd3_0.points)
        idx3_0_1=where(pcd3_0_array[:,0]<=Node.maxbound[0])
        
        pcd3_1=pcd3_0.select_by_index(idx3_0_1[0])              #孩子节点3 y坐标
        pcd3_1_array=narray(pcd3_1.points)
        idx3_1=where(pcd3_1_array[:,1]>center[1])
        pcd3_1_1=pcd3_1.select_by_index(idx3_1[0])
        pcd3_1_1_array=narray(pcd3_1_1.points)
        idx3_1_1=where(pcd3_1_1_array[:,1]<=Node.maxbound[1])
        
        pcd3_2=pcd3_1_1.select_by_index(idx3_1_1[0])              #孩子节点3 z坐标
        pcd3_2_array=narray(pcd3_2.points)
        idx3_2=where(pcd3_2_array[:,2]<=center[2])
        pcd3_2_1=pcd3_2.select_by_index(idx3_2[0])            
        pcd3_2_array=narray(pcd3_2_1.points)
        idx3_2_1=where(pcd3_2_array[:,2]>Node.minbound[2])


        idx4_0=where(pcd_array[:,0]<=center[0])
        pcd4_0=pcd.select_by_index(idx4_0[0])
        pcd4_0_array=narray(pcd4_0.points)
        idx4_0_1=where(pcd4_0_array[:,0]>Node.minbound[0])
    
        pcd4_1=pcd4_0.select_by_index(idx4_0_1[0])              #孩子节点4 y坐标
        pcd4_1_array=narray(pcd4_1.points)
        idx4_1=where(pcd4_1_array[:,1]<=center[1])
        pcd4_1_1=pcd4_1.select_by_index(idx4_1[0])
        pcd4_1_1_array=narray(pcd4_1_1.points)
        idx4_1_1=where(pcd4_1_1_array[:,1]>Node.minbound[1])
        
        pcd4_2=pcd4_1_1.select_by_index(idx4_1_1[0])              #孩子节点4 z坐标
        pcd4_2_array=narray(pcd4_2.points)
        idx4_2=where(pcd4_2_array[:,2]>center[2])
        pcd4_2_1=pcd4_2.select_by_index(idx4_2[0])            
        pcd4_2_array=narray(pcd4_2_1.points)
        idx4_2_1=where(pcd4_2_array[:,2]<=Node.maxbound[2])

        idx5_0=where(pcd_array[:,0]>center[0])
        pcd5_0=pcd.select_by_index(idx5_0[0])
        pcd5_0_array=narray(pcd5_0.points)
        idx5_0_1=where(pcd5_0_array[:,0]<=Node.maxbound[0])
        
        pcd5_1=pcd5_0.select_by_index(idx5_0_1[0])              
        pcd5_1_array=narray(pcd5_1.points)
        idx5_1=where(pcd5_1_array[:,1]<=center[1])
        pcd5_1_1=pcd5_1.select_by_index(idx5_1[0])
        pcd5_1_1_array=narray(pcd5_1_1.points)
        idx5_1_1=where(pcd5_1_1_array[:,1]>Node.minbound[1])
        
        pcd5_2=pcd5_1_1.select_by_index(idx5_1_1[0])           
        pcd5_2_array=narray(pcd5_2.points)
        idx5_2=where( pcd5_2_array[:,2]>center[2])
        pcd5_2_1=pcd5_2.select_by_index(idx5_2[0])            
        pcd5_2_array=narray(pcd5_2_1.points)
        idx5_2_1=where( pcd5_2_array[:,2]<=Node.maxbound[2])
        
        idx6_0=where(pcd_array[:,0]<=center[0])
        pcd6_0=pcd.select_by_index(idx6_0[0])
        pcd6_0_array=narray(pcd6_0.points)
        idx6_0_1=where(pcd6_0_array[:,0]>Node.minbound[0])
        
        pcd6_1=pcd6_0.select_by_index(idx6_0_1[0])              
        pcd6_1_array=narray(pcd6_1.points)
        idx6_1=where(pcd6_1_array[:,1]>center[1])
        pcd6_1_1=pcd6_1.select_by_index(idx6_1[0])
        pcd6_1_1_array=narray(pcd6_1_1.points)
        idx6_1_1=where(pcd6_1_1_array[:,1]<=Node.maxbound[1]) 
        
        pcd6_2=pcd6_1_1.select_by_index(idx6_1_1[0])           
        pcd6_2_array=narray(pcd6_2.points)
        idx6_2=where(pcd6_2_array[:,2]>center[2])
        pcd6_2_1=pcd6_2.select_by_index(idx6_2[0])            
        pcd6_2_array=narray(pcd6_2_1.points)
        idx6_2_1=where(pcd6_2_array[:,2]<=Node.maxbound[2])
        
        
        idx7_0=where(pcd_array[:,0]>center[0])
        pcd7_0=pcd.select_by_index(idx7_0[0])
        pcd7_0_array=narray(pcd7_0.points)
        idx7_0_1=where(pcd7_0_array[:,0]<=Node.maxbound[0])
        
        pcd7_1=pcd7_0.select_by_index(idx7_0_1[0])              
        pcd7_1_array=narray(pcd7_1.points)
        idx7_1=where(pcd7_1_array[:,1]>center[1])
        pcd7_1_1=pcd7_1.select_by_index(idx7_1[0])
        pcd7_1_1_array=narray(pcd7_1_1.points)
        idx7_1_1=where(pcd7_1_1_array[:,1]<=Node.maxbound[1])
        
        pcd7_2=pcd7_1_1.select_by_index(idx7_1_1[0])           
        pcd7_2_array=narray(pcd7_2.points)
        idx7_2=where(pcd7_2_array[:,2]>center[2])
        pcd7_2_1=pcd7_2.select_by_index(idx7_2[0])            
        pcd7_2_array=narray(pcd7_2_1.points)
        idx7_2_1=where(pcd7_2_array[:,2]<=Node.maxbound[2])
        
        pcd0=pcd0_2_1.select_by_index(idx0_2_1[0])          # pcd0  pointcloud  
        pcd1=pcd1_2_1.select_by_index(idx1_2_1[0])
        pcd2=pcd2_2_1.select_by_index(idx2_2_1[0])
        pcd3=pcd3_2_1.select_by_index(idx3_2_1[0])
        pcd4=pcd4_2_1.select_by_index(idx4_2_1[0])
        pcd5=pcd5_2_1.select_by_index(idx5_2_1[0])
        pcd6=pcd6_2_1.select_by_index(idx6_2_1[0])
        pcd7=pcd7_2_1.select_by_index(idx7_2_1[0])
        
        
        child0_Node=OctreeNode(Node.minbound, center, pcd0,depth_)
        child1_Node=OctreeNode(Node.minbound+array([Node.node_size/2,0,0]),center+array([Node.node_size/2,0,0]), pcd1,depth_)
        child2_Node=OctreeNode(Node.minbound+array([0,Node.node_size/2,0]), center+array([0,Node.node_size/2,0]), pcd2,depth_)
        child3_Node=OctreeNode(Node.minbound+array([Node.node_size/2,Node.node_size/2,0]), center+array([Node.node_size/2,Node.node_size/2,0]), pcd3,depth_)
        child4_Node=OctreeNode(Node.minbound+array([0,0,Node.node_size/2]), center+np.array([0,0,Node.node_size/2]), pcd4,depth_)
        child5_Node=OctreeNode(Node.minbound+array([Node.node_size/2,0,Node.node_size/2]), center+array([Node.node_size/2,0,Node.node_size/2]), pcd5,depth_)
        child6_Node=OctreeNode(Node.minbound+array([0,Node.node_size/2,Node.node_size/2]), center+array([0,Node.node_size/2,Node.node_size/2]), pcd6,depth_)
        child7_Node=OctreeNode(Node.minbound+array([Node.node_size/2,Node.node_size/2,Node.node_size/2]), center+array([Node.node_size/2,Node.node_size/2,Node.node_size/2]), pcd7,depth_)
        
        Node.child0=set_octree(child0_Node)
        Node.child1=set_octree(child1_Node)
        Node.child2=set_octree(child2_Node)
        Node.child3=set_octree(child3_Node)
        Node.child4=set_octree(child4_Node)
        Node.child5=set_octree(child5_Node)
        Node.child6=set_octree(child6_Node)
        Node.child7=set_octree(child7_Node)
        
  
        return Node  #那就不存在叶子节点，只要没有pcd的叶子节点在创建树的时候返回的是None  深度达到的仍有点云的节点不叫叶子节点 只是孩子返回none


       
def print_children(root):
    print([root.child0,root.child1,root.child2,root.child3,root.child4,root.child5,root.child6,root.child7])      
     
def distance_point_plane(point,plane):
    distance=np.absolute(plane[0]*point[0]+plane[1]*point[1]+plane[2]*point[2]+plane[3])/np.sqrt(np.sum(plane[:3]**2))   # plane ax+by+cz+d=0  distance=|ax0+by0+cz0+d|/sqrt(a**2+b**2+c**2)
    return distance

@jit(nopython=True)
def t_distance(sensor_position,plane,unit_vector): # the distance along the ray from sensor to the plane
    if plane[0]!=0:
        t=(-plane[3]-sensor_position[0])/unit_vector[0]   #in the function intersection, unit_vector[0]=0 is excluded
    elif plane[1]!=0:
        t=(-plane[3]-sensor_position[1])/unit_vector[1]
    elif plane[2]!=0:
        t=(-plane[3]-sensor_position[2])/unit_vector[2]
    return t
                   

@jit(nopython=True)
def intersection(voxel_position,sensor_position,unit_vector,voxel_size):   #unit_vector 是28800个array 针对输入的节点
    array=np.array
    nmax=np.max
    nmin=np.min
    plane1=array([1,0,0,-voxel_position[0]])
    plane2=array([0,1,0,-voxel_position[1]])
    plane3=array([0,0,1,-voxel_position[2]])
    plane4=array([1,0,0,-voxel_position[0]-voxel_size])
    plane5=array([0,1,0,-voxel_position[1]-voxel_size])
    plane6=array([0,0,1,-voxel_position[2]-voxel_size])
    if sensor_position[0]>=voxel_position[0] and sensor_position[0]<=voxel_position[0]+voxel_size:
        if sensor_position[1]>=voxel_position[1] and sensor_position[1]<=voxel_position[1]+voxel_size:
            if sensor_position[2]>=voxel_position[2] and sensor_position[2]<=voxel_position[2]+voxel_size:
                return True
    # tmin=np.float('-inf')
    # tmax=np.float('inf')
    
    tmin=-100000
    tmax=100000
    if unit_vector[0]==0:      #if ray parallel to plane yoz
       if sensor_position[0]<(voxel_position[0]) or sensor_position[0]>(voxel_position[0]+voxel_size):
           return False
#在这里 有一种情况没有考虑
    else:
        t1=t_distance(sensor_position,plane1,unit_vector)
        t4=t_distance(sensor_position,plane4,unit_vector)
        tmin1=nmin(array([t1,t4]))
        tmin=nmax(array([tmin1,tmin]))       #此处numba 要求 np.min()其中必须为array
        tmax1=nmax(array([t1,t4]))
        tmax=nmin(array([tmax,tmax1]))
    
    if unit_vector[1]==0:      #if ray parallel to plane xoz
        if sensor_position[1]<(voxel_position[1]) or sensor_position[1]>(voxel_position[1]+voxel_size):
            return False

    else:
        t2=t_distance(sensor_position,plane2,unit_vector)
        t5=t_distance(sensor_position,plane5,unit_vector)
        tmin2=nmin(array([t2,t5]))
        tmin=nmax(array([tmin2,tmin]))
        tmax2=nmax(array([t2,t5]))
        tmax=nmin(array([tmax,tmax2]))
 
    if unit_vector[2]==0:      #if ray parallel to plane xoz
        if sensor_position[2]<(voxel_position[2]) or sensor_position[2]>(voxel_position[2]+voxel_size):
            return False

    else:
        t3=t_distance(sensor_position,plane3,unit_vector)
        t6=t_distance(sensor_position,plane6,unit_vector)
        tmin3=nmin(array([t3,t6]))
        tmin=nmax(array([tmin3,tmin]))
        tmax3=nmax(array([t3,t6]))
        tmax=nmin(array([tmax,tmax3]))
      
    if  tmin<0 or tmax<0 or  tmin>tmax or tmin>100:
        return False
    else: 
        return tmin




def init(Node,sensor_position):
    if Node==None:
        return False
    else:
        if sensor_position[0]>Node.minbound[0] and sensor_position[0]<Node.maxbound[0]:
            if sensor_position[1]>Node.minbound[1] and sensor_position[1]<Node.maxbound[1]:
                if sensor_position[2]>Node.minbound[2] and sensor_position[2]<Node.maxbound[2]:
                    return True
                else: 
                    return False
            else:
                return False
        else:
            return False
       
def order_traverse(vmask):
    for i in range(8):
        a=util.int2ba(i,3)
        b=a^vmask
        c=util.ba2int(b)
        print(c)


def find_point(OctreeNode,sensor_position,unit_vector,point=None):    #通过判断返回 candidate_list
    point_array=np.asarray(OctreeNode.pointcloud.points)
    if len(point_array)==1:
        return point_array[0]
    distance=distance_point_ray(point_array,sensor_position,unit_vector)
    
    min_distance=np.amin(distance)
    print(min_distance)
    index=np.where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
    point=point_array[index[0]]    
    
    return point


def dfs_traverse(OctreeNode,sensor_position,unit_vector,tmin=float('inf')):    #通过判断返回 candidate_list
    point=None
    narray=np.asarray
    amin=np.amin
    where=np.where
    if not OctreeNode:
        pass
    else:
        vmask=bitarray(3)
        if np.sign(unit_vector[0])<0:
            vmask[2]=1
        else:
            vmask[2]=0
        if np.sign(unit_vector[1])<0:
            vmask[1]=1
        else:
            vmask[1]=0
        if np.sign(unit_vector[2])<0:
            vmask[0]=1
        else:
            vmask[0]=0
        #vmaks=bitarray([0,0,0])    
        if vmask==bitarray('000'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)                     
                       return point
                else:
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)  
                    
               
        elif vmask==bitarray('001'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point
                else:
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)  
       
            

        elif vmask==bitarray('010'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point
                else:
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)  
        elif vmask==bitarray('011'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point

                else:
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector) 
                  
        elif vmask==bitarray('100'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point
          
                else:
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)   
               

        elif vmask==bitarray('101'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point
        
                else:
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)    
        elif vmask==bitarray('110'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point
                
            
                else:
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)  

        elif vmask==bitarray('111'):
            t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)  
            if t:
                if OctreeNode.depth==1:
                    point_array=narray(OctreeNode.pointcloud.points) 
                    distance=distance_point_ray(point_array,sensor_position,unit_vector)
                    min_distance=amin(distance)
                    if min_distance<=0.02:
                       index=where(distance[:]==min_distance)[0]  # np.where 之后要取一个 [0]
                       point=point_array[index[0]] 
                       #print(min_distance)
                       return point
                else:
                    point=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector)
                    if point is not None:
                        return point
                    point=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector)   
    
    return point



def dfs_traverse2(OctreeNode,sensor_position,unit_vector,tmin=float('inf'),Node=None):    #通过判断返回 candidate_list

    if not OctreeNode:
        pass
    else:
        t=intersection(OctreeNode.minbound,sensor_position,unit_vector,OctreeNode.node_size)
        if t:
            if OctreeNode.depth==1:
                if t<=tmin:
                    tmin=t
                    Node=OctreeNode
            else:
                Node,tmin=dfs_traverse(OctreeNode.child0,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child1,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child2,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child3,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child4,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child5,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child6,sensor_position,unit_vector,tmin,Node)
                Node,tmin=dfs_traverse(OctreeNode.child7,sensor_position,unit_vector,tmin,Node)  
    return Node,tmin




        
def angle_lines(vector1,vector2):   #vector1 是所有点都在的 n*3  array  
    dot=vector1.dot(vector2)   # n*1  array
    norm_1=np.linalg.norm(vector1,axis=1)    #n*1 array
    norm_2=1
    cos=dot/(norm_1*norm_2)
    radian=np.arccos(cos)
    return radian


def distance_point_ray(point_array,position,unit_vector):      # point  nd array
    vector=point_array-position                                #position of sensor       
    radian=angle_lines(vector,unit_vector)     # n*1 array
    area=np.linalg.norm(vector,axis=1)*np.linalg.norm(unit_vector,axis=0)*np.sin(radian)*np.sign(np.cos(radian))    #利用余弦符号去掉 反方向的点
    distance=area/np.linalg.norm(unit_vector) 
    return distance    
    
# def distance_point_ray(points_list,position,unit_vector):    # point_list ndarray 28800 个ray 各自的相交的节点  包括了None
#     vector_list=[i-position ]


def traverse(OctreeNode,candidate_list):     #返回所有孩子节点
    if OctreeNode==None:
        pass
    else:
        if OctreeNode.depth==1:
           candidate_list.append(OctreeNode)

        else:
           traverse(OctreeNode.child0,candidate_list)
           traverse(OctreeNode.child1,candidate_list)
           traverse(OctreeNode.child2,candidate_list)
           traverse(OctreeNode.child3,candidate_list)
           traverse(OctreeNode.child4,candidate_list)
           traverse(OctreeNode.child5,candidate_list)
           traverse(OctreeNode.child6,candidate_list)
           traverse(OctreeNode.child7,candidate_list)
    return candidate_list

def depth_change(root):
    if root==None:
        pass
    else:
       if root.depth==1:
            root=None
       else:
            root.depth-=1
            depth_change(root.child0)
            depth_change(root.child1)
            depth_change(root.child2)
            depth_change(root.child3)
            depth_change(root.child4)
            depth_change(root.child5)
            depth_change(root.child6)
            depth_change(root.child7)


def find_minimum(candidate_list,position,unit_vector,min_distance=float('inf')):   #找到最近节点以及距离
    if candidate_list==[]:
        return None
    for node in candidate_list:
        distance=np.linalg.norm(node.minbound-position,axis=0)
        print(distance)
        if distance<min_distance:
            min_distance=distance
            nearest_node=node
    return nearest_node
          
def write_position(candidate_list):    #给一个list 元素是 node 得到每个node的位置并保存
    node_position=[]
    count=1
    save_path="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\"
    for node in candidate_list:
        node_position.append(node.minbound)
        node_position.append(node.maxbound)
        print(count)
        count+=1
    position_array=np.asarray(node_position)
    np.savetxt(save_path+"_nodeposition.txt",position_array,fmt="%.14f",delimiter=",")
        
def write_pointcloud(candidate_list):    # 给一个candidate (node)  把所有包含点云 保存 
    pointcloud=np.array([[1,1,1]])
    count=0
    for node in candidate_list:
        if node.pointcloud.points:
            pointcloud=np.asarray(node.pointcloud.points)
            # count+=len(np.asarray(node.pointcloud.points))
            np.savetxt(save_path+"test\\"+str(count)+"_pcd.txt",pointcloud,fmt="%.14f",delimiter=",")
            count+=1
            print(count)
        else:
            continue

def point_on_ray(sensor_position,ray,t):
    point=sensor_position+ray*t
    return point

    
    
def traverse_position(root,depth_,position_list):   # 给一个树， 以及遍历高度。 保存每个节点的minbound 以及maxbound  z.B. depth_=10, root.depth=12 那么遍历到root的孩子们的bound
    if root==None or root.depth==depth_:
       pass
    else:
        position_list.append(root.minbound)
        position_list.append(root.maxbound)
        traverse_position(root.child0,depth_,position_list)
        traverse_position(root.child1,depth_,position_list)
        traverse_position(root.child2,depth_,position_list)
        traverse_position(root.child3,depth_,position_list)
        traverse_position(root.child4,depth_,position_list)
        traverse_position(root.child5,depth_,position_list)
        traverse_position(root.child6,depth_,position_list)
        traverse_position(root.child7,depth_,position_list)
        

    
        

def loop_ray(unit_vector_world,sensor_position,root):
    intersected_list=[]
    start=time.time()
    append=intersected_list.append
    count=0
    for ray in unit_vector_world:
        point=dfs_traverse(root,sensor_position,ray)
        if point is not None:
           append(point)
        count+=1
        print(count)
    end=time.time()
    print("runtime is: ",end-start)
    return intersected_list




@njit
def ident_loops(x):
    r = np.empty_like(x)
    n = len(x)
    for i in range(n):
        r[i] = np.cos(x[i]) ** 2 + np.sin(x[i]) ** 2
    return r     


# @jit(nopython=True)        
if __name__=="__main__":
    map_="Merged_122831_122944_123114"  #Merged_122831_122944_123114
    pcd=o3d.io.read_point_cloud('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\maps\\'+map_+'.ply')
    reference_trajectory= np.loadtxt('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\extractet_traj_points.txt', delimiter=",")
    save_path="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\"
    sensor_position_array=reference_trajectory[:,2:5]
    start_tree=time.time()
    root=OctreeNode(0,0,pcd,13,True)
    set_octree(root)
    end_tree=time.time()
    unit_vector_whole=Sensor_total.world_coordinate
    print("time of build a tree:", end_tree-start_tree,'s')
    start=8921
    narray=np.asarray
    for position in range(0,len(sensor_position_array)): 
        # sensor_start=time.time()
        sensor_position=sensor_position_array[position]
        unit_vector_world=unit_vector_whole[position]
        intersected_list=loop_ray(unit_vector_world,sensor_position,root)
        intersected_array=narray(intersected_list)
        np.savetxt(save_path+"octree_hero_lod4\\"+str(start).zfill(5)+"_simulation.txt",intersected_array,fmt='%.14f',delimiter=",")
        start+=1
        print(start," ","success")
            

        