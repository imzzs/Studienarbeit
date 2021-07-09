# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:46:02 2021

@author: 123
"""
import numpy as np
import open3d as o3d
import glob

def get_list_voxel(voxel_grid): #voxel_grid 得到其中每一个voxel的index shape（1,3)
    voxel_grid=voxel_grid.get_voxels()
    list=np.asarray(voxel_grid)
    Index_list=[]
    for i in list:
        Index_list.append(i.grid_index.tolist())
    Index_list=np.array(Index_list)
    return Index_list    




def get_geo_coordinate_voxel(voxel_grid):   #输入是voxel_grid
    origin=voxel_grid.origin     #origin 是index
    Index_list=get_list_voxel(voxel_grid)
    origin_index=voxel_grid.get_voxel(origin)
    coor_list=(Index_list-origin_index)*voxel_grid.voxel_size+origin
    return coor_list      


Invest=["0.05","classified","bresenham"]
voxel_size=0.5
path_simulation="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\bresenham-"+Invest[0]+"-"+Invest[1]+"\\*.txt"
path_real="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\trajectory_utm_adj\\trajectory_utm_adj\\*.txt"
name_simu_e=len(path_simulation)
name_simu_s=len(path_simulation)-5
name_real_e=len(path_simulation)
name_real_s=len(path_simulation)-5

simulation_files=glob.glob(path_simulation)
data_files=glob.glob(path_real)
max_size=np.size(simulation_files)
simulation_files=np.array(simulation_files)
concatenated=list(range(0,1,1))
concatenated=np.asarray(concatenated,dtype=int)
concatenated=concatenated[:,None]   #变成独立元素
count=8921
for simulation in simulation_files[concatenated]:    #simualation_files should be arrays, also concatenated
    simulation_data=np.loadtxt(simulation[0],delimiter=",")
    real_data=np.loadtxt("E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\trajectory_utm_adj\\trajectory_utm_adj\\"+str(simulation[0][name_simu_s:name_simu_e])+"_global.txt",delimiter=',')
    simu_pcd=o3d.geometry.PointCloud()
    simulation_data=o3d.utility.Vector3dVector(simulation_data)
    simu_pcd.points=simulation_data
    real_data=o3d.utility.Vector3dVector(real_data)
    real_pcd=o3d.geometry.PointCloud()
    real_pcd.points=real_data
    
    simu_voxel=o3d.geometry.VoxelGrid.create_from_point_cloud(simu_pcd,voxel_size)
    real_voxel=o3d.geometry.VoxelGrid.create_from_point_cloud(real_pcd,voxel_size)
    origin_x=min(simu_voxel.origin[0],real_voxel.origin[0])
    origin_y=min(simu_voxel.origin[1],real_voxel.origin[1])
    origin_z=min(simu_voxel.origin[2],real_voxel.origin[2])
    origin_simu=simu_voxel.get_voxel([origin_x,origin_y,origin_z])
    origin_real=real_voxel.get_voxel([origin_x,origin_y,origin_z])  #array([x,y,z])
    simu_voxels=simu_voxel.get_voxels()
    real_voxels=real_voxel.get_voxels()
    simu_index=[]
    real_index=[]
    result=[]
    for i in simu_voxels:
        voxel_index=i.grid_index-origin_simu   #array([x,y,z])
        simu_index.append(voxel_index.tolist())
    for k in real_voxels:
        data_index=k.grid_index-origin_real   #array([x,y,z])    index based on the origin it should be
        real_index.append(data_index.tolist())
    for j in real_index:
        for k in simu_index:
            if j==k:
                x=k[0]*voxel_size+origin_x
                y=k[1]*voxel_size+origin_y
                z=k[2]*voxel_size+origin_z
                result.append([x,y,z])
    # for l in real_index:
    #     x=l[0]*voxel_size+origin_x
    #     y=l[1]*voxel_size+origin_y
    #     z=l[2]*voxel_size+origin_z
    #     result.append(np.array([x,y,z]))
    # simulation=get_geo_coordinate_voxel(simu_voxel)
    # real=get_geo_coordinate_voxel(real_voxel)    
        
                
    
    # simu_list=get_geo_coordinate_voxel(simu_voxel)
    # real_voxel=get_geo_coordinate_voxel(real_voxel)
    np.savetxt('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\test\\'+str(count).zfill(5)+"_result.txt",np.asarray(result),fmt='%.14f',delimiter=",")
    # np.savetxt('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\test\\'+str(count).zfill(5)+"_real.txt",np.asarray(real),fmt='%.14f',delimiter=",")
    
                
    
    
    
    
