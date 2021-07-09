# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:56:59 2021

@author: 123
"""
import numpy as np
import glob
import open3d as o3d

path="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\output_pointcloud_every_epoch\\output_pointcloud_every_epoch\\*.ply"
files=glob.glob(path)
files.sort()
max_epoch=np.size(files)
   #路径数组
files.sort(key=lambda x:int(x[121:-4]))  
stepsize=1
concatenated =list(range(0,max_epoch,stepsize))  #序列 0到maxpcl
concatenated=np.asarray(concatenated,dtype=int)   #list->array
concatenated=concatenated[:,None]   #变成一列 每行之后一个array元素
count=8921
files=np.asarray(files)
for name in files[concatenated]:
    pcl_map_load = o3d.io.read_point_cloud(name[0])  #在open3d里面加载地图，得到点云
    points=np.asarray(pcl_map_load.points)
    np.savetxt("E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\output_pointcloud_every_epoch\\output_txt\\"+str(count).zfill(5)+"_epoch.txt",points,fmt='%.14f',delimiter=",")
    count+=1