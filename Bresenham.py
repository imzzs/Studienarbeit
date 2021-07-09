# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:24:29 2021

@author: 123
"""
import Sensor_total
import open3d as o3d
import numpy as np
import math
import os

class Workspace(object):
    def __init__(self,pcd,position,voxel_size):
        self.pcd=pcd
        self.position=position
        self.voxel_size=float(voxel_size)
    def pre_processing(self,pcd,remove_mode='radius',nb_neighbors=20,std_ratio=0.8,nb_points=100,radius=1):   #预处理，downsampling 以及 outlier removing 返回处理后的点云
        #downsample=pcd.voxel_down_sample(voxel_size)
        downsample=pcd
        if remove_mode=='statistical':
            cl,ind=downsample.remove_statistical_outlier(nb_neighbors,std_ratio)   #ind index of inlier
        if remove_mode=='radius':
            cl, ind = downsample.remove_radius_outlier(nb_points, radius)
        return cl
   


    def get_geo_coordinate_voxel(self,voxel_grid):   #输入是voxel_grid
        origin=voxel_grid.origin     #origin 是index
        Index_list=get_list_voxel(voxel_grid)
        origin_index=voxel_grid.get_voxel(origin)
        coor_list=(Index_list-origin_index)*voxel_grid.voxel_size+origin
        return coor_list
        # return coor_list       
    def voxelization(self,pcd,position_coor):

        downsampling=pcd
        coor_whole=np.asarray(downsampling.points)    
        # savepath_workspace="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\workspace"+str(self.position)+"\\"
        # if os.path.isdir(savepath_workspace)==False:
        #     os.mkdir(savepath_workspace)
        distance=np.linalg.norm(coor_whole-position_coor,axis=1)
        idx=np.where((distance<100))[0]
        candidate_coor=downsampling.select_by_index(idx)
        voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(candidate_coor,self.voxel_size)
        # coor_list=self.get_geo_coordinate_voxel(voxel_grid)
        return voxel_grid

    # def get_list_voxel(self,voxel_grid): #voxel_grid 得到其中每一个voxel的index shape（1,3)
    #     voxel_grid=voxel_grid.get_voxels()
    #     list=np.asarray(voxel_grid)
    #     Index_list=[]
    #     for i in list:
    #         Index_list.append(i.grid_index.tolist())
    #     Index_list=np.array(Index_list)
    #     return Index_list  
        


def get_list_voxel(voxel_grid,shift_vector): #voxel_grid 得到其中每一个voxel的index shape（1,3)
    voxel_grid=voxel_grid.get_voxels()
    list=np.asarray(voxel_grid)
    Index_list=[]
    for i in list:
        index=i.grid_index-shift_vector
        Index_list.append(index)   #index is array
    Index_list=np.array(Index_list)
    Index_list=Index_list.astype(int)
    return Index_list    

# set a array to save bool value for each voxel: occupied true not false
def vector_3d(voxel_grid,voxel_size,shift_vector):
    vector_voxel=np.zeros((int(200/voxel_size),int(200/voxel_size),int(50/voxel_size)),dtype=bool)   #200 200 40 试出来的
    Index_list=get_list_voxel(voxel_grid,shift_vector)
    for i in Index_list:
        if i[0]>=vector_voxel.shape[0]:
            continue
        if i[1]>=vector_voxel.shape[1]:
            continue
        if i[2]>=vector_voxel.shape[2]:
            continue
        vector_voxel[i[0],i[1],i[2]]=True
    return vector_voxel

def get_geo_coordinate(voxel_grid,voxel,shift_vector):
    origin=voxel_grid.origin     #origin 是utm 
    origin_index=-shift_vector   #此时 shift为原点
    x=(voxel[0]-origin_index[0])*voxel_grid.voxel_size+origin[0]
    y=(voxel[1]-origin_index[1])*voxel_grid.voxel_size+origin[1]
    z=(voxel[2]-origin_index[2])*voxel_grid.voxel_size+origin[2]
    coor=np.array([x,y,z])
    return coor
    origin=voxel_grid.origin     #origin 是index
    origin_index=voxel_grid.get_voxel(origin)
    x=(voxel[0]-origin_index[0])*voxel_grid.voxel_size+origin[0]
    y=(voxel[1]-origin_index[1])*voxel_grid.voxel_size+origin[1]
    z=(voxel[2]-origin_index[2])*voxel_grid.voxel_size+origin[2]
    coor=np.array([x,y,z])
    return coor

#bresenham simulate the ray one by one voxel until we reach a voxel whose bool value (from vector_3d) is true 
def bresenham(start,endposition_voxel,workspace_voxel,vector_voxel):
     vector=endposition_voxel-start
     dx=vector[0]
     dy=vector[1]
     dz=vector[2]
     x=start[0]
     y=start[1]
     z=start[2]
     xmax=np.shape(vector_voxel)[0]
     ymax=np.shape(vector_voxel)[1]
     zmax=np.shape(vector_voxel)[2]
     voxel_grid=workspace_voxel
     coordinate=None
     print(vector)
     if(dx==0):
         if(dy==0):
             if(dz>0):
                 while(z!=endposition_voxel[2]):
                      if vector_voxel[x,y,z]==True:
                           coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                           print('1',[x,y,z])
                           break
                      z=z+1
                      if z>=vector_voxel:
                          break
             if(dz<0):
                 z=-z
                 while(z!=-endposition_voxel[2]):
                      if vector_voxel[x,y,-z]==True:
                           z=-z
                           coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                           print('2',[x,y,z])
                           break
                      z=z+1
                      if z>=zmax:
                          break
                      
         if(dy>0):
             if(dz>0):
                 o=abs(((endposition_voxel[2])-start[2])/(endposition_voxel[1]-start[1]))    #o=dz/dy
                 if o<=1:
                     h=2*dz-dy
                     while(y!=endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                             if z>=500:
                                break
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,y,z]==True:
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('3',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break
                 if o>1:
                     templatey=y
                     y=z
                     z=templatey
                     h=2*dz-dy
                     while(y!=endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,z,y]==True:
                             templatey=y
                             y=z
                             z=templatey
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('4',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break
             if(dz==0):
                 while(y!=endposition_voxel[1]):
                      if vector_voxel[x,y,z]==True:
                           coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                           print('5',[x,y,z])
                           break
                      y=y+1
                      if y>=ymax:
                          break
             if(dz<0):
                 z=-z
                 dz=-dz
                 o=abs(((endposition_voxel[2])-start[2])/(endposition_voxel[1]-start[1]))    #o=dz/dy
                 if o<=1:
                     h=2*dz-dy
                     while(y!=endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,y,-z]==True:
                             z=-z
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('6',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break
                 if o>1:
                     templatey=y
                     y=z
                     z=templatey
                     h=2*dz-dy
                     while(y!=endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,z,-y]==True:   
                             templatey=y
                             y=z
                             z=templatey
                             z=-z  
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('57',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break
         if(dy<0):
             y=-y
             dy=-dy
             if(dz>0):
                 o=abs(((endposition_voxel[2])-start[2])/(endposition_voxel[1]-start[1]))    #o=dz/dy
                 if o<=1:
                     h=2*dz-dy
                     while(y!=-endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,-y,z]==True:
                             y=-y         
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('7',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break
                 if o>1:
                     templatey=y
                     y=z
                     z=templatey
                     h=2*dz-dy
                     while(y!=-endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,-z,y]==True:         
                             templatey=y
                             y=z
                             z=templatey
                             y=-y
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('58',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break
             if(dz<0):
                 z=-z
                 dz=-dz
                 o=abs(((endposition_voxel[2])-start[2])/(endposition_voxel[1]-start[1]))    #o=dz/dy
                 if o<=1:
                     h=2*dz-dy
                     while(y!=-endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,-y,-z]==True:
                             y=-y         
                             z=-z
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('8',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break                         
                         
                 if o>1:
                     templatey=y
                     y=z
                     z=templatey
                     h=2*dz-dy
                     while(y!=-endposition_voxel[1]):
                         if h>0:
                             h=h+2*dz-2*dy
                             z=z+1
                         elif h<=0:
                             h=h+2*dz
                             z=z
                         if vector_voxel[x,-z,-y]==True:
                             templatey=y
                             y=z
                             z=templatey
                             y=-y
                             z=-z
                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                             print('59',[x,y,z])
                             break
                         y=y+1
                         if y>=ymax:
                             break           
             if(dz==0):
                 while(y!=-endposition_voxel[1]):
                      if vector_voxel[x,-y,z]==True:
                           y=-y
                           coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                           print('9',[x,y,z])
                           break
                      y=y+1
                      if y>=ymax:
                           break                                 
     else:
         if(dy==0):
             n=abs((endposition_voxel[2]-start[2])/(endposition_voxel[0]-start[0]))
             if(dx>0):
                 
                 if(dz>0):
                      if n<=1:
                         h=2*dz-dx
                         while(x!=endposition_voxel[0]):
                             if h>0:
                                 h=h+2*dz-2*dx
                                 z=z+1
                             elif h<=0:
                                 h=h+2*dz
                                 z=z
                             if vector_voxel[x,y,z]==True:
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('61',[x,y,z])
                                 break
                             x=x+1    
                      if n>1:
                        templatex=x
                        x=z
                        z=templatex
                        h=2*dz-dx
                        while(x!=endposition_voxel[0]):
                            if h>0:
                                h=h+2*dz-2*dx
                                z=z+1
                            elif h<=0:
                                h=h+2*dz
                                z=z
                            if vector_voxel[z,y,x]==True:
                                templatex=x
                                x=z
                                z=templatex
                                coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                print('62',[x,y,z])
                                break
                            x=x+1
                 if(dz==0):
                      while(x!=endposition_voxel[0]):
                          if vector_voxel[x,y,z]==True:                  
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('60',[x,y,z])
                               break   
                          x=x+1
                 if(dz<0):
                      z=-z
                      dz=-dz
                      if n<=1:
                         h=2*dz-dx
                         while(x!=endposition_voxel[0]):
                             if h>0:
                                 h=h+2*dz-2*dx
                                 z=z+1
                             elif h<=0:
                                 h=h+2*dz
                                 z=z
                             if vector_voxel[x,y,-z]==True:
                                 z=-z
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('63',[x,y,z])
                                 break
                             x=x+1    
                      if n>1:
                            z=-z
                            templatex=x
                            x=z
                            z=templatex           
                            h=2*dz-dx
                            while(x!=endposition_voxel[0]):
                                if h>0:
                                    h=h+2*dz-2*dx
                                    z=z+1
                                elif h<=0:
                                    h=h+2*dz
                                    z=z
                                if vector_voxel[z,y,-x]==True:                             
                                    templatex=x
                                    x=z
                                    z=templatex
                                    z=-z
                                    coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                    print('64',[x,y,z])
                                    break
                                x=x+1

             if(dx<0):
                 x=-x
                 dx=-dx
                 if(dz>0):
                      if n<=1:
                         h=2*dz-dx
                         while(x!=-endposition_voxel[0]):
                             if h>0:
                                 h=h+2*dz-2*dx
                                 z=z+1
                             elif h<=0:
                                 h=h+2*dz
                                 z=z
                             if vector_voxel[-x,y,z]==True:
                                 x=-x
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('65',[x,y,z])
                                 break
                             x=x+1    
                      if n>1:
                        templatex=x
                        x=z
                        z=templatex
                        h=2*dz-dx
                        while(x!=-endposition_voxel[0]):
                            if h>0:
                                h=h+2*dz-2*dx
                                z=z+1
                            elif h<=0:
                                h=h+2*dz
                                z=z
                            if vector_voxel[-z,y,x]==True:
                                templatex=x
                                x=z
                                z=templatex
                                x=-x
                                coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                print('62',[x,y,z])
                                break
                            x=x+1                     
                 if(dz==0):
                     while(x!=-endposition_voxel[0]):
                          if vector_voxel[-x,y,z]==True:
                               x=-x
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('66',[x,y,z])
                               break
                          x=x+1    
                 if(dz<0):
                      z=-z
                      dz=-dz
                      if n<=1:
                         h=2*dz-dx
                         while(x!=-endposition_voxel[0]):
                             if h>0:
                                 h=h+2*dz-2*dx
                                 z=z+1
                             elif h<=0:
                                 h=h+2*dz
                                 z=z
                             if vector_voxel[-x,y,-z]==True:
                                 z=-z
                                 x=-x
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('67',[x,y,z])
                                 break
                             x=x+1    
                      if n>1:
                            templatex=x
                            x=z
                            z=templatex           
                            h=2*dz-dx
                            while(x!=-endposition_voxel[0]):
                                if h>0:
                                    h=h+2*dz-2*dx
                                    z=z+1
                                elif h<=0:
                                    h=h+2*dz
                                    z=z
                                if vector_voxel[-z,y,-x]==True:                             
                                    templatex=x
                                    x=z
                                    z=templatex
                                    z=-z
                                    x=-x
                                    coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                    print('68',[x,y,z])
                                    break
                                x=x+1                     
         m=abs((endposition_voxel[1]-start[1])/(endposition_voxel[0]-start[0]))
         n=abs((endposition_voxel[2]-start[2])/(endposition_voxel[0]-start[0]))
         o=abs((endposition_voxel[2]-start[2]))/(endposition_voxel[1]-start[1])

     

     #判断方向
     
         if m<=1:
            if n<=1:
               if dx>0:
                  if dy>0 and dz>0:
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[x,y,z]==True:
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('10',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break
                          
                          # list_test.append(coordinate)
                  if dy>0 and dz<0:
                      z=-z
                      dz=-dz
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[x,y,-z]==True:
                               z=-z
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('12',[x,y,z])
                               break   
                          x=x+1
                          if x>=xmax:
                              break
                  if dy<0 and dz<0:
                      z=-z
                      dz=-dz
                      y=-y
                      dy=-dy
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[x,-y,-z]==True:
                               z=-z
                               y=-y
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('13',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break
                  if dy<0 and dz>0:
                      y=-y
                      dy=-dy
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[x,-y,z]==True:
                               y=-y
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('14',[x,y,z])
                               break
                          x=x+1  
                          if x>=xmax:
                              break
               if dx<0:
                  x=-x     
                  dx=-dx
                  if dy>0 and dz>0:
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=-endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[-x,y,z]==True:
                               x=-x
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('15',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break
                  if dy>0 and dz<0:
                      z=-z
                      dz=-dz
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=-endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[-x,y,-z]==True:
                               x=-x
                               z=-z
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('16',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break
                  if dy<0 and dz<0:
                      z=-z
                      dz=-dz
                      y=-y
                      dy=-dy
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=-endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[-x,-y,-z]==True:
                               z=-z
                               y=-y
                               x=-x
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('17',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break
                  if dy<0 and dz>0:
                      y=-y
                      dy=-dy
                      p=2*dy-dx
                      q=2*dz-dx
                      while(x!=-endposition_voxel[0]):
                          if p>0:
                              p=p+2*dy-2*dx
                              y=y+1
                          elif p<=0:
                              p=p+2*dy
                              y=y
                          if q>0:
                              q=q+2*dz-2*dx
                              z=z+1
                          elif q<=0:
                              q=q+2*dz
                              z=z
                          
                          if vector_voxel[-x,-y,z]==True:
                               y=-y
                               x=-x
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('18',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break                     
            if n>1:
                
               if dx>0:
                  if dy>0 and dz>0:
                      dx1=dx
                      dx2=dx
                      template2=dz
                      dz=dx2
                      dx2=template2
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                            
                          if vector_voxel[z,y,x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('19',[x,y,z])
                               break
                          x=x+1
                          if x>=xmax:
                              break
                  if dy>0 and dz<0:  
                      dx1=dx
                      dx2=dx
                      z=-z   #换符号
                      dz=-dz
                      
                      template2=dz
                      dz=dx2
                      dx2=template2  
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=-endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                           
                          if vector_voxel[z,y,-x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               z=-z
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('20',[x,y,z])
                               break
                          x=x+1 
                          if x>=xmax:
                              break
                  if dy<0 and dz<0:  
                      dx1=dx
                      dx2=dx
                      z=-z   #换符号
                      dz=-dz
                      y=-y
                      dy=-dy  
                      template2=dz
                      dz=dx2
                      dx2=template2  
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=-endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                        
                          if vector_voxel[z,-y,-x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               z=-z
                               y=-y
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('21',[x,y,z])
                               
                          x=x+1
                          if x>=xmax:
                              break
                  if dy<0 and dz>0:  
                      dx1=dx
                      dx2=dx
                      #换符号
                      y=-y
                      dy=-dy  
                      template2=dz
                      dz=dx2
                      dx2=template2  
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                          
                          if vector_voxel[z,-y,x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               y=-y
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('22',[x,y,z])
                               break
                          x=x+1 
                          if x>=xmax:
                              break
                           
               if dx<0:
                  x=-x
                  dx=-dx
                  if dy>0 and dz>0:
                      dx1=dx
                      dx2=dx
                      template2=dz
                      dz=dx2
                      dx2=template2
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                           
                          if vector_voxel[-z,y,x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               x=-x
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('23',[x,y,z])
                               break
                          x=x+1 
                          if x>=xmax:
                              break
                  if dy>0 and dz<0:  
                      dx1=dx
                      dx2=dx
                      z=-z   #换符号
                      dz=-dz
                      
                      template2=dz
                      dz=dx2
                      dx2=template2  
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=-endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                          
                          if vector_voxel[-z,y,-x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               x=-x
                               z=-z
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('24',[x,y,z])
                               break
                          x=x+1  
                          if x>=xmax:
                              break
                  if dy<0 and dz<0:  
                      dx1=dx
                      dx2=dx
                      z=-z   #换符号
                      dz=-dz
                      y=-y
                      dy=-dy  
                      template2=dz
                      dz=dx2
                      dx2=template2  
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=-endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z                     
                          if vector_voxel[-z,-y,-x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               z=-z
                               y=-y
                               x=-x
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('25',[x,y,z])
                               break
                          x=x+1  
                          if x>=xmax:
                              break
                  if dy<0 and dz>0:  
                      dx1=dx
                      dx2=dx
                      #换符号
                      y=-y
                      dy=-dy  
                      template2=dz
                      dz=dx2
                      dx2=template2  
                      templatez=z
                      z=x
                      x=templatez
                      
                      p=2*dy-dx1
                      q=2*dz-dx2
                      while(x!=endposition_voxel[2]):
                          if q>0:
                               q=q+2*dz-2*dx2
                               if p>0:
                                   p=p+2*dy-2*dx1
                                   y=y+1
                               elif p<=0:
                                   p=p+2*dy
                                   y=y
                               z=z+1
                          elif q<=0:
                               q=q+2*dz
                               z=z
                   
                          if vector_voxel[-z,-y,x]==True:
                               templatez=z
                               z=x
                               x=templatez
                               x=-x
                               y=-y
                               coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                               print('26',[x,y,z])
                               break
                          if x>=xmax:
                              break
         
                             
         if m>1:
             
            if n<=1:
                if dx>0:
                    if dy>0 and dz>0:    # m<1 n>1 dx>0 dy>0 dz>0   x按照z变化，所以x与z交换。然后y原本按照x变化，现在按照z。
                        dx1=dx
                        dx2=dx
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        
                        p=2*dy-dx1
                        q=2*dz-dx2
                        while(x!=endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y
                                 
                            
                            if vector_voxel[y,x,z]==True:
                                 templatey=y
                                 y=x
                                 x=templatey
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('27',[x,y,z])
                                 break
                            x=x+1 
                            if x>=xmax:
                                break
                    if dy>0 and dz<0:   
                        dx1=dx
                        dx2=dx
                        z=-z
                        dz=-dz
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        p=2*dy-dx1
                        q=2*dz-dx2
                        while(x!=endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y
                                 
                            
                            if vector_voxel[y,x,-z]==True:
                                 templatey=y
                                 y=x
                                 x=templatey
                                 z=-z
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('28',[x,y,z])
                                 break  
                            x=x+1 
                            if x>=xmax:
                                break
                    if dy<0 and dz<0:
                        dx1=dx
                        dx2=dx
                        z=-z
                        dz=-dz
                        y=-y
                        dy=-dy
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        p=2*dy-dx
                        q=2*dz-dx
                        while(x!=-endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y
                            
                            if vector_voxel[y,-x,-z]==True:     # 交换符号在原来的位置
                                 templatey=y
                                 y=x
                                 x=templatey
                                 z=-z
                                 y=-y
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('29',[x,y,z])
                                 break  
                            x=x+1
                            if x>=xmax:
                                break
                    if dy<0 and dz>0:
                        dx1=dx
                        dx2=dx
                        y=-y
                        dy=-dy
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        p=2*dy-dx
                        q=2*dz-dx
                        while(x!=-endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y
                            
                            if vector_voxel[y,-x,z]==True:
                                 templatey=y
                                 y=x
                                 x=templatey
                                 y=-y
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 print('30',[x,y,z])
                                 break
                            x=x+1
                            if x>=xmax:
                                break
                   
                if dx<0:
                    x=-x
                    dx=-dx
                    if dy>0 and dz>0:
                        dx1=dx
                        dx2=dx
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        
                        p=2*dy-dx1
                        q=2*dz-dx2
                        while(x!=endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y                         
                            
                            if vector_voxel[-y,x,z]==True:     
                                 templatey=y
                                 y=x
                                 x=templatey
                                 x=-x
                                 print('31',[x,y,z])
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                
                                 break
                            x=x+1  
                            if x>=xmax:
                                break
                    if dy>0 and dz<0:
                        dx1=dx
                        dx2=dx
                        z=-z
                        dz=-dz
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        p=2*dy-dx1
                        q=2*dz-dx2
                        while(x!=endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y
                            if vector_voxel[-y,x,-z]==True:
                                 templatey=y
                                 y=x
                                 x=templatey
                                 z=-z
                                 x=-x
                                 print('32',[x,y,z])
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                 break    
                                 
                            x=x+1  
                            if x>=xmax:
                                break
                    if dy<0 and dz<0:     #dx<0 -x dy<0 -y dz<0 -z
                        dx1=dx
                        dx2=dx
                        z=-z
                        dz=-dz
                        y=-y
                        dy=-dy
                        template1=dy
                        dy=dx1
                        dx1=template1
                        templatey=y
                        y=x
                        x=templatey
                        p=2*dy-dx
                        q=2*dz-dx
                        while(x!=-endposition_voxel[1]):
                            if p>0:
                                 p=p+2*dy-2*dx1
                                 if q>0:
                                    q=q+2*dz-2*dx2
                                    z=z+1
                                 elif q<=0:
                                    q=q+2*dz
                                    z=z                          
                                 y=y+1
                            elif p<=0:
                                 p=p+2*dy
                                 y=y
                            
                            if vector_voxel[-y,-x,-z]==True:
                                 templatey=y
                                 y=x
                                 x=templatey
                                 x=-x
                                 z=-z
                                 y=-y
                                 print('33',[x,y,z])
                                 coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                
                                 break          
                            x=x+1       
                            if x>=xmax:
                                break
                    if dy<0 and dz>0:
                          dx1=dx
                          dx2=dx
                          y=-y
                          dy=-dy
                          template1=dy
                          dy=dx1
                          dx1=template1
                          templatey=y
                          y=x
                          x=templatey
                          p=2*dy-dx
                          q=2*dz-dx
                          while(x!=-endposition_voxel[1]):
                              if p>0:
                                   p=p+2*dy-2*dx1
                                   if q>0:
                                      q=q+2*dz-2*dx2
                                      z=z+1
                                   elif q<=0:
                                      q=q+2*dz
                                      z=z                          
                                   y=y+1
                              elif p<=0:
                                   p=p+2*dy
                                   y=y
                              
                              if vector_voxel[-y,-x,z]==True:
                                   templatey=y
                                   y=x
                                   x=templatey
                                   x=-x
                                   y=-y
                                   print('34',[x,y,z])
                                   coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                  
                                   break
                              x=x+1
                              if x>=xmax:
                                  break
          
            else: 
                if o<=1:    #此时不考虑 z对于x的影响 只考虑 y对x的以及y对z的
                     if dx>0:
                             if dy>0 and dz>0:    
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[y,x,z]==True:
                                             templatey=y
                                             y=x
                                             x=templatey
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('37',[x,y,z])
                                             break
                                        x=x+1   
                                        if x>=xmax:
                                            break
                             if dy>0 and dz<0:    
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 z=-z
                                 dz=-dz
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[y,x,-z]==True:
                                             z=-z
                                             dz=-dz
                                             templatey=y
                                             y=x
                                             x=templatey
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('38',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                             if dy<0 and dz<0:   
                                 y=-y
                                 dy=-dy
                                 z=-z
                                 dz=-dz
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=-endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[y,-x,-z]==True:
                                             templatey=y
                                             y=x
                                             x=templatey
                                             y=-y
                                             z=-z
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('39',[x,y,z])
                                             break
                                        x=x+1   
                                        if x>=xmax:
                                            break
                             if dy<0 and dz>0:   
                                 y=-y
                                 dy=-dy
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=-endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[y,-x,z]==True:
                                             templatey=y
                                             y=x
                                             x=templatey
                                             y=-y
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('40',[x,y,z])
                                             break
                                        x=x+1      
                                        if x>=xmax:
                                            break
                     if dx<0:
                             x=-x
                             dx=-dx
                             if dy>0 and dz>0:    
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[-y,x,z]==True:
                                             templatey=y
                                             y=x
                                             x=templatey
                                             x=-x
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('41',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                             if dy>0 and dz<0:    
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 z=-z
                                 dz=-dz
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[-y,x,-z]==True:
                                             z=-z
                                             templatey=y
                                             y=x
                                             x=templatey
                                             x=-x
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('42',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                             if dy<0 and dz<0:    
                                 y=-y
                                 dy=-dy
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 z=-z
                                 dz=-dz
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=-endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[-y,-x,-z]==True:
                                             z=-z
                                             dz=-dz
                                             templatey=y
                                             y=x
                                             x=templatey
                                             x=-x
                                             y=-y
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('43',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                             if dy<0 and dz>0:   
                                 y=-y
                                 dy=-dy
                                 template1=dy
                                 dy=dx
                                 dx=template1
                                 templatey=y
                                 y=x
                                 x=templatey  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=-endposition_voxel[1]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[-y,-x,z]==True: 
                                             templatey=y
                                             y=x
                                             x=templatey
                                             y=-y
                                             x=-x
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('44',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                if o>1:  #此时不考虑 y对于x的影响 只考虑 z对于x以及z对y的
                     if dx>0:
                             if dy>0 and dz>0:    
                                 template2=dz
                                 dz=dx
                                 dx=template2
                                 templatez=z
                                 z=x
                                 x=templatez  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=endposition_voxel[2]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[z,y,x]==True:
                                             templatez=z
                                             z=x
                                             x=templatez
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('45',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                             if dy>0 and dz<0:    
                                 z=-z
                                 template2=dz
                                 dz=dx
                                 dx=template2
                                 templatez=z
                                 z=x
                                 x=templatez                            
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=-endposition_voxel[2]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[z,y,-x]==True:
                                             templatez=z
                                             z=x
                                             x=templatez
                                             z=-z
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('46',[x,y,z])
                                             break
                                        x=x+1  
                                        if x>=xmax:
                                            break
                             if dy<0 and dz<0:  
                                 z=-z
                                 y=-y
                                 template2=dz
                                 dz=dx
                                 dx=template2
                                 templatez=z
                                 z=x
                                 x=templatez  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=-endposition_voxel[2]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[z,-y,-x]==True:
                                             templatez=z
                                             z=x
                                             x=templatez
                                             y=-y
                                             z=-z
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             print('47',[x,y,z])
                                             break
                                        x=x+1       
                                        if x>=xmax:
                                            break
                             if dy<0 and dz>0:  
                                 y=-y
                                 template2=dz
                                 dz=dx
                                 dx=template2
                                 templatez=z
                                 z=x
                                 x=templatez  
                                 p=2*dy-dx
                                 q=2*dz-dx
                                 while(x!=endposition_voxel[2]):
                                        if p>0:
                                             p=p+2*dy-2*dx                 
                                             y=y+1
                                        elif p<=0:
                                             p=p+2*dy
                                             y=y
                                        if q>0:
                                           q=q+2*dz-2*dx
                                           z=z+1
                                        elif q<=0:
                                           q=q+2*dz
                                           z=z     
                                        if vector_voxel[z,-y,x]==True:
                                             templatez=z
                                             z=x
                                             x=templatez
                                             y=-y
                                             coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                             return coordinate
                                             print('48',[x,y,z])
                                             break
                                        x=x+1    
                                        if x>=xmax:
                                            break
                     if dx<0:
                        x=-x
                        dx=-dx
                        if dy>0 and dz>0:    
                            template2=dz
                            dz=dx
                            dx=template2
                            templatez=z
                            z=x
                            x=templatez  
                            p=2*dy-dx
                            q=2*dz-dx
                            while(x!=endposition_voxel[2]):
                                   if p>0:
                                        p=p+2*dy-2*dx                 
                                        y=y+1
                                   elif p<=0:
                                        p=p+2*dy
                                        y=y
                                   if q>0:
                                      q=q+2*dz-2*dx
                                      z=z+1
                                   elif q<=0:
                                      q=q+2*dz
                                      z=z     
                                   if vector_voxel[-z,y,x]==True:
                                        templatez=z
                                        z=x
                                        x=templatez
                                        x=-x
                                        dx=-dx
                                        coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                        
                                        print('49',[x,y,z])
                                        break
                                   x=x+1  
                                   if x>=xmax:
                                       break
                        if dy>0 and dz<0:    
                            z=-z
                            template2=dz
                            dz=dx
                            dx=template2
                            templatez=z
                            z=x
                            x=templatez                            
                            p=2*dy-dx
                            q=2*dz-dx
                            while(x!=-endposition_voxel[2]):
                                   if p>0:
                                        p=p+2*dy-2*dx                 
                                        y=y+1
                                   elif p<=0:
                                        p=p+2*dy
                                        y=y
                                   if q>0:
                                      q=q+2*dz-2*dx
                                      z=z+1
                                   elif q<=0:
                                      q=q+2*dz
                                      z=z     
                                   if vector_voxel[-z,y,-x]==True:
                                        templatez=z
                                        z=x
                                        x=templatez
                                        z=-z
                                        x=-x
                                        dx=-dx
                                        coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                        
                                        print('50',[x,y,z])
                                        break
                                   x=x+1  
                                   if x>=xmax:
                                       break
                        if dy<0 and dz<0:  
                            z=-z
                            y=-y
                            template2=dz
                            dz=dx
                            dx=template2
                            templatez=z
                            z=x
                            x=templatez  
                            p=2*dy-dx
                            q=2*dz-dx
                            while(x!=-endposition_voxel[2]):
                                   if p>0:
                                        p=p+2*dy-2*dx                 
                                        y=y+1
                                   elif p<=0:
                                        p=p+2*dy
                                        y=y
                                   if q>0:
                                      q=q+2*dz-2*dx
                                      z=z+1
                                   elif q<=0:
                                      q=q+2*dz
                                      z=z     
                                   if vector_voxel[-z,-y,-x]==True:
                                        templatez=z
                                        z=x
                                        x=templatez
                                        y=-y
                                        z=-z
                                        x=-x
                                        coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                    
                                        print('51',[x,y,z])
                                        break
                                   x=x+1  
                                   if x>=xmax:
                                       break
                        if dy<0 and dz>0:  
                            y=-y
                            template2=dz
                            dz=dx
                            dx=template2
                            templatez=z
                            z=x
                            x=templatez  
                            p=2*dy-dx
                            q=2*dz-dx
                            while(x!=endposition_voxel[2]):
                                   if p>0:
                                        p=p+2*dy-2*dx                 
                                        y=y+1
                                   elif p<=0:
                                        p=p+2*dy
                                        y=y
                                   if q>0:
                                      q=q+2*dz-2*dx
                                      z=z+1
                                   elif q<=0:
                                      q=q+2*dz
                                      z=z     
                                   if vector_voxel[-z,-y,x]==True:
                                        templatez=z
                                        z=x
                                        x=templatez
                                        y=-y
                                        x=-x
                                        coordinate=get_geo_coordinate(voxel_grid,np.array([x,y,z]),shift_vector)
                                        print('52',[x,y,z])
                                     
                                        break
                                   x=x+1     
                                   if x>=xmax:
                                       break
        
     
        
     return coordinate


def function_of_rays(sensor_position,unit_vector,distance):    #from position of sensor and direction of rays get the endpoint
    position_endpoint=sensor_position+distance*unit_vector
    return position_endpoint
def get_index(endposition_points,origin_endpoints):
    vector=endposition_points-origin_endpoints
    index=np.floor(vector/voxel_size)
    return index


    
#sensor_total          world coordinate


if __name__=="__main__":
   Invest=["classified","0.05"]
   if Invest[0]=="LOD2":
        map_="002"
   if Invest[0]=="LOD3":
        map_="003"
   if Invest[0]=="LOD4":
        map_="004"
   if Invest[0]=="classified":
        map_="173_174_184_187"
   pcd=o3d.io.read_point_cloud('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\maps\\'+map_+'.ply')
   voxel_size=float(Invest[1])
   savepath_simulation="E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\bresenham-"+Invest[1]+"-"+Invest[0]+"\\" 
   if os.path.isdir(savepath_simulation)==False:
       os.mkdir(savepath_simulation)
   count=10041
   

   reference_trajectory= np.loadtxt('E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\extractet_traj_points.txt', delimiter=",")
   sensor_position_array=reference_trajectory[:,2:5]
   unit_vector=Sensor_total.world_coordinate      #[1915,28800,3]    
   for position in range(count-8921,len(sensor_position_array),1):
                 count_rays=0
                 rays=0
                 candidate_list=[]
                 list=[]
                 start_coor=sensor_position_array[position]                 
                 workspace=Workspace(Invest[0],position,Invest[1])
                 workspace_voxel=workspace.voxelization(pcd,start_coor)   #voxelization return voxel_grid
                 start=workspace_voxel.get_voxel(start_coor)    # index of start point
                 origin_workspace=workspace_voxel.origin   #原点utm
                 endposition_points=function_of_rays(start_coor,unit_vector[position],distance=100)   #utm
                 endposition_voxel_index=get_index(endposition_points,origin_workspace)    #rays 的index
                 #光线的坐标一定包含着扫描物体
                 shift_vector=np.asarray([0,0,0])#np.asarray([min(endposition_voxel_index[:,0]),min(endposition_voxel_index[:,1]),min(endposition_voxel_index[:,2])])
                 #把光线和物体合在同一个坐标系内
                 start=(start-shift_vector)     
                 vector_voxel=vector_3d(workspace_voxel,voxel_size,shift_vector)
                 for endposition_index in endposition_voxel_index:
                     coordinate=bresenham(start, endposition_index,workspace_voxel,vector_voxel)    #主函数里面的endposition_voxel_index 包含了所有rays的endposition   workspace_voxel 是voxel_grid
                     if coordinate is None:
                        rays+=1
                        print(rays,'fail')
                        continue
                
                     candidate_list.append(coordinate)
                     rays+=1
                     print(rays,'success')
               
                  # if count==10000:
                 #     break
                 # 
                 for j in range(len(candidate_list)):
                    candidate_point=candidate_list[j].tolist()
                    number=0
                    for k in candidate_list[j+1:]:        
                        distance=np.linalg.norm(candidate_list[j]-k)
                        if distance<5:
                           number+=1
                        if number>=5:
                            list.append(candidate_list[j])
                            print(count_rays)
                            count_rays+=1
                            break
                        
   

               # count2+=1
               # print(count2,'success')
                      
           # file_write_obj = open("E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\intersectedtest.txt", 'w')
           # for point in list:
           #       count=0
           #       for coordinate in point:
           #           count+=1
           #           coordinate=str(coordinate)
           #           file_write_obj.writelines(coordinate)
           #           if count%3!=0:
           #               file_write_obj.writelines(',')
           #           else:
           #               continue
           #       file_write_obj.write('\n')
           # file_write_obj.close()
                 np.savetxt(savepath_simulation+str(count).zfill(5)+"_simulation.txt",np.asarray(list),fmt='%.14f',delimiter=",")
                 print(count,'success')
                 count+=1
         
            
        
                                    

        
    

                                
                
                            
                                 
                        
                        

                      

    
            
            
              
             
              
                      

                 
                 
               
    
    
#     x=-start[0]
#     y=start[1]
#     z=start[2]

# dx<0 dy>0 dz>0 0<m<1

#     if np.sign(dx)<0 and np.sign(dy)>0 and abs(m)<1:
#         p=2*dy+dx
#         q=2*dz+dx
#         while(x!=-endposition_voxel[0]):
#             if p>0:
#                 p=p+2*dy+2*dx
#                 y=y+1
#             elif p<=0:
#                 p=p+2*dy
#             if q>0:
#                 q=q+2*dz+2*dx
#                 z=z+1
#             elif q<=0:
#                 q=q+2*dz
#             x=x+1
            
            

            
       