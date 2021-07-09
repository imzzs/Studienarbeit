# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:59:14 2021

@author: 123
"""



import open3d as o3d
import numpy as np
import math
import Voxelization
import Sensor

def angle_lines(vector1,vector2):   
    dot=vector1.dot(vector2)
    norm_1=np.sqrt(vector1.dot(vector1))
    norm_2=np.sqrt(vector2.dot(vector2))
    cos=dot/(norm_1*norm_2)
    radian=np.arccos(cos)
    return radian #radian

def distance_point_ray(point,position,unit_vector):      #compute the distance between the origin of a voxel and a ray
    vector=point-position                                #position of sensor       
    radian=angle_lines(vector,unit_vector)
    area=np.sqrt(vector.dot(vector))*np.sqrt(unit_vector.dot(unit_vector))*math.sin(radian)*np.sign(np.cos(radian))    #利用余弦符号去掉 反方向的点
    distance=area/np.linalg.norm(unit_vector) 
    return distance

def distance_point_point(point1,point2):   #the distance between 2 points
    distance=np.sqrt(np.sum((point1-point2)**2))
    return distance

def distance_point_plane(point,plane):
    distance=np.absolute(plane[0]*point[0]+plane[1]*point[1]+plane[2]*point[2]+plane[3])/np.sqrt(np.sum(point[:3]**2))
    return distance
 
def t_distance(sensor_position,plane,unit_vector): # the distance along the ray from sensor to the plane
    if plane[0]!=0:
        t=(-plane[3]-sensor_position[0])/unit_vector[0]   #in the function intersection, unit_vector[0]=0 is excluded
    elif plane[1]!=0:
        t=(-plane[3]-sensor_position[1])/unit_vector[1]
    elif plane[2]!=0:
        t=(-plane[3]-sensor_position[2])/unit_vector[2]
    return t

def Bubble_sort(list):
    for j in range(len(list)-1):                # sort the candidate with the distance to sensor position
        for k in range(len(list)-j-1):
            if list[k][1]>list[k+1][1]:
                template=list[k]
                list[k]=list[k+1]
                list[k+1]=template
    return list



def intersection(voxel_position,sensor_position,unit_vector,voxel_size=0.5):  #slab algorithm to check if intersected
    plane1=np.array([1,0,0,-voxel_position[0]])
    plane2=np.array([0,1,0,-voxel_position[1]])
    plane3=np.array([0,0,1,-voxel_position[2]])
    plane4=np.array([1,0,0,-voxel_position[0]-voxel_size])
    plane5=np.array([0,1,0,-voxel_position[1]-voxel_size])
    plane6=np.array([0,0,1,-voxel_position[2]-voxel_size])

    tmin=float('-inf')
    tmax=float('inf')
    if unit_vector[0]==0:      #if ray parallel to plane yoz
       if sensor_position[0]<(voxel_position[0]) or sensor_position[0]>(voxel_position[0]+voxel_size):
           return False

    else:
        t1=t_distance(sensor_position,plane1,unit_vector)
        t4=t_distance(sensor_position,plane4,unit_vector)
        tmin1=min(t1,t4)
        tmin=max(tmin1,tmin)
        tmax1=max(t1,t4)
        tmax=min(tmax,tmax1)
    
    if unit_vector[1]==0:      #if ray parallel to plane xoz
       if sensor_position[1]<(voxel_position[1]) or sensor_position[1]>(voxel_position[1]+voxel_size):
           return False

    else:
        t2=t_distance(sensor_position,plane2,unit_vector)
        t5=t_distance(sensor_position,plane5,unit_vector)
        tmin2=min(t2,t5)
        tmin=max(tmin2,tmin)
        tmax2=max(t2,t5)
        tmax=min(tmax,tmax2)
 
    if unit_vector[2]==0:      #if ray parallel to plane xoz
       if sensor_position[2]<(voxel_position[2]) or sensor_position[2]>(voxel_position[2]+voxel_size):
           return False

    else:
        t3=t_distance(sensor_position,plane3,unit_vector)
        t6=t_distance(sensor_position,plane6,unit_vector)
        tmin3=min(t3,t6)
        tmin=max(tmin3,tmin)
        tmax3=max(t3,t6)
        tmax=min(tmax,tmax3)
    if tmin>tmax:
        return False
    else: 
        return True

          
if __name__=="__main__":
    intersected_voxel=[]
    voxel_size=0.5
    a=0   
    point_list = np.asarray(Voxelization.coor_list)
    sensor_position = Sensor.sensor_position_list[1500]

    for ray in Sensor.array_unit_vector:          
            vector=point_list-sensor_position                                      
            dot=vector.dot(ray)
            norm_1=np.linalg.norm(vector,axis=1)
            norm_2=np.linalg.norm(ray)
            cos=dot/(norm_1*norm_2)
            radian=np.arccos(cos)
            area=np.linalg.norm(vector,axis=1)*np.linalg.norm(ray)*np.sin(radian)*np.sign(np.cos(radian))    #利用余弦符号去掉 反方向的点
            distance=area/np.linalg.norm(ray) 
            
            distancepoint= np.linalg.norm(point_list-Sensor.sensor_position_list[1500],axis=1)
            idx=np.where(((distance<math.sqrt(3*voxel_size**2)) & (distance>0)))[0]
            candidate_voxel=np.concatenate((point_list[idx],distancepoint[idx][:,np.newaxis]),axis=1)
            candidate_voxel=candidate_voxel[candidate_voxel[:,3].argsort()]   

    #Slab 算  
            for voxel in candidate_voxel:
                  if intersection(voxel,Sensor.sensor_position_list[1500],ray):
                    intersected_voxel.append(voxel)
                    a+=1
                    print(a,' ','success')
                    break

            
    np.savetxt("E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\intersected",np.asarray(intersected_voxel),delimiter=",")

  
    # ray_point=[]
    # for t in np.arange(0,50,0.05):
    #     point=Sensor.function_of_rays(Sensor.sensor_position_list[1500],ray,t)
    #     ray_point.append(point)

    # file_write_obj = open("E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\rays.txt", 'w')
    # for point in ray_point:
    #     count=0
    #     for coordinate in point:
    #         count+=1
    #         coordinate=str(coordinate)
    #         file_write_obj.writelines(coordinate)
    #         if count%3!=0:
    #             file_write_obj.writelines(',')
    #         else:
    #             continue
  
    #     file_write_obj.write('\n')
    # file_write_obj.close()
            
    # file_write_obj = open("/home/axmann/Dokumente/Masterthesis/Zisen/test_intersect.txt", 'w')
    # for point in intersected_voxel:
    #     count=0
    #     for coordinate in point:
    #         count+=1
    #         coordinate=str(coordinate)
    #         file_write_obj.writelines(coordinate)
    #         if count%4!=0:
    #             file_write_obj.writelines(',')
    #         else:
    #             continue
    #     file_write_obj.write('\n')
    # file_write_obj.close()
        
    # file_write_obj = open("E:\\5\\documents-export-2021-04-22\\Zisen_Zhao\\Zisen_Zhao\\candidate_voxel.txt", 'w')
    # for point in candidate_voxel:
    #     count=0
    #     for coordinate in point[0]:
    #         count+=1
    #         coordinate=str(coordinate)
    #         file_write_obj.writelines(coordinate)
    #         if count%3!=0:
    #             file_write_obj.writelines(',')
    #         else:
    #             continue
    #     file_write_obj.write('\n')
    # file_write_obj.close()                     
                    
            
            
            
            
            
    
    
   

        
    

    
    
    
    
    