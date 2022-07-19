import numpy as np
import math
from noise_functions import julia, my_perl
import random

class mountain():
    
    def __init__(self, x, y, size=1.0,erosion = -1):
        self.position = np.asarray([x,y])
        self.size=size
        self.num_lines = int(math.sqrt(size)) * 2.
        if erosion < 0 or erosion > 1:
            self.erosion = random.random()
        else:
            self.erosion = erosion
    def get_base_height(self,x,y):
        abs_coords = [x-self.position[0], y-self.position[1]]
        dist = math.sqrt(abs_coords[0]**2 + abs_coords[1]**2)/self.size
        if dist <= 1: #Dist is relative to size, so dist=1 is always the edge
            height_inv = 1/(1+dist)
            height_flat = 1 - dist
            
            angl = np.arctan2(abs_coords[0],abs_coords[1])
            height_line = np.sin(angl*self.num_lines)**2 *dist
            branch_id = int(angl*self.num_lines*4)+int(self.position[0]*10+self.position[1]*3.14)
            
            
            #Version using sin of dist to find branches on main lines
            #height_branch = np.sin(dist*10*(2+abs(height_line)))*height_line*(dist-height_line)
            
            #Version using julia fractal to generate branches
            height_branch = (1-dist) * (julia(1-height_line, np.sin((dist+height_line)*5*self.size),c = 1j, n = 10,cap = 2) *0.5+0.5)*height_line
            
            height = self.size * ((height_inv+height_line+height_branch) * height_flat)#normalise?
        else:
            height = height_branch=0
        return(height)
    
    def get_height(self,x,y):
        base_height = self.get_base_height(x,y)
        if base_height > 0:
            abs_coords = [x-self.position[0], y-self.position[1]]
            dist = math.sqrt(abs_coords[0]**2 + abs_coords[1]**2)/self.size
            offset = my_perl.sample(x,y,octaves=5,neg_octaves = -1, fade=0.5)#*(1+dist)
            extra_height = (self.get_base_height(offset[0]*self.size + x,offset[1]*self.size + y)*self.erosion*abs(offset[2])/self.size + 1-self.erosion)**2
        else:
            extra_height = 1
        return(base_height*extra_height*self.erosion + base_height * (1-self.erosion))
    
class voron_mountain():
    
    def __init__(self, x, y, size=1.0,erosion = -1):
        self.position = np.asarray([x,y])
        self.size=size
        self.num_lines = int(math.sqrt(size)) * 2.
        if erosion < 0 or erosion > 1:
            self.erosion = random.random()
        else:
            self.erosion = erosion
        self.centroids = []
        self.heights =[]
        for i in range (int(self.num_lines)): #Depends on size?
            angl = (i + random.random()) * np.pi*2 / self.num_lines
            rad = random.gauss(size,size/4)
            c, s = rad*np.cos(angl), rad*np.sin(angl)
            self.centroids.append(np.asarray([c,s])) 
            self.heights.append(random.random()) #Unused?
            
        
            
    def get_base_height(self,x,y):
        abs_coords = [x-self.position[0], y-self.position[1]]
        dist = math.sqrt(abs_coords[0]**2 + abs_coords[1]**2)/self.size
        if dist <= 1: #Dist is relative to size, so dist=1 is always the edge
            height_inv = 1/(1+dist)
            height_flat = 1 - dist
            if self.num_lines > 0:
                offsets = self.centroids - np.asarray(abs_coords)
                distances = np.linalg.norm(offsets, axis = -1)/self.size #normalised distance for each centroid
                centr_dist = min(distances) #Distance to closest centroid
                centr_num = np.argmin(distances) #ID of closest centroid
            
            
                height =  self.size * ((height_inv*(centr_dist)) * height_flat)#normalise?
            else:
                height = self.size * (height_inv * height_flat)
        else:
            height = centr_dist = 0
        return(height)
    
    def get_height(self,x,y):
        offset = my_perl.sample(x,y,octaves=2,neg_octaves = 2, fade=0.75)*0.1*self.erosion
        base_height = self.get_base_height(offset[0]*self.size + x,offset[1]*self.size + y)
        if False and base_height > 0:
            abs_coords = [x-self.position[0], y-self.position[1]]
            dist = math.sqrt(abs_coords[0]**2 + abs_coords[1]**2)/self.size
            offset = my_perl.sample(x,y,octaves=5,neg_octaves = -2, fade=0.5)#*(1+dist)
            extra_height = (self.get_base_height(offset[0]*self.size + x,offset[1]*self.size + y)*self.erosion*abs(offset[2])/self.size + 1-self.erosion)**2
        else:
            extra_height = 1
        return(base_height)#*extra_height*self.erosion + base_height * (1-self.erosion))