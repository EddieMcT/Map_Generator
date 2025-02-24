import numpy as np
from noise_functions import my_perl
import random
import math
import gc


class landscape_gen():
    def __init__(self,lat = 10, long = 10, num_plates = 5, boundaries = True):
        self.lat = lat 
        self.long = long 
        self.lin_sca = math.sqrt(lat*long)
        self.mountains_done = False 
        self.rivers_done = False 
        self.centroids = []
        self.heights =[]
        if boundaries:
            self.centroids = [np.asarray([0,lat]), np.asarray([long,0]),
                              np.asarray([long, lat]), np.asarray([0,0]),
                              np.asarray([0,lat*0.5]), np.asarray([long*0.5,0]),
                              np.asarray([long*0.5, lat]), np.asarray([long, lat*0.5])] #Seafloor tiles on each corner and middles of edges. Redundant if world is made round
            self.heights = [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]
        self.mountains = []
        self.mountains_tiles_x = []
        self.mountains_tiles_y = []
        for i in range (num_plates): 
            new_xy = np.asarray([random.random()*long, random.random()*lat])
            self.centroids.append(new_xy)
            max_height = np.divide(new_xy, [long, lat])
            max_height = np.multiply(max_height, 1 - max_height)
            max_height = np.sum(max_height)
            self.heights.append(max(random.random(),random.random())*4*max_height-1) #To do: scale heights so they're higher in the middle and lower near edges
        self.centroids = np.asarray(self.centroids)        
  
    def compute_offsets(self, x, y,offs = 1, fine_offs =1, **kwargs): #consider using a different function for fine_offset
        offset = my_perl.sample(x,y,neg_octaves = 4, octaves=-1,ndims=2)  ##REFERENCES NOISE, UPDATE AS NECESSARY. Current method with -4 gives range of +-20km (real world ~= 250), each neg_octave doubles this
        fine_offset = my_perl.sample(x,y,neg_octaves = 1, octaves = 4,ndims=3) #Use ndims=3 for hill noise?
        offset =  np.add(fine_offset[:,:,0:2],offset) * offs
        fine_offset = np.add(fine_offset,my_perl.sample(x,y,neg_octaves = -4, octaves = 8,ndims=3)) * fine_offs
        fine_x = np.add(x, fine_offset[:,:,0])
        fine_y = np.add(y, fine_offset[:,:,1])
        #fine_offset = fine_offset[:,:,2] use if a third dimension is needed
        coarse_x = x+offset[:,:,0]
        coarse_y = y+offset[:,:,1]
        return(coarse_x, coarse_y, fine_x, fine_y)
    def get_rivers(self,x,y,z,weight=1, **kwargs):
        river_z = np.zeros_like(x) #placeholder
        #river_Z = my_perl.dendry(self, x, y,base_scale = 1, octaves = 5, subsampling = 10, control_function = self.get_base_height, **kwargs)
        return(river_z)
    def get_base_height(self, x,y, **kwargs):
        x = np.array(x)
        y = np.array(y)
        #noise_height = offset[2] #Necessary? May cause tiling depending on noise
        distances = np.sqrt((x[:, :, np.newaxis] - self.centroids[:, 0])**2 + (y[:, :, np.newaxis] - self.centroids[:, 1])**2)

        #plate_num = np.argmin(distances, axis = -1)
        plate_dist = np.sort(distances, axis = -1)[:,:,0]#Distance to closest plate
        
        distances = np.subtract(distances, plate_dist[:,:,np.newaxis]) #How much further a plate is than the closest plate
        
        distances = np.minimum(distances, self.lin_sca/10) * 10/self.lin_sca #Cap the distance at 10%? of the world size, ranging from 0 to 1
        
        distances = 1 - distances #Weightings of each plate, this is effectively a blurring as you move away and causes primary ridges
        
        base_height = 1*np.sum(np.multiply(distances, np.asarray(self.heights)[np.newaxis, np.newaxis, :]), axis = -1)
        #return(base_height) This point gives a very old, eroded landscape, similar to canyons or the Blue Mountains
        #Secondary shape should result in a curve that dips negative, making negative plates cause a ridge on neighbours, mimicking subduction
        distances = np.multiply(distances, np.cos(distances*2*math.pi)) #Current secondary curve method, can be replaced.
        distances = 1*np.sum(np.multiply(distances, np.asarray(self.heights)[np.newaxis, np.newaxis, :]), axis = -1)
        return(base_height, distances)
    
    def get_mountain_heights(self, x,y,weight, bias = -0.175, **kwargs):
        #output = 0.125*distances#tbd, importance of the boundary itself, to generally change elevation rather than just mountain creation
        output = np.multiply(np.maximum(weight, 0.0) , my_perl.sample(x,y,voron=True,neg_octaves=3,octaves=2,ndims=1,fade=0.3)[:,:,0])
        output = 0.05* output + bias
        output =  np.maximum(output, 0)#Add mountain texture
        #base_height += np.multiply(2 ** (-1 * base_height**2), fine_offset)*0.125 #Add hills
        return (output)
    
    def sample_nearby(self,x,y,scale = 1, sample_depth=5, **kwargs): #Deprecated?
        output = np.zeros((x.shape[0],x.shape[1],sample_depth))
        for i in range(sample_depth):
            offx = np.sin(i)*scale*0.001
            offy = np.cos(i)*scale*0.001
            output[:,:,i] = self.get_base_height(x+offx,y+offy,**kwargs)/sample_depth
        return(output)

    def layerise(self,X,Y,Z, freq = 30, weight = 0.25, **kwargs):
        layer_noise = my_perl.sample(X,Y,neg_octaves = 3, octaves = 1,ndims=3)/8
        layer_input = Z * (2-np.absolute(layer_noise[:,:,0]))
        #layer_input += np.multiply(X, layer_noise[:,:,1])
        #layer_input += np.multiply(Y, layer_noise[:,:,2])
        layer_input = layer_input*freq#number of layers to have per kilometer
        layer_output = np.sin(layer_input) + 0.35*np.sin(3*layer_input)+ 0.2*np.sin(5*layer_input)
        Z += weight*layer_output/freq
        return(Z)

    def get_height(self, x,y,mountainsca = 1, **kwargs):
        coarse_x, coarse_y, fine_x, fine_y = self.compute_offsets(x,y,**kwargs) 
        #are fine_x and fine_y necessary for anything besides mountains?
        if mountainsca == 0:
            del fine_x, fine_y
            gc.collect()
        
        base, secondary = self.get_base_height(coarse_x, coarse_y,**kwargs)
        del coarse_x, coarse_y
        gc.collect()
        river_z = self.get_rivers(x,y,base, **kwargs) #consider using secondary, or a combination of the two, or use these as weights
        if mountainsca > 0:
            mountains = self.get_mountain_heights(fine_x, fine_y, secondary,**kwargs)*mountainsca
            del fine_x, fine_y
            gc.collect()
            layered = self.layerise(x,y,base+mountains)
            return(base, mountains, layered)
        else:
            layered = self.layerise(x,y,base)
            return (base, np.zeros_like(x) , layered)