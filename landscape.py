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
        
    def get_base_height_old(self, x,y, offset = 1, fine_offset =1, mountainsca=1):

        offset = my_perl.sample(x,y,neg_octaves = 3, octaves=1,ndims=2) * offset ##REFERENCES NOISE, UPDATE AS NECESSARY. Current method with -4 gives range of +-20km (real world ~= 250), each neg_octave doubles this
        fine_offset = my_perl.sample(x,y,neg_octaves = -1, octaves = 4,ndims=3) * fine_offset#Use ndims=3 for hill noise?
        
        fine_x = np.add(x, fine_offset[:,:,0])
        fine_y = np.add(y, fine_offset[:,:,1])
        #fine_pos = np.asarray([x,y])+fine_offset[0:2]
        x += offset[:,:,0]
        y += offset[:,:,1]

        x = np.array(x)
        y = np.array(y)
        #noise_height = offset[2] #Necessary? May cause tiling depending on noise
        xdist = x[:, :, np.newaxis] - self.centroids[:, 0]
        ydist = y[:, :, np.newaxis] - self.centroids[:, 1]
        distances = np.sqrt(xdist**2 + ydist**2)
        
        del xdist
        del ydist
        gc.collect()
        #Find distance for each plate 
        #xdist = np.stack([x for _ in self.centroids])
        #xdist -= np.asarray(my_landscape.centroids)[:,0] #Specify axis for broadcasting? what if shape of x == num of centroids?
        #xdist = np.square(xdist)
        #ydist = np.stack([y for _ in self.centroids])
        #ydist -= np.asarray(my_landscape.centroids)[:,1] #Specify axis for broadcasting? what if shape of x == num of centroids?
        #ydist = np.square(ydist)
        #distances = xdist + ydist
        #distances = np.sqrt(distances)    

        dist_argsort = np.argsort(distances, axis = -1)
        distances = np.sort(distances, axis = -1)
        plate_dist = distances[:,:,0]#Distance to closest plate
        sec_dist = distances[:,:,1]
        third = distances[:,:,2]

        plate_num = dist_argsort[:,:,0]#ID of closest plate
        base_height = np.take(self.heights, plate_num) #Height this ground should be if there were no additional features
        sec_num = dist_argsort[:,:,1] #ID of second closest plate, ID of third not used?
        sec_height = np.take(self.heights, sec_num)
        
        del distances
        del dist_argsort
        gc.collect()
        #TO DO: replace this with calculation of appropriate height that includes subduction

        primary = (base_height + sec_height - 0.8)/1.2 #Mountain ranges or valleys
        #Symmetrical over boundary, raised for high pairs and depressed for low pairs
        #Usual value range is -1 to 1
        bound_weight = 10*np.maximum(third - 2*sec_dist + plate_dist,0)/self.lin_sca
        #How relevant the boundary is, ~= proximity. Should be in the scale 0:1 ish
        sec_weight = (plate_dist - sec_dist)/self.lin_sca #Relative importance of the second plate here, ranges from -1(variable) to 0 (hard maximum)
        secondary = 1*(base_height - sec_height)*sec_weight #DISCONTINUOUS!
        #Subduction. 
        
        #Should be 0 at boundary, and pulse up or down as it goes away
        #Plates higher than neighbour go up, lower go down. Current range is ~-1 to 1, seafloor plates may be more extreme

        boundary_height = (primary + secondary) * bound_weight
        return(boundary_height)
        hill_height = fine_offset[:,:,2] * np.maximum(np.minimum(1,1-bound_weight),0)*0.1
        #Add mountain texture
        boundary_height += mountainsca * np.multiply(np.maximum(boundary_height, 0) , my_perl.sample(fine_x,fine_y,voron=True,neg_octaves=2,octaves=2,ndims=1,fade=0.3)[:,:,0])###IS THIS SCALE CORRECT?)

        #Blur with second plate using weighted sum (weights = sec_weight)
        #Note: Only uses second plate, not third, so will create artifacts around joins of multiple plates
        sec_weight = np.maximum(sec_weight*10,-0.5)
        sec_height = sec_height * (0.5+sec_weight)
        
        
        base_height = base_height * (0.5-sec_weight) + sec_height + boundary_height + hill_height

        
        outer_weight = np.maximum(np.absolute(x - 0.5*self.long)/self.long,0.5, np.absolute(y - 0.5*self.lat)/self.lat)#0.5 inside bounds, increasing as you leave
        outer_weight = np.maximum(1.5 - outer_weight,0) #Causes all height calculations to drop to 0 as you leave the zone
        return(outer_weight*base_height)
    
    
    
    
    def get_base_height(self, x,y, offs = 1, fine_offs =1, mountainsca=1):

        offset = my_perl.sample(x,y,neg_octaves = 4, octaves=-1,ndims=2)  ##REFERENCES NOISE, UPDATE AS NECESSARY. Current method with -4 gives range of +-20km (real world ~= 250), each neg_octave doubles this
        fine_offset = my_perl.sample(x,y,neg_octaves = 1, octaves = 4,ndims=3) #Use ndims=3 for hill noise?
        offset =  np.add(fine_offset[:,:,0:2],offset) * offs
        fine_offset = np.add(fine_offset,my_perl.sample(x,y,neg_octaves = -4, octaves = 8,ndims=3)) * fine_offs


        fine_x = np.add(x, fine_offset[:,:,0])
        fine_y = np.add(y, fine_offset[:,:,1])
        fine_offset = fine_offset[:,:,2]
        #fine_pos = np.asarray([x,y])+fine_offset[0:2]
        x += offset[:,:,0]
        y += offset[:,:,1]

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
        
        #base_height += 0.125*distances#tbd, importance of the boundary itself, to generally change elevation rather than just mountain creation
        base_height = base_height + np.maximum(0.05*mountainsca * np.multiply(np.maximum(distances, 0.0) , my_perl.sample(fine_x,fine_y,voron=True,neg_octaves=3,octaves=2,ndims=1,fade=0.3)[:,:,0]) -0.175, 0)#Add mountain texture
        #base_height += np.multiply(2 ** (-1 * base_height**2), fine_offset)*0.125 #Add hills
        return (base_height)
    
    def sample_nearby(self,x,y,scale = 1, sample_depth=5, **kwargs):
        output = np.zeros((x.shape[0],x.shape[1],sample_depth))
        for i in range(sample_depth):
            offx = np.sin(i)*scale*0.001
            offy = np.cos(i)*scale*0.001
            output[:,:,i] = self.get_base_height(x+offx,y+offy,**kwargs)/sample_depth
        return(output)
    def rivers(self,x,y,z,scale = 1, sample_depth=5, **kwargs):
        output = np.zeros_like(x)
        w = 1/sample_depth
        for i in range(sample_depth):
            offx = np.sin(i)*scale*0.001
            offy = np.cos(i)*scale*0.001
            output+= np.where(z>self.get_base_height(x+offx,y+offy,**kwargs),w,0)
        return(output)
    def layerise(self,X,Y,Z):
        layer_noise = my_perl.sample(X,Y,neg_octaves = 3, octaves = 1,ndims=3)/8
        layer_input = Z * (2-np.absolute(layer_noise[:,:,0]))
        #layer_input += np.multiply(X, layer_noise[:,:,1])
        #layer_input += np.multiply(Y, layer_noise[:,:,2])
        layer_freq = 30#number of layers to have per kilometer
        layer_input = layer_input*layer_freq
        layer_output = np.sin(layer_input) + 0.35*np.sin(3*layer_input)+ 0.2*np.sin(5*layer_input)
        Z += 0.25*layer_output/layer_freq
        return(Z)

    def get_height(self, x,y,**kwargs):
        height = 0
        base = self.get_base_height(x,y,**kwargs)
        #height += base
        
        return(base) #Currently mountains and rivers are unused, and not adapted for vectorisation