import numpy as np
from noise_functions import my_perl
import random

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
            self.heights = [0.,0.,0.,0.,0.,0.,0.,0.]
        self.mountains = []
        self.mountains_tiles_x = []
        self.mountains_tiles_y = []
        for i in range (num_plates): 
            self.centroids.append(np.asarray([random.random()*long, random.random()*lat])) 
            self.heights.append(random.random())
        #TO DO: precalculate a perlin noise texture with size = lat*long (ie one sample per kilometre)
    
    def generate_mountains(self, n=-1, additional = False, overwrite = True):
        if additional or not self.mountains_done:
            if overwrite:
                self.mountains = []
                self.mountains_tiles_x = []
                self.mountains_tiles_y = []
            if n < 0: #Default or unaccepted values
                n = self.lat*self.long
            try:
                n = int(n)
            except:
                n = self.lat*self.long
            for _ in range(n):
                x = random.random()*self.long #TO DO: Improve location randomisation of mountains so that they don't stack on top of each other
                y = random.random()*self.lat
                base = self.get_base_height(x,y)
                if random.random() > 1/(1+abs(base)):
                    size = random.random() * 3**base #Maximum height of mountains
                    self.mountains.append(voron_mountain(x,y,size=size)) #Erosion is currently randomised, include additional methods?
                    self.mountains_tiles_x.append(x//10)
                    self.mountains_tiles_y.append(y//10)#VERY basic spatial acceleration structure. Landscape is divided into 10km tiles, and only mountains within one tile either side (incl. diagonals) are measured for height later
        self.mountains_done = True
        
        #Second version of tiling: an array of large size with separated lists of mountains per region
        #TO DO: Make this a separate step, it really slows down any repeated mountain generation (eg aiming to reach a certain number of mountains)
        self.tile_size = max(int(max([mount.size for mount in self.mountains]) +1))
        tiles_x = self.long//self.tile_size + 2
        tiles_y = self.lat//self.tile_size + 2
        self.mountains_tiled = [[[] for _ in range(tiles_y)] for i in range(tiles_x)]
        for mount in self.mountains:
            pos = mount.position
            tile_pos_x = int(pos[0]//self.tile_size)
            tile_pos_y = int(pos[1]//self.tile_size)
            self.mountains_tiled[tile_pos_x][tile_pos_x].append(mount)
        
        
    def get_base_height(self, x,y):
        
        offset = my_perl.sample(x,y,neg_octaves = 5, octaves=1,ndims=2) ##REFERENCES NOISE, UPDATE AS NECESSARY. Current method with -4 gives range of +-20km (real world ~= 250), each neg_octave doubles this
        fine_offset = [0,0]#my_perl.sample(x,y,neg_octaves = -1, octaves = 3,ndims=2) #Use ndims=3 for hill noise?
        fine_pos = np.asarray([x,y])+fine_offset
        x += offset[0]
        y += offset[1]
        #noise_height = offset[2] #Necessary? May cause tiling depending on noise
        
        distances = []
        for centre in self.centroids:
            distances.append((centre[0]-x)**2 + (centre[1]-y)**2) #Find square distance for each plate, no need for square root probably
        
        #New version of distance calculations, should be faster, and gives actual distance not square distance
        offsets = self.centroids - np.asarray([x,y])
        distances = np.linalg.norm(offsets, axis = -1) #Find distance for each plate    
        
        plate_dist = min(distances) #Distance to closest plate
        plate_num = np.argmin(distances) #distances.index(plate_dist) #ID of closest plate
        base_height = self.heights[plate_num] #Height this ground should be if there were no additional features
        
        distances[plate_num] = max(distances) #Ignore the closest plate when finding the second
        
        sec_dist = min(distances) #Distance to second closest plate
        sec_num = np.argmin(distances) #distances.index(sec_dist) #ID of second closest plate
        boundary_height=0
        
        distances[sec_num] = max(distances)
        third = min(distances)
        
        #if  sec_dist < plate_dist * 1.5: #Threshold for what counts as "at the boundary"
        #boundary_height = self.heights[sec_num]/ (1+0.01*(sec_dist-plate_dist)) #TO DO: replace this with calculation of appropriate height that includes subduction

        primary = (base_height + self.heights[sec_num] - 0.8)/1.2 #Mountain ranges or valleys
        #Symmetrical over boundary, raised for high pairs and depressed for low pairs
        #Usual value range is -1 to 1

        bound_weight = 10*max(third - 2*sec_dist + plate_dist,0)/self.lin_sca
        #How relevant the boundary is, ~= proximity. Should be in the scale 0:1 ish
        #To do? make this fade out at edges
        sec_weight = (plate_dist - sec_dist)/self.lin_sca #Relative importance of the second plate here, ranges from -1(variable) to 0 (hard maximum)
        secondary = 10*(base_height - self.heights[sec_num])*sec_weight #Subduction. 
        #Should be 0 at boundary, and pulse up or down as it goes away
        #Plates higher than neighbour go up, lower go down. Current range is ~-1 to 1, seafloor plates may be more extreme

        boundary_height = (primary + secondary) * bound_weight
        
        if boundary_height > 0: #Add mountain texture
            boundary_height += boundary_height * my_perl.sample(fine_pos[0],fine_pos[1],voron=True,neg_octaves=1,octaves=2,ndims=1)[0]
        
        #Blur with second plate using weighted sum (weights = sec_weight)
        #Note: Only uses second plate, not third, so will create artifacts around joins of multiple plates
        sec_weight = max(sec_weight*10,-0.5)
        sec_height = self.heights[sec_num] * (0.5+sec_weight)
        base_height = base_height * (0.5-sec_weight) + sec_height
        
        height = base_height + boundary_height
        
        outer_weight = max(abs(x - 0.5*self.long)/self.long,0.5, abs(y - 0.5*self.lat)/self.lat)#0.5 inside bounds, increasing as you leave
        outer_weight = max(1.5 - outer_weight,0) #Causes all height calculations to drop to 0 as you leave the zone
        return(outer_weight*height)
        
    def get_height(self, x,y):
        height = 0
        base = self.get_base_height(x,y)
        height += base
        
        mountain_height = 0
        if self.mountains_done: #If this landscape has mountains, their contribution to height is determined here
            try: #This will fail at certain edges. TO DO: fix for looped worlds?
                tile_pos_x = int(x//self.tile_size)
                tile_pos_y = int(y//self.tile_size)
                for x_off in [-1,0,1]:
                    for y_off in [-1,0,1]:
                        for mount in self.mountains_tiled[tile_pos_x+x_off][tile_pos_y+y_off]:
                            mountain_height += mount.get_height(x,y)
                
            except: #Use backup w/ default 10km tiles, checking every mountain
                for i in range(len(self.mountains)):
                    if abs(x//10-self.mountains_tiles_x[i]) < 2 and abs(y//10-self.mountains_tiles_y[i]) < 2: #Quick? check of whether a mountain should be looked at
                        mount = self.mountains[i]
                        mountain_height += mount.get_height(x,y)
            height += mountain_height
        river_height = 0 #Likewise for rivers
        if self.rivers_done:
            print("Warning: Rivers not yet implemented") #Warning, to remove once rivers have been written
            for river in self.rivers:
                river_height = river.get_height(x,y) #NOT YET IMPLEMENTED 
            height += river_height
        return(height)