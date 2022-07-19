import numpy as np
from noise_functions import my_perl
import random

class landscape_gen():
    def __init__(self,lat = 10, long = 10, num_plates = 5, boundaries = True):
        self.lat = lat 
        self.long = long 
        self.mountains_done = False 
        self.rivers_done = False 
        self.centroids = []
        self.heights =[]
        if boundaries:
            self.centroids = [np.asarray([0,long]), np.asarray([lat,0]),
                              np.asarray([lat, long]), np.asarray([0,0]),
                              np.asarray([0,long*0.5]), np.asarray([lat*0.5,0]),
                              np.asarray([lat*0.5, long]), np.asarray([lat, long*0.5])] #Seafloor tiles on each corner and middles of edges. Redundant if world is made round
            self.heights = [0.,0.,0.,0.,0.,0.,0.,0.]
        self.mountains = []
        self.mountains_tiles_x = []
        self.mountains_tiles_y = []
        for i in range (num_plates): 
            self.centroids.append(np.asarray([random.random()*lat, random.random()*long])) 
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
        
        offset = my_perl.sample(x,y,neg_octaves = 6, octaves=5) ##REFERENCES NOISE, UPDATE AS NECESSARY. Current method with -4 gives range of +-20km (real world ~= 250), each neg_octave doubles this
        x += offset[0]
        y += offset[1]
        noise_height = offset[2] #Necessary? May cause tiling depending on noise
        
        distances = []
        for centre in self.centroids:
            distances.append((centre[0]-x)**2 + (centre[1]-y)**2) #Find square distance for each plate, no need for square root probably
        
        #New version of distance calculations, should be faster, and gives actual distance not square distance
        #offsets = self.centroids - np.asarray([x,y])
        #distances = np.linalg.norm(offsets, axis = -1) #Find distance for each plate    
        
        plate_dist = min(distances) #Distance to closest plate
        plate_num = distances.index(plate_dist) #ID of closest plate
        base_height = self.heights[plate_num] #Height this ground should be if there were no additional features
        
        distances[plate_num] = max(distances)
        
        sec_dist = min(distances) #Distance to second closest plate
        sec_num = distances.index(sec_dist) #ID of second closest plate
        boundary_height=0
        
        if sec_dist < plate_dist * 1.5: #Threshold for what counts as "at the boundary"
            boundary_height = self.heights[sec_num]/ (1+0.01*(sec_dist-plate_dist)) #TO DO: replace this with calculation of appropriate height that includes subduction
        
        height = base_height + boundary_height
        return(height)
        
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