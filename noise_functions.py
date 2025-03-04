import numpy as np 
import math
from scipy.interpolate import CubicSpline

# =========================
# Helper functions (to move to another file?)
# =========================

def generate_spline_points(x1, y1, x2, y2, n_points=10, delta=0.05, method = 'count'):
    """
    Generate points along a spline between two points.
    For now, this is implemented as a simple linear interpolation (placeholder).
    """
    if method == 'count': #TODO implement other methods (e.g., based on distance)
        t = np.linspace(0, 1, n_points)
    sx = x1 + t * (x2 - x1)
    sy = y1 + t * (y2 - y1)
    return sx, sy

def find_nearest_points(x, y, candidate_x, candidate_y):
    #Given query points (x, y) (of any shape) and a set of candidate points per query point (candidate_x, candidate_y),
    #returns nearest tree point coordinates and square distances.
    
    x_dist = x[:,:,np.newaxis] - candidate_x
    y_dist = y[:,:,np.newaxis] - candidate_y
    sq_distances = x_dist**2 + y_dist**2
    nearest_idx = np.argmin(sq_distances, axis=-1)

    return nearest_idx, sq_distances

def index_within_subgrid(val, x, y):
    # for a 4D array of x,y,x,y (ie a subgrid per point in array) take the values indicated by x (in the last dimension) and y (in the second to last dimension)
    # Shape of val is (a,b,c,d), shape of x, y, and output = (a,b,e,f)

    # Create two placeholder arrays, values = 0:a and 0:b, to index the first two (outer) axes and allow for fancy indexing
    index_1, index_2 = np.ogrid[:val.shape[0], :val.shape[1]]
    #Project these along the later axes to match output shape
    index_1 = np.broadcast_to(index_1[:,:,np.newaxis, np.newaxis], x.shape)
    index_2 = np.broadcast_to(index_2[:,:,np.newaxis, np.newaxis], x.shape)
    return(val[index_1, index_2, y,x])

def check_inner_grid(c = None):
    # For each of the inner 3*3 grid points in a 5*5 matrix, find which of its eight neighbours has the lowest value of c
    # If no value of c is provided, use distance?
    # Expected shape of  c is (X, Y, n_points, n_points)
    # Shape of output is (X, Y, n_points-2, n_points-2) because we exclude the outer two rows and columns from consideration

    # Find the c values of neighbours for each point being calculated
    # shape=(X,Y,n_points-2, n_points-2, 3,3)
    neighbour_c = np.lib.stride_tricks.sliding_window_view(c, (3, 3), axis=(-2, -1))


    # Take the index of the lowest neighbour in the n_points-2 by n_points-2 grid
    neighbour_x = np.argmin(neighbour_c, axis=-1) 
    min_x = np.min(neighbour_c, axis=-1)
    neighbour_y = np.argmin(min_x, axis=-1) 
    # Take the neighbour_yth entry for each of neighbour_x entries
    neighbour_x = np.take_along_axis(neighbour_x, neighbour_y[:,:,:, :, None], axis=-1).squeeze()

    # This gives the index of the lowest neighbour in the 3*3 grid for each point being calculated
    # Correct to indices within original 5*5 grid:
    indices = np.indices(neighbour_x.shape)
    neighbour_x += indices[-1]
    neighbour_y += indices[-2]
    
    return neighbour_x, neighbour_y

class perlin_generator(): #NOTE: this is not yet Perlin noise, but is already computationally intensive
    
    def __init__(self,x=128,y=128,max_oct=32):
        np.random.seed(1)
        self.pattern_ref = np.random.rand(x,y,3)*2 -1 #array of random values between -1 and 1
        self.cos_lut = [math.cos(2*i) for i in range(max_oct)] #used instead of calculating trig functions per pixel at runtime
        self.sin_lut = [math.sin(2*i) for i in range(max_oct)]
        #for i in range(max_oct): #so that negative values of i correspond to negative angles
        #    self.cos_lut.append(math.cos(2*(i-max_oct)))
        #    self.sin_lut.append(math.sin(2*(i-max_oct)))
            
    def pattern(self,x,y,ndims=3):
        return(self.pattern_ref[x.astype(int)%self.pattern_ref.shape[0],y.astype(int)%self.pattern_ref.shape[1],0:ndims])
        #Converts this version back to old method (pregen noise pattern) 
        
        #This is an attempt at procedural noise, but it produces too many artifacts to replace the pattern method
        primes = [4937,6053,5843,6701,6133,7919,7823,5281,5407,5443]
        output = np.zeros(ndims)
        for i in range(ndims):
            output[i] = 0.5 - ((x*(y+i+3)*7717 + y*(x+10*i)*7907*7717)%primes[i])/primes[i] #Pseudorandom output between -0.5 and 0.5
        return(output)
    def find_grid(self, x, y, n=5, scale=1.0, epsilon=0.5, frequency = 1000):
        # Generate a grid of jittered points around each input (x,y).
        # x and y are assumed to have shape (res, res).
        # Returns tree_x and tree_y with shape (res, res, n, n).

        # Generate a regular grid of points centered around 0. The set of offsets is the same in x and y
        #offsets = np.linspace(-0.5,0.5,n, endpoint=False) * scale*n
        offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * scale #check this line for different n
        
        # Expand x and y to shape (res, res, n, n) including added offsets of (n*n)
        x_exp = np.stack([np.stack([x for _ in range(n)], axis=-1)+offset for offset in offsets], axis = -1)
        y_exp =np.stack([np.stack([y+offset for offset in offsets], axis = -1) for _ in range(n)], axis=-1)
        
        # Compute jitter based on the pattern function
        jitter = self.pattern(x_exp*frequency, y_exp*frequency, ndims=2) #having a grid size close to one means that cells receive the same jitter as their neighbours, as they are falling into the same bins in the pattern. 
        #Frequency upsamples that so that nearby cells are less likely to have the same jitter
        x_exp += epsilon*jitter[:, :,:,:, 0]
        y_exp += epsilon*jitter[:, :,:,:, 1]
        return x_exp, y_exp



    def find_grid_old(self, x, y, n=5, scale = 1): #Returns a grid of points around the input points, with n points in each direction
        #will reuse logic of find_nearest, but will return a grid of points not relative distance
        return(None)#placeholder

    def base_sample(self,x,y,**kwargs): #ADD PERLIN SAMPLER HERE, currently simple interpolation
        lox = np.floor(x)
        loy = np.floor(y)
        a = self.pattern(lox,loy,**kwargs)
        b = self.pattern(lox,(loy+1),**kwargs)
        c = self.pattern((lox+1),loy,**kwargs)
        d = self.pattern((lox+1),(loy+1),**kwargs)
        
        #weights = [1-x%1, x%1, 1-y%1, y%1]
        #a = a*(weights[0]*weights[2])
        #b = b*(weights[0]*weights[3])
        #c = c*(weights[1]*weights[2])
        #d = d*(weights[1]*weights[3])
        weight1 = x%1
        weight1 = weight1[:,:,None]
        weight1 = weight1 * weight1 * weight1 * (weight1 * (6 * weight1 - 15) + 10)
        weight0 = 1-weight1
        weight3 = y%1
        weight3 = weight3[:,:,None]
        weight3 = weight3 * weight3 * weight3 * (weight3 * (6 * weight3 - 15) + 10)
        weight2 = 1-weight3
        a *= weight0
        a *= weight2
        b *= weight0
        b *= weight3
        c *= weight1
        c *= weight2
        d *= weight1
        d *= weight3
        a = np.add(a,b)
        c = np.add(c,d)
        
        s = np.add(a,c)
        return(s)#abs(s)*s*(3-2*s))
    
    
    def sample(self,x,y,octaves=1,neg_octaves=0, fade=0.5,voron=False,ndims=3, **kwargs) -> np.ndarray: 
        output = np.zeros(ndims)
        for i in range(neg_octaves*-1, octaves):
            a = 2 ** i #Scale the starting positions by 2 for each octave
            ax = x * a
            ay = y * a
            c = self.cos_lut[i] #faster than recalculating every time, but does give a different angle for negative i values
            s = self.sin_lut[i]
            qx = c * ax - s * ay
            qy = s * ax + c * ay
            if voron:
                output = output+ self.voron(qx,qy)*fade**i
            #TO DO: Add noise variant for Dendry, wherein nearest centroid is found
            else:
                output = output+ self.base_sample(qx,qy,ndims=ndims)*fade**i
        return(output)
    
    def get_height(self,x,y,channel=-1, **kwargs):
        return(self.sample(x,y,**kwargs)[:,:,channel]) #Note that ndims >1 is irrelevant if channel is not a list/array, as only one channel will be selected
    def dendry(self, x, y, base_scale=1, octaves=5, subsampling=10, control_function=None, **kwargs):
        """
        Generate dendry (river) noise for input coordinates x, y (shape: (res, res)).
        
        Base level (octave 0):
          - Generates a 5x5 grid per evaluation point.
          - For each inner cell (i,j where i,j in 1..3) of that grid,
            finds the chosen neighbor from its 3x3 neighborhood.
            If a control_function is provided, it is used to determine
            the "height" at each grid point and the lowest height is chosen.
            If not, the nearest neighbor (by Euclidean distance) is chosen.
          - For each inner cell, a single spline is generated (using generate_spline_points)
            between the grid cell's center and its chosen neighbor.
            This yields 9 splines per evaluation point.
        
        Higher octaves:
          - For each octave, a new grid is generated using n=3 (the entire grid is used).
          - For each grid point, the nearest existing tree point (from lower octaves)
            is found and a spline is generated between them.
        
        Finally, the function computes, for each evaluation point, the minimum distance to any
        spline sample point in the union of all splines, and returns this distance field.
        """
        res = x.shape[0]
        all_tree_x = []
        all_tree_y = []
        
        # --- Base level: use a 5x5 grid ---
        tree_x, tree_y = self.find_grid(x, y, n=5, scale=base_scale)
        
        # If control_function is provided, sample it to get heights.
        # Otherwise, use Euclidean distance from the center of each cell.
        if control_function is not None:
            heights = control_function(tree_x, tree_y, **kwargs)
        else:
            # Define heights as the Euclidean distance from the grid cell center to (x,y)
            center_x = tree_x[:, :, 2, 2]  # center of the 5x5 grid
            center_y = tree_y[:, :, 2, 2]
            # Broadcast to shape (res, res, 5, 5)
            heights = np.sqrt((tree_x - center_x[..., np.newaxis, np.newaxis])**2 +
                              (tree_y - center_y[..., np.newaxis, np.newaxis])**2)
        
        # For each inner grid cell (indices 1,2,3) in the 5x5 grid,
        # generate one spline from that cell to its chosen neighbor
        for i in range(1, 4):
            for j in range(1, 4):
                # Extract the 3x3 neighborhood for cell (i,j)
                neighborhood = heights[:, :, i-1:i+2, j-1:j+2]
                # For each evaluation point, choose the neighbor index with minimum height
                min_idx = np.argmin(neighborhood.reshape(res, res, -1), axis=2)
                # Convert flat indices to 2D offsets relative to the 3x3 block
                offset_i = min_idx // 3  # values 0,1,2
                offset_j = min_idx % 3
                # Compute the absolute indices in the 5x5 grid
                chosen_i = offset_i + (i - 1)
                chosen_j = offset_j + (j - 1)
                
                # For each evaluation point, generate spline points between (i,j) and the chosen neighbor.
                for ii in range(res):
                    for jj in range(res):
                        p0_x = tree_x[ii, jj, i, j]
                        p0_y = tree_y[ii, jj, i, j]
                        p1_x = tree_x[ii, jj, chosen_i[ii, jj], chosen_j[ii, jj]]
                        p1_y = tree_y[ii, jj, chosen_i[ii, jj], chosen_j[ii, jj]]
                        sx, sy = generate_spline_points(p0_x, p0_y, p1_x, p1_y, subsampling)
                        all_tree_x.extend(sx)
                        all_tree_y.extend(sy)
        
        # --- Higher octaves: use a 3x3 grid (only inner cells needed) ---
        for octave in range(1, octaves):
            new_scale = base_scale * (2 ** octave)
            new_x, new_y = self.find_grid(x, y, n=3, scale=new_scale)
            # Here, the entire grid (3x3) is considered.
            # Find nearest tree point from the already generated tree for each new grid point.
            # new_x and new_y have shape (res, res, 3, 3)
            inner_new_x = new_x  # shape (res, res, 3, 3)
            inner_new_y = new_y
            nearest_x, nearest_y, _ = find_nearest_points(inner_new_x, inner_new_y, np.array(all_tree_x), np.array(all_tree_y))
            
            for i in range(3):
                for j in range(3):
                    for ii in range(res):
                        for jj in range(res):
                            p0_x = new_x[ii, jj, i, j]
                            p0_y = new_y[ii, jj, i, j]
                            p1_x = nearest_x[ii, jj, i, j]
                            p1_y = nearest_y[ii, jj, i, j]
                            sx, sy = generate_spline_points(p0_x, p0_y, p1_x, p1_y, subsampling)
                            all_tree_x = np.append(all_tree_x, sx)
                            all_tree_y = np.append(all_tree_y, sy)
        
        # Compute final distance for each evaluation point (x,y) to the union of all spline points.
        all_tree_x = np.array(all_tree_x)
        all_tree_y = np.array(all_tree_y)
        _, _, distances = find_nearest_points(x, y, all_tree_x, all_tree_y)
        return distances.reshape(res, res)
    
    def dendry_old(self, x, y,base_scale = 1, octaves = 5, subsampling = 10, control_function = None, **kwargs):
        #for each point, generate a 5*5 grid around it (snapping to the nearest tile possibly using my_perl.find_nearest)
        #x shape = y shape = (res,res)
        #tree_x shape = tree_y shape = (res,res,5,5)
        tree_x, tree_y = self.find_grid(x, y, n=5, scale = base_scale) #grid function, still to be implemented
        if control_function is None:
            height = np.zeros_like(tree_x)
            #for each of these 25, determine its height using the control function
            for i in range(5):
                for j in range(5):
                    height[:,:,i, j] = control_function(tree_x[:,:,i, j], tree_y[:,:,i, j], **kwargs)
            
            #for each of the inner 3*3 grid, find its lowest altitude neighbour (or itself if it is the lowest)
        else:
            #for each of the inner 3*3 grid, find its nearest neighbour
            neighbours = None #placeholder
        #create a spline between each point and its chosen neighbour with n=subsampling points, including the start and end
        #this set of points (of the splines only) is the lowest level tree, and can be used to generate a river
        #flatten to the set of points, #tree_x shape = tree_y shape = (res,res,3*3*subsampling)

        for octave in range(octaves):
            #for each octave, generate a new 5*5 grid around each starting point (x,y) 
            new_level_x, new_level_y = self.find_grid(x, y, n=5, scale = base_scale*2**octave)
            #Optional: remove the points for which a previous point is already in their grid 
            #Find the nearest point on the existing tree for each of the new points
            # add a spline from each of these to the nearest point on the existing tree
            #add points along that spline equal to subsampling (consider length based?)
        #for each input point, find the distance to the nearest point on its tree
        return(None)


    def find_nearest(self,x,y,randomness = 0.5): #Create a voronoi (or Worley noise) pattern from the same starting pattern, returning distance to nearest centroid
        lox = np.floor(x)
        loy = np.floor(y)
        sqdist = np.zeros_like(x) + 1000
        for x_off in range(-2,4):
            for y_off in range(-2,4):
                #get the random offset of that location in the pattern. Pattern is -1 to 1, scale by 0.5 keeps points from overlapping
                centroid = self.pattern(lox + x_off, loy + y_off,ndims = 1)*0.5
                centroid += [x_off, y_off]
                
                centroid = np.square(centroid)
                centroid = np.sum(centroid, axis = -1)
                
                sqdist = np.minimum(sqdist, centroid)#If this centroid is closer than previous, keep this distance (always a closest neighbour search, second closest neighbour not used here
        return(np.sqrt(sqdist)[:,:,None])
    def voron(self,x,y,randomness = 0.5): #Create a voronoi (or Worley noise) pattern from the same starting pattern, returning distance to nearest centroid
        lox = np.floor(x)
        loy = np.floor(y)
        frac = np.stack([x%1, y%1], axis = -1)
        sqdist = np.zeros_like(x) + 1000
        dist=  np.zeros_like(x) + 10#Instantiate distance as something (hopefully?) larger than all distances
        for x_off in range(-1,3):
            for y_off in range(-1,3):
                #get the random offset of that location in the pattern. Pattern is -1 to 1, scale by 0.5 keeps points from overlapping
                centroid = self.pattern(lox + x_off, loy + y_off,ndims = 2)
                centroid += [x_off, y_off]
                centroid = np.subtract(frac, centroid)#calculate vector between this centroid and the sampled location
                
                centroid = np.square(centroid)
                centroid = np.sum(centroid, axis = -1)
                
                sqdist = np.minimum(sqdist, centroid)#If this centroid is closer than previous, keep this distance (always a closest neighbour search, second closest neighbour not used here
        return(np.sqrt(sqdist)[:,:,None])
    
    
    def voron_old(self,x:float,y:float,randomness = 0.5) -> float: #DEPRECATED:, works with non-vectorised inputs
        loc = np.asarray([x,y]) #Actual location of sampled point within the pattern (looped over x and y limits of pattern)
        x = math.floor(x)
        y = math.floor(y)
        sqdist = 100
        dist=10 #Instantiate distance as something (hopefully?) larger than all distances
        for x_off in range(-1,3):
            for y_off in range(-1,3):
                centr = np.asarray([x+x_off,y+y_off])
                centroid = np.add(centr, self.pattern(centr[0],centr[1],2)*randomness) #get the random offset of that location in the pattern. Pattern is -1 to 1, scale by 0.5 keeps points from overlapping
                centroid = np.add(centroid * -1, loc) #calculate vector between this centroid and the sampled location
                new_sq = centroid[0]**2 + centroid[1]**2
                if sqdist > new_sq: #Only calculate real distance is square
                    sqdist = new_sq
                    dist = min(dist, np.linalg.norm(centroid))#If this centroid is closer than previous, keep this distance (always a closest neighbour search, second closest neighbour not used here
        return(dist)
    def profile_base_sample(self, x, y, ndims=3): #Not to be used in production code, only for profiling current version of base_sample
        import time
        # Time each major operation
        timings = {}
        
        start = time.perf_counter()
        lox = np.floor(x)
        loy = np.floor(y)
        timings['floor'] = time.perf_counter() - start
        
        start = time.perf_counter()
        a = self.pattern(lox, loy, ndims=ndims)
        b = self.pattern(lox, (loy+1), ndims=ndims)
        c = self.pattern((lox+1), loy, ndims=ndims)
        d = self.pattern((lox+1), (loy+1), ndims=ndims)
        timings['pattern_lookup'] = time.perf_counter() - start
        
        start = time.perf_counter()
        weight1 = x % 1
        weight1 = weight1[:,:,None]
        weight1 = weight1 * weight1 * weight1 * (weight1 * (6 * weight1 - 15) + 10)
        weight0 = 1-weight1
        weight3 = y % 1
        weight3 = weight3[:,:,None]
        weight3 = weight3 * weight3 * weight3 * (weight3 * (6 * weight3 - 15) + 10)
        weight2 = 1-weight3
        timings['weight_calc'] = time.perf_counter() - start
        
        start = time.perf_counter()
        a *= weight0
        a *= weight2
        b *= weight0
        b *= weight3
        c *= weight1
        c *= weight2
        d *= weight1
        d *= weight3
        a = np.add(a,b)
        c = np.add(c,d)
        s = np.add(a,c)
        timings['interpolation'] = time.perf_counter() - start
        
        return s, timings

my_perl = perlin_generator(256,256)

def julia(x,y,c = 1j, n = 5,cap = 2):
    z = x + y*1j
    start = abs(z)
    val = 0
    new = abs(z)
    for reps in range(n):
        if new > cap:
            ec = reps + 1 - math.log2(new)#*3.32192809489*0.5
            ec = max(ec,1)
            return(1-(1/ec))
        z = z**2 + c
        new = abs(z)
        if new == start:
            return(1)
    return(1)
        
class julia_gen():
    def __init__(self, c=1j,n=5):
        self.c = c
        self.n = n
    
    def get_height(self,x,y,channel=-1, octaves=1,neg_octaves=0, fade=0.5,rotation = 4): #Channel is unused, this is just to keep in line with other functions
        height = 0
        cap_factor = fade ** (neg_octaves) #TO DO: Move all possible calculations OUTSIDE of per-pixel loop
        for octave in range(-1*neg_octaves,octaves):
            c, s = np.cos(rotation*octave), np.sin(rotation*octave)
            a = c*x - s*y
            b = s*x + c*y
            a = a*1.2**octave
            b = b*1.2**octave
            height += julia(a,b,self.c,self.n)*fade**octave
            
            
        return (height*cap_factor)

class fractal_julia():
    def __init__(self, c=1j,n=5):
        self.c = c
        self.n = n
    
    def get_height(self,x,y,channel=-1, octaves=1,neg_octaves=0, fade=0.5,rotation = 4): #Channel is unused, this is just to keep in line with other functions
        height = 0
        #cap_factor = fade ** (neg_octaves) #TO DO: Move all possible calculations OUTSIDE of per-pixel loop
        angl = np.arctan2(x,y)
        new_height = julia(x,y,self.c,self.n)
        for octave in range(-1*neg_octaves,octaves):
            if new_height > 0:
                height += new_height/(octaves+neg_octaves)
                angl += rotation
                c, s = np.cos(angl), np.sin(angl)
                c = c*(1+fade-new_height)/(new_height+fade)
                s = s*(1+fade-new_height)/(new_height+fade)
                new_height = julia(c,s,self.c,self.n)
            
            
        return (height)
if __name__ == '__main__':
    x, y = np.meshgrid(np.linspace(0, 10, 4, endpoint=False), np.linspace(0, 10, 3))
    tree_x, tree_y = my_perl.find_grid(x, y, n=5, scale=1.0, epsilon=0.25)
    #print(tree_x.shape, tree_y.shape) #(3, 4, 5, 5) (3, 4, 5, 5)
    c = np.zeros_like(tree_x)
    for i in range(5):
        for j in range(5):
            c[:,:,i,j] = my_perl.sample(tree_x[:,:,i,j], tree_y[:,:,i,j], ndims=1)[:,:,0]
    print(c.shape)
    chosen_idx_x, chosen_idx_y = check_inner_grid(c)
    chosen_x = index_within_subgrid(tree_x, chosen_idx_x, chosen_idx_y)
    chosen_y = index_within_subgrid(tree_y, chosen_idx_x, chosen_idx_y)
    print(chosen_x.shape)

    spline_start_x = tree_x[...,1:-1,1:-1]
    spline_start_y = tree_y[...,1:-1,1:-1]
    spline_points = generate_spline_points(spline_start_x, spline_start_y, chosen_x, chosen_y, n_points=10)
    print(spline_points.shape)
    #flatten the tree entries to a single list of points per input point of x, y
    tree_x = tree_x.reshape(x.shape[0], x.shape[1], -1)
    tree_y = tree_y.reshape(y.shape[0], y.shape[1], -1)
    #print(tree_x.shape) #(3, 4, 25)
    nearest_in_tree, dist = find_nearest_points(x, y, tree_x, tree_y)
    #print(nearest_in_tree.shape) #(3,4)
    #print(dist.shape) #(3,4,25)
    #Select entry from dist indexed by nearest_in_tree
    dist = dist[np.arange(dist.shape[0])[:,None], np.arange(dist.shape[1]), nearest_in_tree]

    print(dist.shape)


    #proofread and test out new functions: 
    # my_perl.dendry, doublecheck iijj loop for spline generation
    # my_perl.find_grid, does it use the pattern correctly? Checked, successful
    #find_nearest_points, does it work correctly? Checked, successful
    #generate_spline_points