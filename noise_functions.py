import numpy as np 
import math

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
    
    def river_noise(self, x,y,octaves=np.array([[1]]),neg_octaves=np.array([[0]])):
        for octave in range (int(-1*np.max(np.ceil(neg_octaves))), int(np.max(np.ceil(octaves)))):
            print(octave)


        return(x)
    def find_nearest(self,x,y,randomness = 0.5): #Create a voronoi (or Worley noise) pattern from the same starting pattern, returning distance to nearest centroid
        lox = np.floor(x)
        loy = np.floor(y)
        sqdist = np.zeros_like(x) + 1000
        for x_off in range(-2,4):
            for y_off in range(-2,4):
                #get the random offset of that location in the pattern. Pattern is -1 to 1, scale by 0.5 keeps points from overlapping
                centroid = self.pattern(lox + x_off, loy + y_off,ndims = 2)
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
