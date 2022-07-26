import numpy as np 
import math

class perlin_generator(): #NOTE: this is not yet Perlin noise, but is already computationally intensive
    
    def __init__(self,x=128,y=128,max_oct=20):
        self.pattern_ref = np.random.rand(x,y,3)*2 -1 #array of random values between -1 and 1
        self.cos_lut = [math.cos(2*i) for i in range(max_oct)] #used instead of calculating trig functions per pixel at runtime
        self.sin_lut = [math.sin(2*i) for i in range(max_oct)]
        #for i in range(max_oct): #so that negative values of i correspond to negative angles
        #    self.cos_lut.append(math.cos(2*(i-max_oct)))
        #    self.sin_lut.append(math.sin(2*(i-max_oct)))
            
    def pattern(self,x:int,y:int,ndims=3) ->float:
        return(self.pattern_ref[x%self.pattern_ref.shape[0]][y%self.pattern_ref.shape[1]][0:ndims])
        #Converts this version back to old method (pregen noise pattern) 
        
        #This is an attempt at procedural noise, but it produces too many artifacts to replace the pattern method
        primes = [4937,6053,5843,6701,6133,7919,7823,5281,5407,5443]
        output = np.zeros(ndims)
        for i in range(ndims):
            output[i] = 0.5 - ((x*(y+i+3)*7717 + y*(x+10*i)*7907*7717)%primes[i])/primes[i] #Pseudorandom output between -0.5 and 0.5
        return(output)
        
    def base_sample(self,x,y): #ADD PERLIN SAMPLER HERE
        a = self.pattern(math.floor(x),math.floor(y))
        b = self.pattern(math.floor(x),(math.floor(y)+1))
        c = self.pattern((math.floor(x)+1),math.floor(y))
        d = self.pattern((math.floor(x)+1),(math.floor(y)+1))
        
        weights = [1-x%1, x%1, 1-y%1, y%1]
        
        a = a*(weights[0]*weights[2])
        b = b*(weights[0]*weights[3])
        c = c*(weights[1]*weights[2])
        d = d*(weights[1]*weights[3])
        
        s = a+b+c+d
        
        return(s)#abs(s)*s*(3-2*s))
    
    
    def sample(self,x,y,octaves=1,neg_octaves=0, fade=0.5,voron=False):
        output = np.asarray([0,0,0])
        for i in range(neg_octaves*-1, octaves):
            coords = np.asarray([x*2**i,y*2**i])
            c = self.cos_lut[i] #faster than recalculating every time, but does give a different angle for negative i values
            s = self.sin_lut[i]
            qx = c * coords[0] - s * coords[1]
            qy = s * coords[0] + c * coords[1]
            if voron:
                output = output+ self.voron(qx,qy)*fade**i
            else:
                output = output+ self.base_sample(qx,qy)*fade**i
        return(output)
    
    def get_height(self,x,y,channel=-1, **kwargs):
        #return(self.voron(x,y)) #Temporary, to test Worley noise
        return(self.sample(x,y,**kwargs)[channel])
    
    def voron(self,x:float,y:float,randomness = 0.5) -> float: #Create a voronoi (or Worley noise) pattern from the same starting pattern, returning distance to nearest centroid
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

my_perl = perlin_generator(20,20)

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
