import numpy as np 
import math

class perlin_generator(): #NOTE: this is not yet Perlin noise, but is already computationally intensive
    
    def __init__(self,x,y):
        self.pattern = np.random.rand(x,y,3)*2 -1
        
    def base_sample(self,x,y): #ADD PERLIN SAMPLER HERE
        x = x%self.pattern.shape[0]
        y = y%self.pattern.shape[1]
        a = self.pattern[int(x)][int(y)]
        b = self.pattern[int(x)][(int(y)+1)%self.pattern.shape[1]]
        c = self.pattern[(int(x)+1)%self.pattern.shape[0]][int(y)]
        d = self.pattern[(int(x)+1)%self.pattern.shape[0]][(int(y)+1)%self.pattern.shape[1]]
        
        weights = [1-x%1, x%1, 1-y%1, y%1]
        
        a = a*(weights[0]*weights[2])
        b = b*(weights[0]*weights[3])
        c = c*(weights[1]*weights[2])
        d = d*(weights[1]*weights[3])
        
        return(a+b+c+d)
    
    def sample(self,x,y,octaves=1,neg_octaves=0, fade=0.5):
        output = np.asarray([0,0,0])
        for i in range(neg_octaves*-1, octaves):
            coords = np.asarray([x*2**i,y*2**i])
            c = math.cos(2 * i)
            s = math.sin(2 * i)
            qx = c * coords[0] - s * coords[1]
            qy = s * coords[0] + c * coords[1]
            output = output+ self.base_sample(qx,qy)*fade**i
        return(output)
    
    def get_height(self,x,y,channel=-1, octaves=1,neg_octaves=0, fade=0.5):
        return(self.sample(x,y,octaves, neg_octaves, fade)[channel])
    
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
