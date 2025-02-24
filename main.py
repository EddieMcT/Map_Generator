from landscape import landscape_gen
import numpy as np
from imaging_functions import normalize

landscape_sca = 200 #Linear scale of the world
my_landscape = landscape_gen(landscape_sca,landscape_sca,num_plates=5,boundaries = True)


my_landscape.centroids = np.asarray([[  0.         ,200.        ], [200.           ,0.        ], [200.         ,200.        ], [  0.           ,0.        ], [  0.         ,100.        ], [100.        ,   0.        ], [100.         ,200.        ], [200.         ,100.        ],[164.74198475  ,24.05747735], [132.11272711 ,108.1256016 ], [ 94.63720828  ,68.80952352], [174.38847647 ,116.60099788],[ 66.47637797 ,139.18754889]])
my_landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.11070644243225414, 0.3668895455363128, -0.28487031743460167, 0.17991270123632352, -0.007167784173236491])
c_x =0 #Offset of the map from world center
c_y = 0
zoom = 1#Relative zoom, 1 = whole world without boundaries
res = 128#resolution in each direction

min_pos = landscape_sca//2 - landscape_sca//(2*zoom)
max_pos = landscape_sca//2 + landscape_sca//(2*zoom)
x = np.linspace(min_pos,max_pos,res)
y = np.linspace(min_pos,max_pos,res)
x += c_x
y += c_y

X, Y = np.meshgrid(x,y)
print(1000*(max_pos-min_pos)/res)
base, mountains, Z = my_landscape.get_height(X,Y, offs = 1.0, fine_offs =1.0, mountainsca = 1.0)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)

print(np.min(Z))
print(np.max(Z))
Z = normalize(Z, "out_close_2801.png")