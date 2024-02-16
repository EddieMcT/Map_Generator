#from imaging_functions import show_map, show_map_3d
#from noise_functions import perlin_generator, julia, my_perl
#from mountains import mountain, voron_mountain
from landscape import landscape_gen

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cv2
#print(matplotlib.get_backend())
from matplotlib import pyplot as plt
landscape_sca = 200 #Linear scale of the world

my_landscape = landscape_gen(landscape_sca,landscape_sca,num_plates=5,boundaries = True)


my_landscape.centroids = np.asarray([[  0.         ,200.        ], [200.           ,0.        ], [200.         ,200.        ], [  0.           ,0.        ], [  0.         ,100.        ], [100.        ,   0.        ], [100.         ,200.        ], [200.         ,100.        ],[164.74198475  ,24.05747735], [132.11272711 ,108.1256016 ], [ 94.63720828  ,68.80952352], [174.38847647 ,116.60099788],[ 66.47637797 ,139.18754889]])
my_landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.11070644243225414, 0.3668895455363128, -0.28487031743460167, 0.17991270123632352, -0.007167784173236491])
c_x =-5# 0 #Offset of the map from world center
c_y = 10#35#
zoom = 25#Relative zoom, 1 = whole world without boundaries
res = 127#4033 #resolution in each direction 8129

min_pos = landscape_sca//2 - landscape_sca//(2*zoom)
max_pos = landscape_sca//2 + landscape_sca//(2*zoom)
x = np.linspace(min_pos,max_pos,res)
y = np.linspace(min_pos,max_pos,res)
x += c_x
y += c_y

running = True
X, Y = np.meshgrid(x,y)
if running:
    print(1000*(max_pos-min_pos)/res)
    Z = my_landscape.get_height(X,Y, offs = 1.0, fine_offs =1.0, mountainsca = 1.0)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)
    X, Y = np.meshgrid(x,y)
else:
    from noise_functions import my_perl
#Z = np.sum(my_perl.sample(X,Y,neg_octaves = 1, octaves = 4,ndims=3), axis = -1)



if False: #3D plotting, disused
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(20,20))
    # =============
    # 3D plot
    # =============
    # set up the axes for the 3D plot
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, linewidth=0, cmap = matplotlib.cm.twilight, antialiased=False)
    ax.set_box_aspect((1,1,0.075))

    ax.set_zlim(-1,3)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
if False:
    R = my_landscape.sample_nearby(X,Y, sample_depth=20)
    R = np.subtract(R, Z[:,:,np.newaxis])
    R = np.where(R>0,1,0)
    print(R.shape)
    Z = np.sum(R, axis = -1)
elif False:
    Z = my_landscape.rivers(X,Y, Z,sample_depth=200)


X, Y = np.meshgrid(x,y)
Z = my_landscape.layerise(X,Y,Z)

if running:
    print(np.min(Z))
    Z = Z + -1 * np.min(Z)
    print(np.max(Z))
    Z = 65535*Z /np.max(Z) - 0
    print(np.max(Z))
    Z = Z.astype(np.uint16)

    print(np.max(Z))
    plt.imshow(Z*0.5 + 0.5)
    plt.show()
    cv2.imwrite("out_close_2801.png",Z)
else:
    my_perl.river_noise(X,Y, X)