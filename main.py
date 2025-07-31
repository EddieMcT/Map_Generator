from map_generator.landscape import landscape_gen
from map_generator import backend_switch as np
from map_generator.imaging_functions import normalize
import os

if not os.path.exists("world_params.txt"):
    landscape_sca = 200 #Linear scale of the world
    my_landscape = landscape_gen(landscape_sca,landscape_sca,num_plates=5,boundaries = True)


    my_landscape.centroids = np.asarray([[  0.,200.], [200.,0.], [200.,200.], [0.,0.], [  0.,100.], [100.,0.], [100.,200.], [200.,100.],[164.74,24.057], [132.11,108.13], [ 94.64,68.81], [174.39,116.60],[ 66.48,139.19]])
    my_landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.910, 0.3669, -0.2849, 0.1799, -0.0072])
    my_landscape.slopes_x = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.04786425, -0.60944186, 0.00379029, 0.05412867, -0.14882481])
    my_landscape.slopes_y = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.28876273, -0.15828599, -0.14998169, -0.04830083, -0.04525484])
    my_landscape.river_density = 20
else:
    with open("world_params.txt", "r") as f:
        lines = f.readlines()
        landscape_sca = int(lines[0].split(":")[1].strip())
        centroids = eval(lines[1].split(":")[1].strip())
        heights = eval(lines[2].split(":")[1].strip())
        slopes_x = eval(lines[3].split(":")[1].strip())
        slopes_y = eval(lines[4].split(":")[1].strip())
        river_density = int(lines[5].split(":")[1].strip())

    my_landscape = landscape_gen(landscape_sca, landscape_sca, num_plates=len(centroids)-8, boundaries=False)
    my_landscape.centroids = np.asarray(centroids)
    my_landscape.heights = np.asarray(heights)
    my_landscape.slopes_x = np.asarray(slopes_x)
    my_landscape.slopes_y = np.asarray(slopes_y)
    my_landscape.river_density = river_density

if not os.path.exists("sampling_coordinates.csv"): # create sampling coordinates in a grid
    # Set parameters for image generation
    c_x =0 #Offset of the map from world center
    c_y = 0
    zoom = 1 # Relative zoom, 1 = whole world without boundaries
    res = 128 # resolution in each direction

    min_pos = landscape_sca//2 - landscape_sca//(2*zoom)
    max_pos = landscape_sca//2 + landscape_sca//(2*zoom)
    x = np.linspace(min_pos,max_pos,res)
    y = np.linspace(min_pos,max_pos,res)
    x += c_x
    y += c_y

    X, Y = np.meshgrid(x,y)
else: # Load (arbitrarily shaped) sampling coordinates
    import pandas as pd
    coords = pd.read_csv("sampling_coordinates.csv")
    X = coords['x'].values
    Y = coords['y'].values
print(1000*(max_pos-min_pos)/res)
base, mountains, Z, _, _ = my_landscape.get_height(X,Y, offs = 1.0, fine_offs =1.0, mountainsca = 1.0)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)

print(np.min(Z))
print(np.max(Z))
Z = normalize(Z, "out_close_2506.png")

# save world parameters to regenerate the same world later
if not os.path.exists("world_params.txt"):
    with open("world_params.txt", "w") as f:
        f.write(f"World size: {landscape_sca}\n")
        f.write(f"Centroids: {my_landscape.centroids.tolist()}\n")
        f.write(f"Heights: {my_landscape.heights.tolist()}\n")
        f.write(f"Slopes X: {my_landscape.slopes_x.tolist()}\n")
        f.write(f"Slopes Y: {my_landscape.slopes_y.tolist()}\n")
        f.write(f"River density: {my_landscape.river_density}\n")