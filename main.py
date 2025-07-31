from map_generator.landscape import landscape_gen
from map_generator import backend_switch as np
from map_generator.imaging_functions import normalize
import os

if not os.path.exists("world_parameters.txt"):
    landscape_sca = 200 #Linear scale of the world
    my_landscape = landscape_gen(landscape_sca,landscape_sca,num_plates=15,boundaries = True)


    # my_landscape.centroids = np.asarray([[  0.,200.], [200.,0.], [200.,200.], [0.,0.], [  0.,100.], [100.,0.], [100.,200.], [200.,100.],[164.74,24.057], [132.11,108.13], [ 94.64,68.81], [174.39,116.60],[ 66.48,139.19]])
    # my_landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.910, 0.3669, -0.2849, 0.1799, -0.0072])
    # my_landscape.slopes_x = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.04786425, -0.60944186, 0.00379029, 0.05412867, -0.14882481])
    # my_landscape.slopes_y = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.28876273, -0.15828599, -0.14998169, -0.04830083, -0.04525484])
    my_landscape.river_density = 15
    mountainsca = 0.2 #Height of mountains in km
    riversca = 500 #Height of rivers in m
else:
    with open("world_parameters.txt", "r") as f:
        lines = f.readlines()
        landscape_sca = int(lines[0].split(":")[1].strip())
        centroids = eval(lines[1].split(":")[1].strip())
        heights = eval(lines[2].split(":")[1].strip())
        slopes_x = eval(lines[3].split(":")[1].strip())
        slopes_y = eval(lines[4].split(":")[1].strip())
        river_density = int(lines[5].split(":")[1].strip())
        mountainsca = float(lines[6].split(":")[1].strip())
        riversca = float(lines[7].split(":")[1].strip())

    my_landscape = landscape_gen(landscape_sca, landscape_sca, num_plates=len(centroids)-8, boundaries=False)
    my_landscape.centroids = np.asarray(centroids)*landscape_sca
    my_landscape.heights = np.asarray(heights)
    my_landscape.slopes_x = np.asarray(slopes_x)
    my_landscape.slopes_y = np.asarray(slopes_y)
    my_landscape.river_density = river_density

if not os.path.exists("sampling_coordinates.csv"): # create sampling coordinates in a grid
    if not os.path.exists("imaging_parameters.txt"):
        # Set parameters for image generation
        c_x =0 #Offset of the map from world center
        c_y = 0
        zoom = 1 # Relative zoom, 1 = whole world without boundaries
        res = 128 # resolution in each direction
    else:
        with open("imaging_parameters.txt", "r") as f:
            lines = f.readlines()
            c_x = float(lines[0].split(":")[1].strip())
            c_y = float(lines[1].split(":")[1].strip())
            zoom = float(lines[2].split(":")[1].strip())
            res = int(lines[3].split(":")[1].strip())

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
print(f"{(1000*(max_pos-min_pos)/res):.2f} meters per pixel at zoom {zoom}") # print the scale of the generated map in meters per pixel to the nearest 2 decimal places
base, mountains, Z, _, _ = my_landscape.get_height(X,Y, offs = 1.0, fine_offs =1.0, mountainsca = mountainsca, riversca=riversca)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)


# save world parameters to regenerate the same world later
if not os.path.exists("world_parameters.txt"):
    with open("world_parameters.txt", "w") as f:
        f.write(f"World size (km): {landscape_sca}\n")
        f.write(f"Centroids (relative position x,y): {(my_landscape.centroids/landscape_sca).tolist()}\n")
        f.write(f"Heights (-1 to 1) of tectonic plates: {[float(h) for h in my_landscape.heights]}\n")
        f.write(f"Slopes X (km across world size): {my_landscape.slopes_x.tolist()}\n")
        f.write(f"Slopes Y (km across world size): {my_landscape.slopes_y.tolist()}\n")
        f.write(f"River density (major branching points per world size in Dendry noise): {my_landscape.river_density}\n")
        f.write(f"Mountain heights (km) (redundant with Dendry): {mountainsca}\n")
        f.write(f"Dendry height (m): {riversca}\n")
if not os.path.exists("imaging_parameters.txt") and not os.path.exists("sampling_coordinates.csv"):
    with open("imaging_parameters.txt", "w") as f:
        f.write(f"Offset X: {c_x}\n")
        f.write(f"Offset Y: {c_y}\n")
        f.write(f"Zoom: {zoom}\n")
        f.write(f"Resolution: {res}\n")
        
print(np.min(Z))
print(np.max(Z))
Z = normalize(Z, "output")