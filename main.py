from map_generator.landscape import landscape_gen
from map_generator import backend_switch as np
from map_generator.imaging_functions import normalize
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    landscape_sca: int
    my_landscape: landscape_gen
    mountainsca: float
    riversca: float
    X: np.ndarray
    Y: np.ndarray
    c_x: float
    c_y: float
    zoom: float
    res: int
    tiling: int
    min_pos: float
    max_pos: float


def load_parameters(
        world_params_path: str,
        imaging_params_path: str,
        sampling_coords_path: str,
):
    if not os.path.exists(world_params_path):
        landscape_sca = 200  # Linear scale of the world
        my_landscape = landscape_gen(landscape_sca, landscape_sca, num_plates=15, boundaries=True)

        # my_landscape.centroids = np.asarray([[  0.,200.], [200.,0.], [200.,200.], [0.,0.], [  0.,100.], [100.,0.], [100.,200.], [200.,100.],[164.74,24.057], [132.11,108.13], [ 94.64,68.81], [174.39,116.60],[ 66.48,139.19]])
        # my_landscape.heights = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.910, 0.3669, -0.2849, 0.1799, -0.0072])
        # my_landscape.slopes_x = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.04786425, -0.60944186,  0.00379029, 0.05412867, -0.14882481])
        # my_landscape.slopes_y = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-0.28876273, -0.15828599, -0.14998169, -0.04830083, -0.04525484])
        my_landscape.river_density = 10
        mountainsca = 0.2  # Height of mountains in km
        riversca = 500  # Height of rivers in m
    else:
        with open(world_params_path, "r") as f:
            lines = f.readlines()
            landscape_sca = int(lines[0].split(":")[1].strip())
            centroids = eval(lines[1].split(":")[1].strip())
            heights = eval(lines[2].split(":")[1].strip())
            slopes_x = eval(lines[3].split(":")[1].strip())
            slopes_y = eval(lines[4].split(":")[1].strip())
            river_density = int(lines[5].split(":")[1].strip())
            mountainsca = float(lines[6].split(":")[1].strip())
            riversca = float(lines[7].split(":")[1].strip())

        my_landscape = landscape_gen(landscape_sca, landscape_sca, num_plates=len(centroids) - 8, boundaries=False)
        my_landscape.centroids = np.asarray(centroids) * landscape_sca
        my_landscape.heights = np.asarray(heights)
        my_landscape.slopes_x = np.asarray(slopes_x)
        my_landscape.slopes_y = np.asarray(slopes_y)
        my_landscape.river_density = river_density

    if not os.path.exists(sampling_coords_path):  # create sampling coordinates in a grid
        if not os.path.exists(imaging_params_path):
            # Set parameters for image generation
            c_x = 0  # Offset of the map from world center
            c_y = 0
            zoom = 1  # Relative zoom, 1 = whole world without boundaries
            res = 128  # resolution in each direction
            tiling = 4  # Tiling factor for the map, 2 means 2 tiles total
        else:
            with open(imaging_params_path, "r") as f:
                lines = f.readlines()
                c_x = float(lines[0].split(":")[1].strip())
                c_y = float(lines[1].split(":")[1].strip())
                zoom = float(lines[2].split(":")[1].strip())
                res = int(lines[3].split(":")[1].strip())
                tiling = int(lines[4].split(":")[1].strip()) if len(lines) > 4 else 4

        min_pos = landscape_sca // 2 - landscape_sca // (2 * zoom)
        max_pos = landscape_sca // 2 + landscape_sca // (2 * zoom)
        x = np.linspace(min_pos, max_pos, res)
        y = np.linspace(min_pos, max_pos, res)
        x += c_x
        y += c_y

        X, Y = np.meshgrid(x, y)
    else:  # Load (arbitrarily shaped) sampling coordinates
        import pandas as pd
        coords = pd.read_csv(sampling_coords_path)
        X = coords['x'].values
        Y = coords['y'].values
        # set defaults to avoid undefined variables later
        c_x = 0; c_y = 0; zoom = 1; res = len(X); tiling = 1
        min_pos = 0; max_pos = 0

    return Parameters(
        landscape_sca=landscape_sca,
        my_landscape=my_landscape,
        mountainsca=mountainsca,
        riversca=riversca,
        X=X,
        Y=Y,
        c_x=c_x,
        c_y=c_y,
        zoom=zoom,
        res=res,
        tiling=tiling,
        min_pos=min_pos,
        max_pos=max_pos,
    )


def run(parameters: Parameters, output_prefix: str) -> None:
    landscape_sca = parameters.landscape_sca
    my_landscape = parameters.my_landscape
    mountainsca = parameters.mountainsca
    riversca = parameters.riversca
    X = parameters.X
    Y = parameters.Y
    zoom = parameters.zoom
    res = parameters.res
    tiling = parameters.tiling
    min_pos = parameters.min_pos
    max_pos = parameters.max_pos

    print(f"{(1000 * (max_pos - min_pos) / res):.2f} meters per pixel at zoom {zoom}")
    import time
    start_time = time.time()
    if tiling > 1:
        import tqdm
        print(f"Tiling the map into {tiling} parts for memory management")
        X = np.array_split(X, tiling)
        Y = np.array_split(Y, tiling)
        Z = [np.zeros_like(X[i]) for i in range(tiling)]
        for i in tqdm.tqdm(range(tiling), desc="Generating map sections"):
            base, mountains, Z_tile, _, _ = my_landscape.get_height(X[i], Y[i], offs=1.0, fine_offs=1.0, mountainsca=mountainsca, riversca=riversca, rivernoise=0.1)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)
            Z[i] = Z_tile
        Z = np.concatenate(Z, axis=0)
    else:
        base, mountains, Z, _, _ = my_landscape.get_height(X, Y, offs=1.0, fine_offs=1.0, mountainsca=mountainsca, riversca=riversca, rivernoise=1)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)
    print(f"Time taken to generate the map: {time.time() - start_time:.2f} seconds")
    print(np.min(Z))
    print(np.max(Z))
    Z = normalize(Z, output_prefix)


def save_parameters(
        parameters: Parameters,
        world_params_path: str,
        imaging_params_path: str,
        sampling_coords_path: str,
):
    # save world parameters to regenerate the same world later
    landscape_sca = parameters.landscape_sca
    my_landscape = parameters.my_landscape
    mountainsca = parameters.mountainsca
    riversca = parameters.riversca
    c_x = parameters.c_x
    c_y = parameters.c_y
    zoom = parameters.zoom
    res = parameters.res
    tiling = parameters.tiling

    if not os.path.exists(world_params_path):
        with open(world_params_path, "w") as f:
            f.write(f"World size (km): {landscape_sca}\n")
            f.write(f"Centroids (relative position x,y): {(my_landscape.centroids / landscape_sca).tolist()}\n")
            f.write(f"Heights (-1 to 1) of tectonic plates: {[float(h) for h in my_landscape.heights]}\n")
            f.write(f"Slopes X (km across world size): {my_landscape.slopes_x.tolist()}\n")
            f.write(f"Slopes Y (km across world size): {my_landscape.slopes_y.tolist()}\n")
            f.write(
                f"River density (major branching points per world size in Dendry noise): {my_landscape.river_density}\n"
            )
            f.write(f"Mountain heights (0-1): {mountainsca}\n")
            f.write(f"Dendry height (0-1000): {riversca}\n")
    if not os.path.exists(imaging_params_path) and not os.path.exists(sampling_coords_path):
        with open(imaging_params_path, "w") as f:
            f.write(f"Offset X: {c_x}\n")
            f.write(f"Offset Y: {c_y}\n")
            f.write(f"Zoom: {zoom}\n")
            f.write(f"Resolution: {res}\n")
            f.write(f"Tiling (number of splits for memory management): {tiling}\n")


def main(
        world_params_path: str = "world_parameters.txt",
        imaging_params_path: str = "imaging_parameters.txt",
        sampling_coords_path: str = "sampling_coordinates.csv",
        output_prefix: str = "output",
):
    parameters = load_parameters(
        world_params_path=world_params_path,
        imaging_params_path=imaging_params_path,
        sampling_coords_path=sampling_coords_path,
    )
    run(parameters, output_prefix=output_prefix)
    save_parameters(
        parameters,
        world_params_path=world_params_path,
        imaging_params_path=imaging_params_path,
        sampling_coords_path=sampling_coords_path,
    )


if __name__ == "__main__":
    main()