from map_generator import backend_switch as np
from map_generator.imaging_functions import normalize
from map_generator.parameters import Parameters, load_parameters, save_parameters


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