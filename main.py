from pathlib import Path

from map_generator import backend_switch as np
from map_generator.globals import REPO_ROOT
from map_generator.imaging_functions import normalize
from map_generator.parameters import load_params, Params, create_landscape


def run(params: Params, autosave=True) -> None:
    root_params = params.root_params
    my_landscape = create_landscape(root_params.world)
    mountainsca = root_params.world.mountain_heights
    riversca = root_params.world.river_scale
    X = params.X
    Y = params.Y
    zoom = root_params.imaging.zoom
    res = root_params.imaging.resolution
    tiling = root_params.imaging.tiling
    min_pos = root_params.min_pos()
    max_pos = root_params.max_pos()

    print(f"{(1000 * (max_pos - min_pos) / res):.2f} meters per pixel at zoom {zoom}")
    import time
    start_time = time.time()
    if tiling > 1:
        import tqdm
        print(f"Tiling the map into {tiling} parts for memory management")
        X = np.array_split(X, tiling)
        Y = np.array_split(Y, tiling)
        Z = [np.zeros_like(X[i]) for i in range(tiling)]
        base = [np.zeros_like(X[i]) for i in range(tiling)]
        mountains = [np.zeros_like(X[i]) for i in range(tiling)]
        river_z = [np.zeros_like(X[i]) for i in range(tiling)]
        secondary = [np.zeros_like(X[i]) for i in range(tiling)]
        for i in tqdm.tqdm(range(tiling), desc="Generating map sections"):
            base[i], mountains[i], Z[i], river_z[i], secondary[i] = my_landscape.get_height(X[i], Y[i], offs=0.5, fine_offs=1.0, mountainsca=mountainsca, riversca=riversca, rivernoise=0.4)
        Z = np.concatenate(Z, axis=0)
        base = np.concatenate(base, axis=0)
        mountains = np.concatenate(mountains, axis=0)
        river_z = np.concatenate(river_z, axis=0)
        secondary = np.concatenate(secondary, axis=0)
    else:
        base, mountains, Z, river_z, secondary = my_landscape.get_height(X, Y, offs=0.5, fine_offs=1.0, mountainsca=mountainsca, riversca=riversca, rivernoise=0.4)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)
    print(f"Time taken to generate the map: {time.time() - start_time:.2f} seconds")
    print(np.min(Z))
    print(np.max(Z))
    if autosave:
        Z = normalize(Z, root_params.timestamped_output_folder)
    else:
        Z = normalize(Z, output_folder=None)
    return base, mountains, Z, river_z, secondary


def main(
        input_folder: Path = REPO_ROOT / "params/default",
        output_folder: Path = REPO_ROOT / "output",
):
    params = load_params(input_folder=input_folder, output_folder=output_folder)
    _,_,_,_,_ = run(params)
    params.root_params.save()

if __name__ == "__main__":
    main()
