import os
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm

def save_tiles_from_svs(svs_path, output_dir, tile_size=512):
    """
    Split an SVS file into tiles of a specific size and save them with a specific naming pattern.

    Args:
        svs_path (str): Path to the SVS file.
        output_dir (str): Directory to save the tiles.
        tile_size (int): Size of the tile (default: 512x512).
    """
    if os.path.exists(output_dir) and os.listdir(output_dir) :
        print(f"Skipping {svs_path}, tiles already exist in {output_dir}")
        return
    # Open the SVS file
    slide = OpenSlide(svs_path)
    level = 0  # Use level 0 for full resolution

    # Get dimensions for level 0
    width, height = slide.level_dimensions[level]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)



    # Iterate through the image to generate tiles
    total_num = 0
    for y in tqdm(range(0, height, tile_size), desc="Processing rows"):
        for x in range(0, width, tile_size):
            # Calculate tile boundaries
            x1, y1 = x, y
            x2, y2 = min(x + tile_size, width), min(y + tile_size, height)
            if x2 - x1 < tile_size or y2 - y1 < tile_size:
                continue

            # Read the region from the slide
            tile = slide.read_region((x1, y1), level, (x2 - x1, y2 - y1))

            # Remove alpha channel (convert to RGB)
            tile = tile.convert("RGB")

            # Generate filename
            filename = f"tile_0_level0_{x1}-{y1}-{x2}-{y2}.png"
            filepath = os.path.join(output_dir, filename)

            # Save the tile
            tile.save(filepath, "PNG")
            total_num += 1

    print(f"Tiles saved to {output_dir}, total number of tiles: {total_num}")


# Example usage
# stages_files = os.listdir('F:\chenhaobin\clip-pytorch\stagePredict_dataset')
stages_files = ['stage2']
for stage_file in stages_files:
    # 判断是不是文件夹
    if os.path.isdir(os.path.join('F:\chenhaobin\clip-pytorch\stagePredict_dataset', stage_file)):
        print(stage_file)

        svs_dir_path = os.path.join('F:\chenhaobin\clip-pytorch\stagePredict_dataset', stage_file, 'svs')
        output_dirs = os.path.join('F:\chenhaobin\clip-pytorch/stagePredict_dataset', stage_file, 'till')
        for svs_file in os.listdir(svs_dir_path):
            if not svs_file.endswith('.svs'):
                continue
            print(svs_file)
            svs_path = os.path.join(svs_dir_path, svs_file)
            output_dir = os.path.join(output_dirs, svs_file)
            save_tiles_from_svs(svs_path, output_dir)


# files = os.listdir('F:\chenhaobin\clip-pytorch\svs_for_stagePredict')
# for file in files:
#     print(file)
#     svs_path = os.path.join('F:\chenhaobin\clip-pytorch\svs_for_stagePredict', file)
#     output_dir = os.path.join('F:\chenhaobin\clip-pytorch/till_svs_for_stagePredict', file)
#     save_tiles_from_svs(svs_path, output_dir)
# svs_path = "F:\chenhaobin\clip-pytorch\svs_for_stagePredict\TCGA-2H-A9GG-01A-01-TS1.3224BBEA-9099-4932-B119-884D9F71FB68.svs"  # Replace with the path to your SVS file
# output_dir = "F:\chenhaobin\clip-pytorch/till_svs_for_stagePredict\TCGA-2H-A9GG-01A-01-TS1.3224BBEA-9099-4932-B119-884D9F71FB68.svs"  # Replace with your desired output directory
# save_tiles_from_svs(svs_path, output_dir)
