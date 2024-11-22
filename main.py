import cv2
import numpy as np
import os
TILE_SIZE = 50
ENLARGEMENT = 1
RES = 5
TILE_BLOCK_SIZE = min(RES, TILE_SIZE)


def process_tile(tile_path):
    image = cv2.imread(tile_path)
    mosaic_tile = cv2.resize(image, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    matching_tile = cv2.resize(image, (int(TILE_SIZE/TILE_BLOCK_SIZE) , int(TILE_SIZE/TILE_BLOCK_SIZE)), interpolation=cv2.INTER_LANCZOS4)
    return (mosaic_tile, matching_tile) 

def get_tiles_data(tiles_folder):
    tiles_paths = [os.path.join(tiles_folder, filename) for filename in os.listdir(tiles_folder)]
    tiles_data = [process_tile(path) for path in tiles_paths]
    return tiles_data
    


def process_target_image(image_paths):
    image = cv2.imread(image_paths)
    
    if image is None:
        raise ValueError(f"Cannot read the image at {image_paths}")
    
    h, w = image.shape[:2]
    
    h_new = h * ENLARGEMENT
    w_new = w * ENLARGEMENT
    
    large_mosaic = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)
    
    h_diff =  int((h_new % TILE_SIZE) / 2)
    w_diff = int((w_new % TILE_SIZE) / 2)
    
    #crop to remove extra pixels
    large_mosaic = large_mosaic[h_diff : h_new-h_diff, w_diff : w_new - w_diff, :]
    
    matching_mosaic = cv2.resize(large_mosaic, (int(w_new / TILE_BLOCK_SIZE), int(h_new / TILE_BLOCK_SIZE)), interpolation=cv2.INTER_LANCZOS4)
    return (large_mosaic, matching_mosaic)

        

def compute_diff(image_crop, tile):
    image_crop = np.mean(image_crop, axis=2)
    print(image_crop.shape)
    tile_crop = np.mean(tile, axis=2)
    print(tile_crop.shape)
    distance = np.sum(np.sqrt((image_crop - tile_crop) ** 2))
    return distance 


def get_best_fit_tile(image_crop, tiles_data):
    index_best_fit_tile = 0
    min_diff = 9999999999999999
    for i, tile in enumerate(tiles_data):
        match_tile = tile[1]
        diff = compute_diff(image_crop, match_tile)
        if diff <= min_diff:
            min_diff = diff
            index_best_fit_tile = i
    return index_best_fit_tile
            
        
def save_tile(target_image, tile, tile_index):
    row, col = tile_index
    target_image[row * TILE_SIZE : row * TILE_SIZE + TILE_SIZE, col * TILE_SIZE : col * TILE_SIZE + TILE_SIZE ] = tile
    return target_image


if __name__ == '__main__':
    #get tiles data
    tiles_data = get_tiles_data('tiles')
    #process target image
    large_mosaic, matching_mosaic = process_target_image('600528.png')
    
    h, w = large_mosaic.shape[:2]
    dx_num = int(w / TILE_SIZE)
    dy_num = int(h/ TILE_SIZE)
    
    s = TILE_SIZE/TILE_BLOCK_SIZE
    for row in range(dx_num):
        for col in range(dy_num):
            start_index_x = int(row * s)
            start_index_y = int(col * s)
            end_index_x = int(start_index_x + s)
            end_index_y = int(start_index_y + s)
            
            image_crop = matching_mosaic[start_index_x: end_index_x, start_index_y : end_index_y, :]
            #getting best_tile
            index = get_best_fit_tile(image_crop, tiles_data)
            best_fit_tile =  tiles_data[index]
            
            mosaic_tile = best_fit_tile[0]
            
            large_mosaic = save_tile(large_mosaic, mosaic_tile, (row, col))
    cv2.imshow(large_mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()