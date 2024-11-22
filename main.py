import cv2
import numpy as np

TILE_SIZE = 50
ENLARGEMENT = 8
RES = 5
TILE_BLOCK_SIZE = min(RES, TILE_SIZE)


def process_tile(tile_paths):
    image = cv2.imread(tile_paths)
    mosaic_tile = cv2.resize(image, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    matching_tile = cv2.resize(image, (TILE_SIZE/TILE_BLOCK_SIZE , TILE_SIZE/TILE_BLOCK_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return (mosaic_tile, matching_tile) 


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
    
    mathcing_mosaic = cv2.resize(large_mosaic, (int(w_new / TILE_BLOCK_SIZE), int(h_new / TILE_BLOCK_SIZE)), interpolation=cv2.INTER_LANCZOS4)
    return (large_mosaic, mathcing_mosaic)

        

def compute_diff(image_crop, tile):
    image_crop = np.mean(image_crop, axis=2)
    tile_crop = np.mean(tile, axis=2)
    distance = np.sum(np.sqrt((image_crop - tile_crop) ** 2))
    return distance 


image = cv2.imread('600528.png')
print(compute_diff(image,image))