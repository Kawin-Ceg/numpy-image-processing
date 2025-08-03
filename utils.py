import numpy as np

def rgb_to_grayscale(image_np):
   
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        grayscale = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])
        return grayscale.astype(np.uint8)
    else:
        raise ValueError("Input image must be RGB (3 channels)")

def apply_blur(image_np, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    h, w = image_np.shape
    pad = kernel_size // 2
    padded_image = np.pad(image_np, pad, mode='constant')
    output = np.zeros_like(image_np)
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    return output.astype(np.uint8)

def sobel_edge_detection(image_np):
    gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    
    gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    
    h, w = image_np.shape
    padded_image = np.pad(image_np, 1, mode='constant')
    edge_output = np.zeros_like(image_np, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+3, j:j+3]
            sx = np.sum(region * gx)
            sy = np.sum(region * gy)
            edge_output[i, j] = np.sqrt(sx**2 + sy**2)
    
    edge_output = np.clip(edge_output, 0, 255)
    return edge_output.astype(np.uint8)

def rotate_image(image_np, angle):
    if angle == 90:
        return np.rot90(image_np, k=1)
    elif angle == 180:
        return np.rot90(image_np, k=2)
    elif angle == 270:
        return np.rot90(image_np, k=3)
    else:
        raise ValueError("Angle must be 90, 180, or 270")

def crop_image(image_np, top, left, height, width):
    return image_np[top:top+height, left:left+width]

def histogram_equalization(image_np):
    flat = image_np.flatten()
    hist, bins = np.histogram(flat, bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    equalized = cdf_normalized.filled(0).astype(np.uint8)
    return equalized[flat].reshape(image_np.shape)
