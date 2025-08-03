from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import rgb_to_grayscale, apply_blur,sobel_edge_detection,rotate_image,crop_image,histogram_equalization

image = Image.open("input/image.png").convert("RGB")
image_np = np.array(image)

gray_img = rgb_to_grayscale(image_np)
Image.fromarray(gray_img).save("output/grayscale.png")

blurred_img = apply_blur(gray_img)
Image.fromarray(blurred_img).save("output/blurred.png")

edges = sobel_edge_detection(gray_img)
Image.fromarray(edges).save("output/edges.png")

rotated_img = rotate_image(gray_img, 90)
Image.fromarray(rotated_img).save("output/rotated_90.png")

cropped_img = crop_image(gray_img, 50, 50, 100, 100)
Image.fromarray(cropped_img).save("output/cropped.png")

equalized_img = histogram_equalization(gray_img)
Image.fromarray(equalized_img).save("output/equalized.png")

plt.imshow(equalized_img, cmap='gray')
plt.title("Histogram Equalized Image")
plt.axis("off")
plt.show()