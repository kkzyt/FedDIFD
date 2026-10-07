import cv2
from hough2image import process

input_img = cv2.imread("./test_imgs/dog2.png")
results = process(
    input_image=input_img,
    prompt="cute dog",
    a_prompt="best quality, extremely detailed",
    n_prompt="lowres, blurry, bad anatomy",
    num_samples=1,
    image_resolution=512,
    detect_resolution=512,
    ddim_steps=20,
    guess_mode=False,
    strength=1.0,
    scale=9.0,
    seed=-1,
    eta=0.0,
    value_threshold=0.1,
    distance_threshold=0.1
)

print(results)


cv2.imwrite(f"output_{0}.png", results[1])
