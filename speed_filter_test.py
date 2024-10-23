import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import time


# Функція для просторової фільтрації (Собель)
def sobel_spatial_filter(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    grad_x = convolve2d(image, sobel_x, mode="same")
    grad_y = convolve2d(image, sobel_y, mode="same")

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return gradient_magnitude


def ideal_high_pass_filter(shape, cutoff=80):
    P, Q = shape
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)
    H = np.zeros_like(D)
    H[D > cutoff] = 1
    return H


def frequency_filter(image, filter_func, cutoff=80):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    H = filter_func(image.shape, cutoff)

    filtered_transform = H * f_transform_shifted

    filtered_transform_ishift = ifftshift(filtered_transform)
    filtered_image = ifft2(filtered_transform_ishift)

    filtered_image_real = np.abs(filtered_image)

    return filtered_image_real


def compare_filtering_speed(image):
    start_time_spatial = time.time()
    for _ in range(100):
        gs = sobel_spatial_filter(image)
    spatial_time = time.time() - start_time_spatial

    start_time_frequency = time.time()
    for _ in range(100):
        gf = frequency_filter(image, ideal_high_pass_filter)
    frequency_time = time.time() - start_time_frequency

    print(f"Час обчислень для просторової фільтрації: {spatial_time} секунд")
    print(f"Час обчислень для частотної фільтрації: {frequency_time} секунд")

    # Порівняння результатів значень пікселів
    d = np.max(np.abs(gs - gf))
    print(f"Максимальна різниця між результатами: {d}")


if __name__ == "__main__":
    pic2 = cv2.imread('pic2.jpg', 0)

    # Порівняння результатів і швидкості
    compare_filtering_speed(pic2)

    # Виведення результатів фільтрації
    spatial_result = sobel_spatial_filter(pic2)
    frequency_result = frequency_filter(pic2, ideal_high_pass_filter)

    # Відображення результатів
    plt.subplot(1, 2, 1)
    plt.imshow(spatial_result, cmap='gray')
    plt.title('Просторова фільтрація (Собель)')

    plt.subplot(1, 2, 2)
    plt.imshow(frequency_result, cmap='gray')
    plt.title('Частотна фільтрація (ВЧ фільтр)')

    plt.show()
