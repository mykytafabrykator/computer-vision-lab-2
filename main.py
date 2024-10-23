import numpy as np
import cv2
import os
from scipy.fftpack import fft2, ifft2, fftshift


# Функція для обчислення частотного зображення
def dft_image(image):
    f_transform = fft2(image)
    f_transform_shift = fftshift(f_transform)
    return f_transform_shift


# Ідеальний НЧ фільтр
def ideal_low_pass_filter(shape, cutoff):
    P, Q = shape
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)
    H = np.double(D <= cutoff)
    return H


# Баттерворта НЧ фільтр
def butterworth_low_pass_filter(shape, cutoff, order):
    P, Q = shape
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)
    H = 1 / (1 + (D / cutoff) ** (2 * order))
    return H


# Гауса НЧ фільтр
def gaussian_low_pass_filter(shape, cutoff):
    P, Q = shape
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)
    H = np.exp(-(D ** 2) / (2 * (cutoff ** 2)))
    return H


# Ідеальний ВЧ фільтр
def ideal_high_pass_filter(shape, cutoff):
    H = 1 - ideal_low_pass_filter(shape, cutoff)
    return H


# Баттерворта ВЧ фільтр
def butterworth_high_pass_filter(shape, cutoff, order):
    H = 1 - butterworth_low_pass_filter(shape, cutoff, order)
    return H


# Гауса ВЧ фільтр
def gaussian_high_pass_filter(shape, cutoff):
    H = 1 - gaussian_low_pass_filter(shape, cutoff)
    return H


# Лапласіан
def laplacian_filter(shape):
    P, Q = shape
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    U, V = np.meshgrid(u, v)
    H = -(U ** 2 + V ** 2)
    return H


# Зворотне перетворення Фур'є з більш агресивною нормалізацією
def inverse_dft(f_transform_shift):
    f_ishift = np.fft.ifftshift(f_transform_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Більш агресивна нормалізація до діапазону 0-255
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back)) * 255
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return img_back


# Основна функція фільтрації
def apply_filter(image, filter_func, output_filename, cutoff=None, order=1):
    f_transform_shift = dft_image(image)
    if cutoff:
        H = filter_func(image.shape, cutoff, order) if order > 1 else filter_func(image.shape, cutoff)
    else:
        H = filter_func(image.shape)  # Для фільтрів, яким не потрібен cutoff
    G = H * f_transform_shift
    filtered_image = inverse_dft(G)

    # Зберегти результат
    if not os.path.exists('result_images'):
        os.makedirs('result_images')

    output_path = os.path.join('result_images', output_filename)
    cv2.imwrite(output_path, filtered_image)
    print(f"Зображення збережено як {output_path}")


if __name__ == "__main__":
    # Читання зображення
    image = cv2.imread('pic1.jpg', 0)

    # Ідеальний НЧ фільтр
    apply_filter(image, ideal_low_pass_filter, 'ideal_low_pass.jpg', cutoff=80)

    # Баттерворта НЧ фільтр
    apply_filter(image, butterworth_low_pass_filter, 'butterworth_low_pass.jpg', cutoff=80, order=2)

    # Гауса НЧ фільтр
    apply_filter(image, gaussian_low_pass_filter, 'gaussian_low_pass.jpg', cutoff=80)

    # Ідеальний ВЧ фільтр
    apply_filter(image, ideal_high_pass_filter, 'ideal_high_pass.jpg', cutoff=80)

    # Баттерворта ВЧ фільтр
    apply_filter(image, butterworth_high_pass_filter, 'butterworth_high_pass.jpg', cutoff=80, order=2)

    # Гауса ВЧ фільтр
    apply_filter(image, gaussian_high_pass_filter, 'gaussian_high_pass.jpg', cutoff=80)

    # Лапласіан
    apply_filter(image, laplacian_filter, 'laplacian_filter.jpg')
