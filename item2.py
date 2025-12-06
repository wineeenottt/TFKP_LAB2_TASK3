import numpy as np
import matplotlib.pyplot as plt

N_points = 1000
t = np.linspace(0, 2 * np.pi, N_points, endpoint=False)

# уравнения сердца
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
z = x + 1j * y

# вычисляем коэффициенты Фурье
c = np.fft.fft(z) / N_points # применяем дискретное преобразование Фурье и нормируем на N точек

# функция для вычисления частичной суммы ряда Фурье
def fourier_partial_sum(t, c, N_max):
    result = np.zeros_like(t, dtype=complex) # инициализируем массив для суммы
    N = len(c) # получаем длину массива коэффициентов Фурье
    for k in range(-N_max, N_max + 1): # перебираем коэффициенты от -N_max до N_max
        if k < 0: # если индекс отрицательный, корректируем его для массива
            idx = N + k
        else:
            idx = k
        if idx >= N: # если индекс выходит за границы, пропускаем
            continue
        ck = c[idx] # берем соответствующий коэффициент Фурье
        result += ck * np.exp(1j * k * t) # прибавляем член ряда Фурье
    return result # возвращаем частичную сумму ряда

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(z.real, z.imag, 'k-', linewidth=2.5)
ax1.set_title("Оригинал", fontsize=14)
ax1.set_xlabel("Re(z)")
ax1.set_ylabel("Im(z)")
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.6)

N_list = [1, 2, 3, 4]
colors = ['red', 'blue', 'green', 'orange']

for i, N_val in enumerate(N_list):
    z_approx = fourier_partial_sum(t, c, N_val)
    ax2.plot(z_approx.real, z_approx.imag,
             color=colors[i],
             linewidth=1.5,
             label=f'N={N_val}')

ax2.set_title("Частичные суммы Фурье", fontsize=14)
ax2.set_xlabel("Re(z)")
ax2.set_ylabel("Im(z)")
ax2.axis('equal')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()