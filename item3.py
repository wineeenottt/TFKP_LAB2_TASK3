import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# исходнные данные
t_vals = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
x = 16 * np.sin(t_vals) ** 3
y = 13 * np.cos(t_vals) - 5 * np.cos(2 * t_vals) - 2 * np.cos(3 * t_vals) - np.cos(4 * t_vals)
z = x + 1j * y

N_coeff = 9 # количество гармоник: от -N_half до +N_half
N_half = N_coeff // 2 # вычисляем дискретное преобразование Фурье для кривой и нормируем


# Фурье коэффициенты
c = np.fft.fft(z) / len(z)
n = len(c)


k_list = []
c_list = []
for k in range(-N_half, N_half + 1): # перебираем гармоники от -N_half до +N_half
    if k < 0:
        idx = n + k # для отрицательных индексов используем циклический сдвиг
    else:
        idx = k
    if idx < n: # положительные индексы остаются как есть
        k_list.append(k)
        c_list.append(c[idx])

k_array = np.array(k_list)
c_array = np.array(c_list)



fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-20, 20)
ax.set_ylim(-25, 15)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title(f"Анимация сердца (N = {N_coeff})")

vector_lines, = ax.plot([], [], 'b-', lw=1, alpha=0.7)
trace_line, = ax.plot([], [], 'r-', lw=2)
head_point, = ax.plot([], [], 'ro', ms=5)

trace_x, trace_y = [], []

def animate(frame):
    global trace_x, trace_y
    t = t_vals[frame]

    current = 0 + 0j
    points = [current]
    for ck, k in zip(c_array, k_array):
        current += ck * np.exp(1j * k * t)
        points.append(current)

    vector_x = [p.real for p in points]
    vector_y = [p.imag for p in points]

    end_point = points[-1]
    trace_x.append(end_point.real)
    trace_y.append(end_point.imag)

    vector_lines.set_data(vector_x, vector_y)
    trace_line.set_data(trace_x, trace_y)
    head_point.set_data([end_point.real], [end_point.imag])

    return vector_lines, trace_line, head_point

ani = FuncAnimation(fig, animate, frames=len(t_vals), interval=20, blit=True, repeat=False)
plt.show()