import numpy as np
import matplotlib.pyplot as plt

# количество точек, из которых будет состоять сердце
N_points = 1000

# массив из N точек значений t, равномерно распределённых от 0 до 2π
t = np.linspace(0, 2 * np.pi, N_points, endpoint=False)

# уравнения сердца
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

# комплексные числа
z = x + 1j * y

# сохраняем рисунок как набор последовательно расположенных точек
np.savetxt('heart_points.txt', np.column_stack((z.real, z.imag)),
           header='Re(z) Im(z)', fmt='%.6f')

plt.figure(figsize=(6, 6))
plt.plot(z.real, z.imag, 'r-', linewidth=2)
plt.title("Сердце")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.show()