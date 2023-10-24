import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np

# Определение функции
def f(x1, x2, x3):
    return 2 * x1**2 + 3 * x2**2 + x3**2 - x1 * x2 + (x1 * x3) / 2 + 10 * x1

def graphic_plotly():
    # Создание сетки значений
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x3 = f(x1, x2, x1)  # x3 зависит от x1 и x2

    # Создание трехмерного графика с Plotly
    trace = go.Surface(z=x3, x=x1, y=x2)
    layout = go.Layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(showgrid=False, showticklabels=False, showline=False),
            yaxis=dict(showgrid=False, showticklabels=False, showline=False),
            zaxis=dict(showgrid=False, showticklabels=False, showline=False)
        )
    )
    fig = go.Figure(data=[trace], layout=layout)

    # Отображение интерактивного графика
    fig.show()

def graphic_mat():
    # Создание сетки значений
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x3 = f(x1, x2, x1)  # x3 зависит от x1 и x2

    # Создание 3D-графика
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, x3, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2, x3)')

    plt.show()

graphic_plotly()