"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)
import matplotlib.pyplot as plt


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = data.matrices
    vectors = data.vectors
    return np.sum(matrices @ vectors, axis=0)



def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)




def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    list_of_unique = []
    for row in matrix:
        list_of_unique.append(np.unique(row).tolist())
    return list_of_unique


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    t_matrix = matrix.T
    list_of_unique = []
    for row in t_matrix:
        list_of_unique.append(np.unique(row).tolist())
    return list_of_unique


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    random_gen = np.random.default_rng(seed)
    matrix = random_gen.normal(mean, std, (rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0)
    )


def plot_matrix_histograms(matrix) -> None:
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    bins = np.linspace(min_value, max_value, 4)

    for i in range(matrix.shape[0]):
        plt.figure(figsize=(8, 5))
        plt.hist(matrix[i, :], bins=bins, edgecolor="black")
        plt.title(f"Гистограмма строки {i}")
        plt.xlabel("Значение")
        plt.ylabel("Кол-во значений")
        plt.xticks(bins)
        plt.show()

    for j in range(matrix.shape[1]):
        plt.figure(figsize=(8, 5))
        plt.hist(matrix[:, j], bins=bins, edgecolor="black")
        plt.title(f"Гистограмма столбца {j}")
        plt.xlabel("Значение")
        plt.ylabel("Кол-во значений")
        plt.xticks(bins)
        plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    matrix = np.where((np.indices((rows, columns)).sum(axis=0) % 2 == 0), first, second)
    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    x_1 = (image_width - width) // 2
    x_2 = x_1 + width
    y_1 = (image_height - height) // 2
    y_2 = y_1 + height

    rectangle = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    rectangle[:] = background_color
    rectangle[y_1:y_2, x_1:x_2] = shape_color

    return rectangle



def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    x_0 = (image_width - 1) / 2
    y_0 = (image_height - 1) / 2
    y, x = np.indices((image_height, image_width))

    ellipse = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    ellipse[:] = background_color
    mask = (((x - x_0) / semi_axis_x) ** 2 + ((y - y_0) / semi_axis_y) ** 2 <= 1)
    ellipse[mask] = shape_color

    return ellipse




def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    arr = np.asarray(data.values)
    window = data.window
    mean = float(np.mean(arr))
    variance = float(np.var(arr))
    std = float(np.std(arr))

    is_max = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])
    local_maxima_indices = np.where(is_max)[0] + 1

    is_min = (arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:])
    local_minima_indices = np.where(is_min)[0] + 1

    moving_average = np.array([
        np.mean(arr[i:i + window])
        for i in range(len(arr) - window + 1)
    ])

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average
    )



def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = np.max(labels) + 1

    arr = np.zeros((len(labels), class_count))
    arr[np.arange(len(labels)), labels] = 1
    return arr

