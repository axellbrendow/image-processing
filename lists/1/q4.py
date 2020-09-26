"""
Segmentação pela técnica limiar
Rotulação
Segmentação por crescimento de regiões
Circularidade, código de cadeia, curva de Phi e número de forma dos objetos
Esqueletos dos objetos

CIRCULARIDADE = P² / (4 * pi * A)
P = PERÍMETRO
A = ÁREA
"""

from typing import List, Dict, Tuple, Callable
from functools import reduce
import math


def print_matrix(matrix: List[List[int]], default_value: str = "."):
    num_lines = len(matrix)
    num_columns = len(matrix[0])

    for i in range(num_lines):
        for j in range(num_columns):
            print(f'{default_value if matrix[i][j] == 0 else matrix[i][j]: ^4}', end='')
        print()


def print_matrix_with_points(
    num_lines: int,
    num_columns: int,
    points: List[Tuple[int, int]],
    values: List[int]
):
    matrix = [["-1" for _ in range(num_columns)] for _ in range(num_lines)]

    for i, point in enumerate(points):
        if matrix[point[0]][point[1]] != "-1":
            matrix[point[0]][point[1]] += f"/{values[i]}"
        else:
            matrix[point[0]][point[1]] = f'{values[i]}'

    matrix[points[0][0]][points[0][1]] = "*" + str(values[0])

    for i in range(num_lines):
        for j in range(num_columns):
            value = matrix[i][j]
            print(f'{value if value != "-1" else ".": ^4}', end='')
        print()


def limiarizar(matrix: List[List[int]], threshold_func: Callable[[int], bool]) -> List[List[int]]:
    return [
        [
            1 if threshold_func(elem) else 0
            for elem in line
        ]
        for line in matrix
    ]


def get_neighbors_labels(i: int, j: int, num_columns: int, labeled_matrix: List[List[int]]):
    """
    B   C   D
    
    A   X

    neighbors[0] = A
    neighbors[1] = B
    neighbors[2] = C
    neighbors[3] = D
    """
    neighbors = [0, 0, 0, 0]

    if j - 1 >= 0:
        neighbors[0] = labeled_matrix[i][j - 1]

    if i - 1 >= 0:
        if j - 1 >= 0:
            neighbors[1] = labeled_matrix[i - 1][j - 1]
        neighbors[2] = labeled_matrix[i - 1][j]
        if j + 1 < num_columns:
            neighbors[3] = labeled_matrix[i - 1][j + 1]

    return neighbors


def get_distinct_labels(neighbors: List[int]):
    return list(set([x for x in neighbors if x != 0]))


def rotular(matrix: List[List[int]]) -> List[List[int]]:
    num_lines = len(matrix)
    num_columns = len(matrix[0])
    num_labels = 0
    labeled_matrix = [[0 for _ in range(num_columns)] for _ in range(num_lines)]
    equivalencies: Dict[int, int] = dict()
    
    for i in range(num_lines):
        for j in range(num_columns):
            if matrix[i][j] == 1:
                labels = get_neighbors_labels(i, j, num_columns, labeled_matrix)
                distinct_labels = get_distinct_labels(labels)

                if len(distinct_labels) == 0:
                    num_labels += 1
                    labeled_matrix[i][j] = num_labels

                elif len(distinct_labels) == 1:
                    labeled_matrix[i][j] = distinct_labels[0]

                else:
                    equivalencies[max(distinct_labels)] = min(distinct_labels)
                    labeled_matrix[i][j] = min(distinct_labels)

    print("Equivalências", equivalencies)

    for key, value in equivalencies.items():
        for i in range(num_lines):
            for j in range(num_columns):
                if labeled_matrix[i][j] == key:
                    labeled_matrix[i][j] = value

    return labeled_matrix


def get_neighbors_rosenfeld(i: int, j: int) -> List[Tuple[int, int, int]]:
    """Assuming P1 = (i, j) and this distribution:

    P9  P2  P3

    P8  P1  P4

    P7  P6  P5

    return a list with [P4, P3, P2, P9, P8, P7, P6, P5]
    """
    return [
    # (ln, col, chain_code)
        (i - 0, j + 1, 0), # right
        (i - 1, j + 1, 1), # right up
        (i - 1, j - 0, 2), # up
        (i - 1, j - 1, 3), # left up
        (i - 0, j - 1, 4), # left
        (i + 1, j - 1, 5), # left down
        (i + 1, j - 0, 6), # down
        (i + 1, j + 1, 7), # right down
    ]


def inside_matrix(i: int, j: int, num_lines: int, num_columns: int):
    return i >= 0 and j >= 0 and i < num_lines and j < num_columns


def get_neighbors_inside_matrix(
    i: int,
    j: int,
    num_lines: int,
    num_columns: int
) -> List[Tuple[int, int, int]]:        
    return [
        (neighbor[0], neighbor[1])
        for neighbor in get_neighbors_rosenfeld(i, j)
        if inside_matrix(neighbor[0], neighbor[1], num_lines, num_columns)
    ]


def get_neighbors_in_region(
    i: int,
    j: int,
    num_lines: int,
    num_columns: int,
    matrix: List[List[int]],
    similarity_func: Callable[[int, int, int], bool]
) -> List[Tuple[int, int, int]]:        
    return [
        (neighbor[0], neighbor[1])
        for neighbor in get_neighbors_inside_matrix(i, j, num_lines, num_columns)
        if similarity_func(neighbor[0], neighbor[1], matrix[neighbor[0]][neighbor[1]])
    ]


def crescimento_de_regioes(matrix: List[List[int]]) -> List[List[int]]:
    num_lines = len(matrix)
    num_columns = len(matrix[0])
    region_matrix = [[0 for _ in range(num_columns)] for _ in range(num_lines)]
    seed_coords = (int(num_lines / 2), int(num_columns / 2))
    seed = matrix[seed_coords[0]][seed_coords[1]]
    processed = [seed_coords]

    while (len(processed) > 0):
        coords = processed.pop(0)
        neighbors = get_neighbors_in_region(
            *coords,
            num_lines,
            num_columns,
            matrix,
            lambda i, j, neighbor:
                region_matrix[i][j] == 0 and abs(seed - neighbor) < 3
        )
        for neighbor in neighbors:
            region_matrix[neighbor[0]][neighbor[1]] = 1
        processed.extend(neighbors)

    return region_matrix


def get_first_object_point(
        num_lines: int, num_columns: int, matrix: List[List[int]]) -> Tuple[int, int]:
    point = None

    for i in range(num_lines):
        for j in range(num_columns):
            if matrix[i][j] != 0:
                point = (i, j)
                break
        else: # else will execute if the intern for doesn't break
            continue
        break

    return point


def rosenfeld(matrix: List[List[int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Discover the outline of the thresholded object and it's chain code
    """
    num_lines = len(matrix)
    num_columns = len(matrix[0])
    first_point = get_first_object_point(num_lines, num_columns, matrix)
    points = [first_point]
    chain_code = []
    p = first_point
    q_index = 4  # This is the chain code from P to Q in the start of the algorithm
    q = (p[0] - 0, p[1] - 1, q_index)
    neighbors = get_neighbors_rosenfeld(*p)

    while True:
        for _ in range(7):
            q_index = (q_index + 1) % 8
            new_q = neighbors[q_index]

            if (not inside_matrix(new_q[0], new_q[1], num_lines, num_columns) or
                    matrix[new_q[0]][new_q[1]] == 0):
                q = new_q

            else:
                p = (new_q[0], new_q[1])
                points.append(p)
                chain_code.append(new_q[2])
                break
        
        if p == first_point:
            break

        neighbors = get_neighbors_rosenfeld(*p)
        q_index = [(neighbor[0], neighbor[1]) for neighbor in neighbors].index((q[0], q[1]))

    return chain_code, points


def derivate_chain_code(chain_code: List[int]):
    return [
        (chain_code[i] - chain_code[(i - 1) % len(chain_code)]) % 8
        for i, code in enumerate(chain_code)
    ]


def get_form_number(chain_code_derivated: List[int]) -> List[int]:
    chain_code_derivated = [str(code) for code in chain_code_derivated]
    form_number = chain_code_derivated
    smallest_number = int(''.join(chain_code_derivated))

    for i in range(len(chain_code_derivated) - 1):
        num_shifts = i + 1
        curr_form_number = (
            chain_code_derivated[-num_shifts:] + chain_code_derivated[:-num_shifts]
        )
        curr_number = int(''.join(curr_form_number))
        if curr_number < smallest_number:
            smallest_number = curr_number
            form_number = curr_form_number

    return [int(code) for code in form_number]


def any_point_clear(
    points: List[Tuple[int, int]],
    num_lines: int,
    num_columns: int,
    matrix: List[List[int]]
) -> bool:
    """Check if any point is 0
    """
    return reduce(lambda accum, point: accum or (
        not inside_matrix(point[0], point[1], num_lines, num_columns) or
        matrix[point[0]][point[1]] == 0
    ), points, False)


def any_right_clear(
    i: int, j: int, num_lines: int, num_columns: int, matrix: List[List[int]]
):
    return any_point_clear(
        [(i - 1, j - 0), (i - 0, j + 1), (i + 1, j - 0)],
        num_lines, num_columns, matrix
    )


def any_down_clear(
    i: int, j: int, num_lines: int, num_columns: int, matrix: List[List[int]]
):
    return any_point_clear(
        [(i - 0, j + 1), (i + 1, j - 0), (i - 0, j - 1)],
        num_lines, num_columns, matrix
    )


def any_left_clear(
    i: int, j: int, num_lines: int, num_columns: int, matrix: List[List[int]]
):
    return any_point_clear(
        [(i - 1, j - 0), (i - 0, j - 1), (i + 1, j - 0)],
        num_lines, num_columns, matrix
    )


def any_up_clear(
    i: int, j: int, num_lines: int, num_columns: int, matrix: List[List[int]]
):
    return any_point_clear(
        [(i - 0, j - 1), (i - 1, j - 0), (i - 0, j + 1)],
        num_lines, num_columns, matrix
    )


def mark_points(
    num_lines: int,
    num_columns: int,
    matrix: List[List[int]],
    neighbor_direction1_clear: Callable[[int, int, int, int, List[List[int]]], bool],
    neighbor_direction2_clear: Callable[[int, int, int, int, List[List[int]]], bool]
) -> List[Tuple[int, int]]:
    points = []

    for i in range(num_lines):
        for j in range(num_columns):
            if matrix[i][j] == 1:
                neighbors = get_neighbors_inside_matrix(i, j, num_lines, num_columns)

                num_shifts = 0
                while ( # rotate neighbors until first neighbor is 0
                    matrix[neighbors[0][0]][neighbors[0][1]] == 1 and
                    num_shifts != len(neighbors)
                ):
                    neighbors = neighbors[-1:] + neighbors[:-1] # circular shift 1 unit
                    num_shifts += 1
                
                if num_shifts == len(neighbors): continue

                num_neighbors = 0
                inside_object = False
                num_times_inside_object = 0
                for neighbor in neighbors:
                    if matrix[neighbor[0]][neighbor[1]] == 1:
                        num_neighbors += 1
                        if not inside_object:
                            inside_object = True
                            num_times_inside_object += 1
                    else:
                        inside_object = False

                if (
                    2 <= num_neighbors <= 6 and
                    num_times_inside_object == 1 and
                    neighbor_direction1_clear(i, j, num_lines, num_columns, matrix) and
                    neighbor_direction2_clear(i, j, num_lines, num_columns, matrix)
                ):
                    points.append((i, j))

    return points


def extract_skelet(matrix: List[List[int]]):
    num_lines = len(matrix)
    num_columns = len(matrix[0])
    skelet = [[elem for elem in line] for line in matrix]

    while True:
        marked_points1 = mark_points(
            num_lines, num_columns, skelet, any_right_clear, any_down_clear)

        for point in marked_points1:
            skelet[point[0]][point[1]] = 0

        marked_points2 = mark_points(
            num_lines, num_columns, skelet, any_left_clear, any_up_clear)

        for point in marked_points2:
            skelet[point[0]][point[1]] = 0

        if len(marked_points1) == 0 and len(marked_points2) == 0: break

    return skelet


def question_4(
    matrix: List[List[int]],
    matrix_name: str,
    threshold_func: Callable[[int], bool],
    threshold_cond: str
):
    """Solve all alternatives using the given matrix
    """
    print(f"Matriz {matrix_name}")
    print_matrix(matrix)
    print()

    print(f"Matriz {matrix_name} Limiarizada ({threshold_cond})")
    limiarizada = limiarizar(matrix, threshold_func)
    print_matrix(limiarizada)
    print()

    print(f"Matriz {matrix_name} Rotulada")
    print_matrix(rotular(limiarizada))
    print()

    print(f"Matriz {matrix_name} Crescimento por Região com semente no meio")
    matrix_crescimento_regiao = crescimento_de_regioes(matrix)
    print_matrix(matrix_crescimento_regiao)
    print()

    print(f"Matriz {matrix_name} código de cadeia e pontos do contorno")
    chain_code, chain_points = rosenfeld(limiarizada)
    print('código de cadeia', chain_code)
    print()

    print('contorno:')
    print('obs.: * é o começo')
    print('obs.: nos pontos onde se passa duas vezes, "cod1/cod2" indica cod1 na ida e cod2 na volta')
    print_matrix_with_points(len(matrix), len(matrix[0]), chain_points[:-1], chain_code)
    print()

    perimeter = len(chain_code)
    area = len(chain_points)
    print('circularidade:', (perimeter * perimeter) / (4 * math.pi * area))
    chain_code_derivated = derivate_chain_code(chain_code)
    print('derivada do código de cadeia:', chain_code_derivated)
    print('número de forma:', get_form_number(chain_code_derivated))
    print()

    print(f"Matriz {matrix_name} esqueleto")
    print_matrix(extract_skelet(matrix_crescimento_regiao), ".")
    print()

    print("------------------")
    print()


A: List[List[int]] = [
    [3, 5, 2, 1, 1],
    [1, 4, 6, 2, 1],
    [1, 1, 5, 6, 2],
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
]

B: List[List[int]] = [
    [5, 1, 2, 1, 8],
    [6, 6, 5, 6, 1],
    [2, 1, 8, 7, 7],
    [6, 1, 2, 8, 8],
    [7, 8, 2, 1, 1],
]

C: List[List[int]] = [
    [1, 1, 9, 1, 1],
    [1, 1, 9, 8, 7],
    [9, 9, 9, 2, 1],
    [1, 1, 2, 8, 8],
    [1, 2, 2, 8, 9],
]

question_4(A, "A", lambda elem: elem > 2, "elem > 2")
question_4(B, "B", lambda elem: elem > 4, "elem > 4")
question_4(C, "C", lambda elem: elem > 6, "elem > 6")
