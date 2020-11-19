from typing import List

import sys
import math

def fourier_cos_sin(imagem: List[int]):
    print('discrete fourier transform for image', imagem)
    print()
    # here, imagem[x] is equal to f(x)
    n = len(imagem)
    real_parts = [0.0] * n
    imaginary_parts = [0.0] * n

    for x in range(n):
        print(f'F({x}) = (1/{n})Σ_x( f(x)[ cos(2π*{x}*x/{n}) - i sen(2π*{x}*x/{n}) ] )')

        for i in range(n):
            print(f'    {imagem[i]}[ cos(2π*{x}*{i}/{n}) - i sen(2π*{x}*{i}/{n}) ] ) +')
            real_parts[x] += imagem[i] * math.cos(2 * math.pi * x * i / n)
            imaginary_parts[x] += imagem[i] * -math.sin(2 * math.pi * x * i / n)

        real_parts[x] = real_parts[x] / n
        imaginary_parts[x] = round(imaginary_parts[x] / n, 3)
        print()

    print('real_parts', real_parts)
    print('imaginary_parts', imaginary_parts)

if len(sys.argv) > 1:
    imagem = [float(value) for value in sys.argv[1:]]
else:
    imagem = [2+0j, 3+0j, 4+0j, 4+0j]

fourier_cos_sin(imagem)
