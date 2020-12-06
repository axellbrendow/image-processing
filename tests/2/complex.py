import numpy
import cmath


def DFT(arr, norm=False):
    d = numpy.fft.fft(arr)
    if(norm):
        d = d/len(arr)
    return d


def IFT(arr, norm=False):
    d = numpy.fft.ifft(arr)
    if(norm):
        d = d*len(arr)
    return d


def IFTreal(arr, norm=False):
    return numpy.real(IFT(arr, norm))


def printnp(arr):
    print("-----------------")
    for i in range(len(arr)):
        print("\n"+"{:.2f}".format(arr[i]).replace('j', 'i'))
    print("-----------------")


max = 30
Z = -100

ct = [6.25+0j, -0.75+1.5j, Z, -0.75-1.5j]  # <-- altere, mantenha o Z

for a in range(max):
    for b in range(max):
        for d in range(max):
            # <-- altere o lugar e valor de 9. no meu caso a coordenada X=3 deve ser 9
            f = [a, b, d, 9]
            c = DFT(f, norm=True)
            equal = True
            for i in range(len(c)):
                if(i == 2):  # <-- altere de acordo com a coordenada X pedida. no meu caso a coordenada eh X=2, entao "i == 2"
                    continue
                if(not cmath.isclose(c[i], ct[i], rel_tol=1e-5)):
                    equal = False
                    break
            if(equal):
                printnp(c)
                printnp(IFTreal(c, norm=True))
