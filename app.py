import sys
import sdl2.ext
from SDLRenderer import SDLRenderer
from FlameSampler import SoftwareFlameSampler
from MathTools import *
import numpy as np


# TODO : Clean up the code and put everything where it belongs
# TODO : Provide screen transform in the FlameSystem in lieu of the final transform, currently it doesn't look good

def l2_norm(v):
    return np.sqrt(np.sum(np.square(v)))

def get_angle(v):
    return np.arctan2(v[0], v[1])

# TODO : Use better functions for variants (i.e : don't recompute the L2-norm all the time)
v1 = Variant2D(lambda _, v: np.sin(v))
v2 = Variant2D(lambda _, v: 1 / l2_norm(v)**2 * v)
v3 = Variant2D(lambda _, v: (v[0] * np.sin(l2_norm(v)**2) - v[1] * np.cos(l2_norm(v)**2), v[0] * np.sin(l2_norm(v)**2) + v[1] * np.cos(l2_norm(v) ** 2)))
v4 = Variant2D(lambda _, v: 1 / l2_norm(v) * np.array([(v[0] - v[1]) * (v[0] + v[1]), 2 * v[0] * v[1]]))
v5 = Variant2D(lambda _, v: (get_angle(v) / np.pi, l2_norm(v) - 1))
v6 = Variant2D(lambda _, v: l2_norm(v) * np.array([np.sin(get_angle(v) + l2_norm(v)), np.cos(get_angle(v) - l2_norm(v))]))
v7 = Variant2D(lambda _, v: l2_norm(v) * np.array([np.sin(l2_norm(v) * get_angle(v)), -np.cos(l2_norm(v) * get_angle(v))]))
v8 = Variant2D(lambda _, v: get_angle(v) / np.pi * np.array([np.sin(np.pi * get_angle(v)), np.cos(np.pi * get_angle(v))]))
v9 = Variant2D(lambda _, v: 1 / l2_norm(v) * np.array([np.cos(get_angle(v)) + np.sin(l2_norm(v)), np.sin(get_angle(v)) - np.cos(l2_norm(v))]))

def spierpinsky():
    f0 = LinearFunction2D(np.array([[1 / 2, 0, 0], [0, 1 / 2, 0]]))
    f1 = LinearFunction2D(np.array([[1 / 2, 0, 1 / 2], [0, 1 / 2, 0]]))
    f2 = LinearFunction2D(np.array([[1 / 2, 0, 0], [0, 1 / 2, 1 / 2]]))

    return LinearFunctionSystem([f0, f1, f2])


def fern():
    f0 = LinearFunction2D(np.array([[0, 0, 0], [0, 0.16, 0]]))
    f1 = LinearFunction2D(np.array([[0.85, 0.04, 0], [-0.04, 0.85, 1.6]]))
    f2 = LinearFunction2D(np.array([[0.2, -0.26, 0], [0.23, 0.22, 1.6]]))
    f3 = LinearFunction2D(np.array([[-0.15, 0.28, 0], [0.26, 0.24, 0.44]]))

    return LinearFunctionSystem([f0, f1, f2, f3], weights=[0.02, 0.84, 0.07, 0.07])


if __name__ == '__main__':
    resources = sdl2.ext.Resources(__file__, 'resources')

    '''sdl2.ext.init()
    window = sdl2.ext.Window('Hello World!', size=(480, 480))
    window.show()

    processor = sdl2.ext.TestEventProcessor()
    processor.run(window)'''
    #ffs = FlameFunctionSystem(fern(), variant_list=[v7, v2])
    #ffs = FlameFunctionSystem(spierpinsky(), variant_list=[v4, v8])
    ffs = FlameFunctionSystem(spierpinsky(), variant_list=[v9, v5, v4, v1, v7], final_transform=LinearFunction2D(np.array([[1.5, 0, -0.3], [0, 1.5, 0.5]]), return_self=False),\
                                weights=[0.2, 0.2, 0.3, 0.1, 0.2], colors=[(0, 32, 139, 134), (0, 41, 62, 72), (130, 231, 45, 23)])
    #ffs = FlameFunctionSystem(spierpinsky(), variant_list=[v4, v8], weights=[0.8, 0.2], colors=[(255, 0, 0, 255), (255, 189, 133, 0), (255, 45, 132, 128)])
    #ffs = FlameFunctionSystem(fern(), final_transform=LinearFunction2D(np.array([[1/3, 0, 1/9], [0, 1/5, -1]]), return_self=False))
    dimensions = (1920, 920)
    sampler = SoftwareFlameSampler(dimensions=dimensions, flame_system=ffs)
    renderer = SDLRenderer(sampler=sampler, dimensions=dimensions)
    renderer.start()
