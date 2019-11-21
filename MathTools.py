import numpy as np
import numpy.random as random
import functools


class LinearFunction2D:
    def __init__(self, coefficients, return_self=True):
        assert coefficients.shape == (2, 3)
        self.coefficients = coefficients
        self.return_self = return_self

    def __call__(self, x, y):
        res = np.matmul(self.coefficients, np.array([x, y, 1]))

        if self.return_self:
            return self, res
        else:
            return res

    def __repr__(self):
        return str(self.coefficients)


def identity_function2D():
    return LinearFunction2D(np.ones((2, 3)))


class LinearFunctionSystem:
    def __init__(self, function_list, weights=None):
        assert len(function_list) > 0
        assert weights is None or len(weights) == len(function_list)

        if weights is None:
            self.weights = [1 / len(function_list) for _ in range(len(function_list))]
        else:
            self.weights = weights

        self.function_list = list(function_list)

    def __getitem__(self, idx):
        assert idx < len(self.function_list)
        return self.function_list[idx]

    def pick_idx(self):
        i = random.choice(list(range(len(self.function_list))), p=self.weights)
        return i

    def pick(self):
        return self[self.pick_idx()]

    def __len__(self):
        return len(self.function_list)

    def __iter__(self):
        return self

    def __next__(self):
        for f in self.function_list:
            yield f


class Variant2D:
    def __init__(self, t):
        self.t = t

    def __call__(self, linfunc_app):
        assert len(linfunc_app) == 2, 'The linear function must have the option \'return_self\' set to True!'
        f, v = linfunc_app
        assert len(v) == 2, f'A Variant2D does not accept inputs with shape {v.shape}!'

        return np.array(self.t(f, v))


class FlameFunctionSystem:
    def __init__(self, linear_system, variant_list=None, post_transform=identity_function2D(), weights=None, colors=None, final_transform=None):
        assert weights is None or len(weights) == 0 or len(weights) == len(variant_list)

        if variant_list is None or len(variant_list) == 0:
            self.variant_list = [Variant2D(lambda _, v: v)]
        else:
            self.variant_list = variant_list

        if weights is None or len(weights) == 0:
            self.weights = [1 / len(self.variant_list) for _ in range(len(self.variant_list))]
        else:
            assert sum(weights) == 1
            self.weights = weights

        assert colors is None or len(colors) <= len(linear_system)
        if colors is None:
            self.colors = [(255, 255, 255, 255) for _ in range(len(linear_system))]
        elif len(colors) < len(linear_system):
            self.colors = colors + [(255, 255, 255, 255) for _ in range(len(linear_system) - len(colors))]
        else:
            self.colors = colors
        self.colors = [np.array(c) for c in self.colors]

        if final_transform is None:
            self.final_transform = lambda x, y: np.array([x, y])
        else:
            assert final_transform.return_self is False, 'The final transform shouldn\'t return self'
            self.final_transform = final_transform

        self.linear_system = linear_system
        self.post_transform = post_transform

        self.function_list = []

        for f in self.linear_system.function_list:
            print(f.coefficients)
            self.function_list.append(lambda x, y, f_l=f: sum(
                [w * variant(f_l(x, y)) for w, variant in zip(self.weights, self.variant_list)]))

    def __getitem__(self, idx):
        res = 0
        return self.function_list[idx], self.colors[idx]

    def pick(self):
        return self[self.pick_idx()]

    def pick_idx(self):
        return self.linear_system.pick_idx()

    def __len__(self):
        return len(self.linear_system)
