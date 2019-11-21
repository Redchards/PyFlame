import numpy.random as random
import numpy as np
from math import floor


def generate_random_bi_unit_point():
    return 2 * (random.random(2) - 1/2)


class SoftwareFlameSampler:
    def __init__(self, dimensions, flame_system, preparation_rounds=100):
        self.preparation_rounds = preparation_rounds
        self.dimensions = dimensions
        self.histogram = self.generate_histogram()
        self.color_hist = self.generate_color_histogram()
        self.global_iteration = 0
        self.flame_system = flame_system
        self.current_point = generate_random_bi_unit_point()

    def generate_histogram(self):
        return np.zeros((self.dimensions[1], self.dimensions[0]))

    def generate_color_histogram(self):
        return np.zeros((self.dimensions[1], self.dimensions[0], 4))

    def to_windows_coordinates(self, x, y):
        return floor((x + 1) / 2 * (self.dimensions[1]) - 1), floor((y + 1) / 2 * (self.dimensions[0]) - 1)

    def reset(self):
        self.histogram = self.generate_histogram()
        self.color_hist = self.generate_color_histogram()
        self.global_iteration = 0
        self.current_point = generate_random_bi_unit_point()

    def step(self):
        self.global_iteration += 1
        fi, c = self.flame_system.pick()
        x, y = fi(self.current_point[0], self.current_point[1])
        self.current_point = (x, y)

        if self.global_iteration > self.preparation_rounds:
            # TODO : This can't stay like this, we need a new construct "FinalTransform" that will always return two ints
            x, y = self.to_windows_coordinates(*self.flame_system.final_transform(x, y))
            x, y = int(x), int(y)
            if self.dimensions[1] > x >= 0 and self.dimensions[0] > y >= 0:
                self.histogram[x, y] += 1
                self.color_hist[x, y] = (self.color_hist[x, y] + c) / 2

