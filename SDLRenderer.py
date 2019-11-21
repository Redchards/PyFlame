from sdl2 import *
import time
import ctypes
import numpy as np


# TODO : Put a switch for log rendering
def histogram_to_pixelmap(histogram, colors):
    x = histogram.flatten()#np.clip(histogram.flatten(), 0, 255)
    x2 = colors.flatten()
    # x = (np.repeat(x, 4) * 255).astype(np.int8)
    x = (255 * np.power((x2 / 255) * np.log(np.repeat(x + 1, 4)) / np.log(x.max() + 1), 1/2.2)).astype(np.int8)
    return x # (np.ones(self.dimensions[0] * self.dimensions[1] * 4) * 150).astype(np.int8)


class SDLRenderer:
    def __init__(self, sampler, dimensions):
        self.sampler = sampler
        self.dimensions = dimensions
        self.running = False

        SDL_Init(SDL_INIT_VIDEO)
        self.window = SDL_CreateWindow(b'Flame Renderer', SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       self.dimensions[0], self.dimensions[1], SDL_WINDOW_OPENGL)

        assert not self.window is None, 'Couldn\'t create the render window!'

        self.internal_renderer = SDL_CreateRenderer(self.window, -1, SDL_RENDERER_ACCELERATED)

        self.renderer_info = SDL_RendererInfo()
        SDL_GetRendererInfo(self.internal_renderer, ctypes.byref(self.renderer_info))

        self.canvas = None

    def generate_canvas(self):
        return SDL_CreateTexture(self.internal_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                 self.dimensions[0], self.dimensions[1])

    def start(self):
        print(f'Starting rendering with {self.renderer_info.name}')
        print('Available texture formats:')
        for i in range(self.renderer_info.num_texture_formats):
            print(SDL_GetPixelFormatName(self.renderer_info.texture_formats[i]))

        self.running = True
        itt = 3000

        if self.canvas is None:
            self.canvas = self.generate_canvas()

        SDL_SetRenderDrawColor(self.internal_renderer, 0, 0, 0, SDL_ALPHA_OPAQUE)

        while self.running:
            SDL_RenderClear(self.internal_renderer)
            evt = SDL_Event()

            while SDL_PollEvent(ctypes.byref(evt)):
                if evt.type == SDL_QUIT:
                    self.stop()
                    break

            self.sampler.step()
            itt -= 1
            if itt <= 0:
                itt = 10000
                pixels = histogram_to_pixelmap(self.sampler.histogram, self.sampler.color_hist)
                SDL_UpdateTexture(self.canvas, None, pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                                  self.dimensions[0] * 4)
                SDL_RenderCopy(self.internal_renderer, self.canvas, None, None)
                SDL_RenderPresent(self.internal_renderer)

    def pause(self):
        self.running = False

    def stop(self):
        self.pause()
        self.sampler.reset()
        self.canvas = None
