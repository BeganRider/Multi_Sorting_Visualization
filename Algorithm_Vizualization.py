import numpy as np
import pandas as pd
import pygame
import pygame as pg

pygame.init()
class VizSettings:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    BACKGROUND_COLOR = (0, 0, 0)

    FONT = pygame.font.SysFont('Arial', 50)

    SIDE_PAD = 100
    TOP_PAD = 100

    def __init__(self,width,height,lst):
        self.width = width
        self.height = height

        self.window = pygame.display.set_mode((width,height))
        pygame.display.set_caption('Algorithm Visualization')
        self.set_list(lst)

    def set_list(self,lst):
        self.lst = lst
        self.min_val = min(lst)
        self.max_val = max(lst)

        self.block_width = round((self.width - self.SIDE_PAD))
        self.block_height = round((self.height - self.TOP_PAD)/(self.max_val-self.min_val)
        self.start_x = self.SIDE_PAD // 2

def draw(VizSettings,algorithm_name,ascending):
    VizSettings.window.fill(VizSettings.BACKGROUND_COLOR)

    title = VizSettings.FONT.render(f"{algorithm_name}", True, VizSettings.BLACK
    VizSettings.window.blit(title, (VizSettings.width/2 - controls.get_width))