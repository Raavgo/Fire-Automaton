import pygame
import numpy as np
from ArrayWorker import cuda_worker
from Utility import validateText

pygame.init()
size = (1024, 1024)
screen = pygame.display.set_mode(size)
pixel = pygame.display.get_window_size()
screen_array = np.zeros(size, dtype=np.uint32)
my_surface = pygame.surfarray.make_surface(screen_array)

input_rect = pygame.Rect(size[0]-180, 0, 180, 56)
font = pygame.font.Font('freesansbold.ttf', 14)
text_g = font.render('p_growth', True, 0x0, None)
text_f = font.render('p_fire', True, 0x0, None)
g_input = pygame.Rect(size[0]-105, 5, 100, 16)
f_input = pygame.Rect(size[0]-105, 30, 100, 16)

i = 0
pos = (0, 0)
override = False
g_active = False
f_active = False

g_color = 0x0
f_color = 0x0

color_active = 0xf9f8dd
color_passive = 0xFFFFFF
p_growth_text = "0.001"
p_fire_text = "0.0001"


while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            exit()

        if event.type == pygame.MOUSEBUTTONDOWN:

            if g_input.collidepoint(event.pos):
                g_active = True
                f_active = False
            elif f_input.collidepoint(event.pos):
                f_active = True
                g_active = False
            else:
                g_active = False
                f_active = False
                override = True

        if event.type == pygame.MOUSEBUTTONUP:
            override = False

        if event.type == pygame.KEYDOWN:
            # Check for backspace
            if event.key == pygame.K_BACKSPACE:
                # get text input from 0 to -1 i.e. end.
                if g_active:
                    p_growth_text = p_growth_text[:-1]
                elif f_active:
                    p_fire_text = p_fire_text[:-1]

            # Unicode standard is used for string
            # formation
            else:
                if g_active:
                    p_growth_text += event.unicode
                elif f_active:
                    p_fire_text += event.unicode

        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    g_color = color_active if g_active else color_passive
    f_color = color_active if f_active else color_passive

    p_growth = validateText(p_growth_text)
    p_fire = validateText(p_fire_text)

    screen_array = cuda_worker(screen_array, p_growth, p_fire)

    if override:
        pos = pygame.mouse.get_pos()
        screen_array[pos[0] - 1: pos[0] + 2, pos[1] - 1: pos[1] + 2] = 0xFF0000

    text_surface_growth = font.render(p_growth_text, True, 0x0)
    text_surface_fire = font.render(p_fire_text, True, 0x0)

    pygame.surfarray.blit_array(screen, screen_array)
    pygame.draw.rect(screen, 0xd7d7d7, input_rect)
    screen.blit(text_g, (size[0] - 175, 5))
    screen.blit(text_f, (size[0] - 175, 30))
    pygame.draw.rect(screen, g_color, g_input)
    pygame.draw.rect(screen, f_color, f_input)
    screen.blit(text_surface_growth, (g_input.x + 5, g_input.y+1))
    screen.blit(text_surface_fire, (f_input.x + 5, f_input.y+1))
    pygame.display.update()



