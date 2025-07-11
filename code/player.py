import pygame
from pygame.math import Vector2
import numpy as np
import math

class Player:

    def __init__(self, x, y, screen, color, cell_size, key_mapping, team = "RED" or "BLUE"):
        
        self.position = Vector2(x, y) #posición en cada frame
        self.old_position = Vector2(x,y) #posicion anterior
        self.direction = Vector2(0, 0) #vector dirección al que apunta

        self.screen = screen
        self.color = color

        self.isAlive = True
        self.trailEnabled = True
        self.has_moved = False

        self.cell_size = cell_size
        self.size = self.cell_size // 2

        self.team = team

        self.move_delay = 150 #ms entre movimientos
        self.time_since_move = 0 #cuenta el tiempo desde que se mueve

        self.key_mapping = key_mapping


    def getTrailEstate(self):
        return self.trailEnabled
    
    def isDead(self):
        if not self.isAlive:
            print(f"Jugador {self.color} ha muerto")
            

    def move(self, dt):
        if not self.isAlive:
            return  # No se mueve si está muerto

        self.time_since_move += dt

        if self.time_since_move >= self.move_delay:
            self.time_since_move = 0
            self.old_position = self.position.copy() #se guarda la posicion antes de moverse para luego pasarla al trazo de luz
            self.position += self.direction    #actualiza la posición
            self.has_moved = True
        else:
            self.has_moved = False
           

    def change_direction(self, new_dir):
        if not self.isAlive:
            return  # No cambia de dirección si está muerto
        # Cambia de dirección si no es la contraria
        if new_dir.x != -self.direction.x and new_dir.y != -self.direction.y:
            self.direction = new_dir

    def draw_player(self):
        if not self.isAlive:
            return  # No dibuja si está muerto
        #drawing now by squares and not pixels. Although, to calculate where is it drawn it must be focused on pixels 
        pixel_x = int(self.position.x * self.cell_size + self.cell_size // 2) #puts the center of the player on the center of the map square
        pixel_y = int(self.position.y * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, self.color, (pixel_x, pixel_y), self.size - 2)


    def get_cone_vision(self, grid_cols, grid_rows, obstacles, fov_deg=90, max_distance=6, num_rays=15):
        """
        obstacles: matriz 2d simple, 0=libre, 1=muro, 2=estela, 3=jugador
        Devuelve un set de (x, y) visibles en el cono.
        """
        vision = set()
        if self.direction.length() == 0:
            return vision

        angle_center = math.atan2(self.direction.y, self.direction.x)
        half_fov = math.radians(fov_deg / 2)
        angles = np.linspace(angle_center - half_fov, angle_center + half_fov, num_rays)  #esto genera los ángulos de los rayos en el cono de visión uwu

        # Precalcula los desplazamientos de cada rayo para todas las distancias
        ds = np.arange(1, max_distance + 1)
        cosines = np.cos(angles)
        sines = np.sin(angles)

        for i in range(num_rays):
            dx = cosines[i]
            dy = sines[i]
            # Vectoriza los puntos del rayo
            rx = np.round(self.position.x + dx * ds).astype(int)
            ry = np.round(self.position.y + dy * ds).astype(int)
            for x, y in zip(rx, ry):
                if 0 <= x < grid_cols and 0 <= y < grid_rows:
                    vision.add((x, y))
                    if obstacles[y, x] == 1 or obstacles[y, x] == 3:  # muro o jugador: oclusión
                        break
                    # Si es estela (2), sigue el rayo
                else:
                    break
        return vision # set de coordenadas (x, y) VISIBLES en el cono de visión pero sin iformación de si es estela, muro o jugador


