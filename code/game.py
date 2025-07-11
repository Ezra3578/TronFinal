import pygame
from player import Player
from LightTrail import LightTrail
import numpy as np
import random

class TronGame:
    def __init__(self):

        self.simulated_time = 0

        self.killed_by = {}

        #diccionario de teclas para cada jugador
        self.mapping_player1 = {
            'left': pygame.K_a,
            'right': pygame.K_d,
            'up': pygame.K_w,
            'down': pygame.K_s,
            'toggle': pygame.K_LSHIFT 

        }

        self.mapping_player3 = {
            'left': pygame.K_f,
            'right': pygame.K_h,
            'up': pygame.K_t,
            'down': pygame.K_g,
            'toggle': pygame.K_RALT
        }

        self.mapping_player2 = {
            'left': pygame.K_LEFT,
            'right': pygame.K_RIGHT,
            'up': pygame.K_UP,
            'down': pygame.K_DOWN,
            'toggle': pygame.K_RSHIFT
        }

        self.mapping_player4 = {
            'left': pygame.K_j,
            'right': pygame.K_l,
            'up': pygame.K_i,
            'down': pygame.K_k,
            'toggle': pygame.K_RCTRL
        }

        

        self.mapping_player4 = {
            'left': pygame.K_j,
            'right': pygame.K_l,
            'up': pygame.K_i,
            'down': pygame.K_k,
            'toggle': pygame.K_RCTRL
        }

        self.width = 1400
        self.height = 840

        self.screen = None
        
        self.cell_size = 40 #la casilla es de 40x40 pixeles, es decir, hay 35 casillas en horizontal y 20 en vertical
        self.grid_cols = self.width // self.cell_size  #cambiado a columnas
        self.grid_rows = self.height // self.cell_size #cambiado a filas

        self.borders = []
        for col in range(self.grid_cols):
            self.borders.append((col,0))        #borde superior
            self.borders.append((col, self.grid_rows-1))  #borde inferior
        for row in range(self.grid_rows-1):
            self.borders.append((0,row+1))
            self.borders.append((self.grid_cols-1, row+1))

        ######generación de mapas
        self.borders = set(self.borders)  #convertir a set 
        
        self.other_maps = random.randint(1, 4) #elige un mapa al azar entre 1 y 4
        if self.other_maps == 1:
            self.borders.update((col, 5) for col in range(5, 14)) 
            self.borders.update((col, 15) for col in range(21, 30))
            self.borders.update((17, row) for row in range(4, 17))


        if self.other_maps == 2:
            self.borders.update((col, row) for col in range(3, 13) for row in (4, 12))
            self.borders.update((col, row) for col in range(22, 32) for row in (8, 16))
            self.borders.update((22, row) for row in range(4, 8))
            self.borders.update((12, row) for row in range(13, 17))
            self.borders.update((17, row) for row in range(8, 13))


        if self.other_maps == 3:
            self.borders.update((col, 16) for col in range(3,10))
            self.borders.update((col, 10) for col in range(9,27))
            self.borders.update((col, 4) for col in range (25,32))
            self.borders.update((17, row) for row in range (4,7))
            self.borders.update((17, row) for row in range (14,17))
            
        if self.other_maps == 4:

            self.borders.update((col, row) for col in range (9,14) for row in (6,14))
            self.borders.update((col, row) for col in range (21,26) for row in (6,14))
            self.borders.update((col, row) for row in range(4, 6) for col in (11,17))
            self.borders.update((col, row) for row in range(15, 17) for col in (17, 23))
            self.borders.update([(17,6),(17,7),(17,13),(17,14),(13,10),(21,10)])
        

        self.clock = pygame.time.Clock()
        self.running = True
        self.render = False
        self.players = []  #lista de jugadores
        self.trails = [] # lista de trazos de luz
       


    def check_collitions(self):

        self.killed_by.clear()

        for i, player in enumerate(self.players):
            if not player.isAlive:  #Si el jugador esta muerto no compara sus colisiones
                continue

            player_pos = (int(player.position.x), int(player.position.y))       #recorre los jugadores y guarda sus posiciones

            if not player.has_moved:        #si el jugador ya se movio detecta colisiones (evitar colision en el frame inicial)
                continue

            for trail in self.trails:   #recorre las listas de trazos de luz
                if player_pos in trail.lightCords:    #si las posiciones coinciden hay colision
                    
                    #Aqui se identifica quien murio y con que estela
                    if trail.player == player:      #Suicidio
                        self.killed_by[player] = None
                    else:   #muerte por estela enemiga
                        self.killed_by[player] = trail.player    
                    
                    player.isAlive = False 
                    break  # Detenemos después de la primera colisión
            
            for (x,y) in self.borders:
                if (x,y) == player_pos:
                    self.killed_by[player] = None
                    player.isAlive = False
                    break
        
    def draw_borders(self):
        for (x, y) in self.borders:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            if self.render:
                pygame.draw.rect(self.screen, (200,200,200), rect)
    
    def update_state(self, dt):

        self.simulated_time += dt/1000.0

        for trail in self.trails:
            trail.updateTrail(self.simulated_time)

        # Actualizamos el movimiento de cada jugador con su propias teclas
        for player in self.players:
            player.move(dt)
        
        self.check_collitions() #comprueba colisiones
        
    
    def draw(self):
        self.screen.fill((0,0,0))
        self.draw_borders()

        for trail in self.trails:
            trail.drawTrail(self.screen, self.cell_size, self.simulated_time)
        for player in self.players:
            player.draw_player()

    def setScreen(self, screen):
        self.screen = screen
        
   


