import pygame
import time
from player import Player

class LightTrail:
    def __init__(self, player, color):

        self.player = player
        self.color = pygame.Color(color)
        self.lightPoints = [] #guarda los trazos de luz (posiciones y tiempo)
        self.lightCords = set() #Guarda solo las coordenadas del trazo para comparaciones rapidas en collitions
        self.duration = 10 #numero de segundos que tarda en desaparecer  
        
        
    def updateTrail(self, current_time):
        if not self.player.isAlive: #Si el jugador muere se elimina su estela
            self.lightPoints.clear()
            self.lightCords.clear()
            return
        #Filtra los trazos viejos aunque el trazo se haya apagado
        self.lightPoints = [
                    (x,y,t) for (x,y,t) in self.lightPoints if current_time - t <self.duration
        ]

        self.lightCords = set((x, y) for (x, y, t) in self.lightPoints)    #Filtra las cords usando lightPoints
        
        if self.player.isAlive and self.player.getTrailEstate():  #Solo guardar la posición si la estela está encendida y el jugador esta vivo
                
                pos = (int(self.player.old_position.x), int(self.player.old_position.y))    #Separa la posicion para el trazo
                timed_pos = (pos[0], pos[1], current_time)                                  #Del tiempo en el que se genera
                self.lightPoints.append(timed_pos)                                          #Guarda cords y tiempo para graficarlas
                self.lightCords.add(pos)                                                    #Guarda solo las cords del trazo


    def drawTrail(self, screen, cell_size, current_time):

        for x, y, t in self.lightPoints:
            transparency = max(0, 255 - int(255 * ((current_time - t) / self.duration)))   #Hace que el bloque se vea más "difuso" en vez de desaparecer de golpe con el tiempo
            faded_color = pygame.Color(self.color.r, self.color.g, self.color.b, self.color.a)
            faded_color.a = transparency

            s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)    #se usa para dibujar el trazo desvaneciendose
            pygame.draw.rect(s, faded_color, (0, 0, cell_size, cell_size))
            screen.blit(s, (x * cell_size, y * cell_size))
        