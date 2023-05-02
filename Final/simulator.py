"""
Programmer: Gaddiel Morales
Course: COSC 527
Purpose: Simulate a car on a race track using a vector of input numbers

"""
import pandas as pd
import math
import numpy as np

class World():
    """
    Simulation environment for cars

    map tiles: 
    0 - out of bounds
    1 - left wall
    2 - right wall
    3 - checkpoint
    4 - road
    5 - start
    """

    #simulator variables
    default_map = "maps/circle.map"
    world_data = [] #note: stores as file lines, meaning a tile is accessed with [y][x]
    world_height = 0
    world_length = 0
    start_position = [0,0]
    previous_tile = 5

    VALID_TILES = [3,4,5]

    #Car class
    class Car:
        """
        This class represents a car racing on the track. The car decisions should be based on the following:
        * distance to left, right, and front walls.
        * current speed

        orientation and distance are used for giving cars points
        """
        max_turn_angle = 35 #represented as degrees
        max_accel = 2 
        def __init__(self,ltc,rtc,ftc,stc,lac,rac,fac,sac, orientation=(360-90)):
            """
            ltc - left turn coefficient
            rtc - right turn coefficient
            ftc - front turn coefficient
            stc - speed turn coefficient
            lac - left acceleration coefficient
            rac - right acceleration coefficient
            fac - front acceleration coefficient
            sac - speed acceleration coefficient

            left_dist, right_dist, front_dist - distance in units to the respective walls

            start_pos - initial car placement on the map


            """
            self.speed = 0
            self.orientation = orientation
            self.position = World.start_position
            self.tile = 5

            self.ltc = ltc
            self.rtc = rtc
            self.ftc = ftc
            self.stc = stc
            self.lac = lac
            self.rac = rac
            self.fac = fac
            self.sac = sac

            #sets left_dist, right_dist, front_dist, left_type, right_type 
            self.calc_dist()

        def turn(self):
            """
            uses car state and coefficients to change car turn angle.
            uses a coefficient to determine how much a car sensor impacts our turn angle.
            """
            #find steering angle
            ltc = (self.ltc / abs(self.ltc)) * (self.ltc ** 2)
            rtc = (self.rtc / abs(self.rtc)) * (self.rtc ** 2)
            ftc = (self.ftc / abs(self.ftc)) * (self.ftc ** 2)
            stc = (self.stc / abs(self.stc)) * (self.stc ** 2)

            turn_angle = max(min(self.left_dist * ltc + 
                                 self.right_dist  * rtc + 
                                 self.front_dist  * ftc + 
                                 self.speed * stc, 

                                 self.max_turn_angle),  #enforce limits
                                 -self.max_turn_angle)

            #rotate car
            self.orientation = (self.orientation + turn_angle) % 360

        def accel(self):
            """
            uses car state and coefficients to change car acceleration
            """
            
            #square of coefficients provides finer control at low values
            # (sign) * value^2
            lac = (self.ltc / abs(self.lac)) * (self.lac ** 2) 
            rac = (self.rtc / abs(self.rac)) * (self.rac ** 2)
            fac = (self.ftc / abs(self.fac)) * (self.fac ** 2)
            sac = (self.stc / abs(self.sac)) * (self.sac ** 2)

            #find acceleration
            accel = max(min(self.left_dist * lac + 
                                 self.right_dist  * rac+ 
                                 self.front_dist  * fac+ 
                                 self.speed * sac, 

                                 self.max_accel),  #enforce limits
                                 0) #no reverse means no backing up to get more points
            

            
            self.speed = self.speed + accel
            
            #calculate displacement within map bounds
            x = min(max(int(self.position[0]) + self.speed * math.cos(math.radians(self.orientation)),
                    0),World.world_length)
            
            y = min(max(int(self.position[0]) + self.speed * math.sin(math.radians(self.orientation)),
                    0),World.world_height)
            
            self.position = [x,y]

            #save current tile
            self.tile = World.world_data[round(y)][round(x)]

        #run the car for one step
        def sim_step(self):
            #move
            self.calc_dist()
            self.turn()
            self.accel()

            
        #find distance from walls
        def calc_dist(self):

            #find left
            test_dir = (self.orientation - 90) % 360    #left direction
            left_found = False
            attempts = 0
            self.left_dist = 0
            test_pos = self.position
            self.left_type = 0 #if the car is moving in reverse, the wall on the left will be a right wall
            #take car postion and orientation, add one distance unit to the left. If it is a left wall, we found our distance
            while(not left_found and (attempts < (max([World.world_height, World.world_length])))):
                #test tile
                test_pos = [test_pos[0] + (attempts * math.cos(math.radians(test_dir))), test_pos[1] + (attempts * math.sin(math.radians(test_dir)))]
                tile = World.world_data[min(max(round(test_pos[1]),0),World.world_height)][min(max(round(test_pos[0]),0),World.world_length)]
                if(int(tile) == 1 or int(tile) == 2):
                    left_found = True
                    self.left_dist = attempts
                    self.left_type = tile

                attempts += 1
                
            #find right
            test_dir = (self.orientation + 90) % 360    #right direction
            right_found = False
            attempts = 0
            self.right_dist = 0
            test_pos = self.position
            self.right_type = 0  #if the car is moving in reverse, the wall on the right will be a left wall
            #take car postion and orientation, add one distance unit to the right. If it is a right wall, we found our distance
            while(not right_found and (attempts < (max([World.world_height, World.world_length])))):
                #test tile
                tile = World.world_data[min(max(round(test_pos[1]),0),World.world_height)][min(max(round(test_pos[0]),0),World.world_length)]
                tile = World.world_data[round(test_pos[1])][round(test_pos[0])]
                if(int(tile) == 1 or int(tile) == 2):
                    right_found = True
                    self.right_dist = attempts
                    self.right_type = tile

                attempts += 1

            #find front
            test_dir = (self.orientation + 90) % 360    #front direction
            front_found = False
            attempts = 0
            self.front_dist = 0
            test_pos = self.position
            #take car postion and orientation, add one distance unit to the front. If it is a front wall, we found our distance
            while(not front_found and (attempts < (max([World.world_height, World.world_length])))):
                #test tile
                tile = World.world_data[min(max(round(test_pos[1]),0),World.world_height)][min(max(round(test_pos[0]),0),World.world_length)]
                tile = World.world_data[round(test_pos[1])][round(test_pos[0])]
                if(int(tile) == 2 or int(tile) == 1):
                    front_found = True
                    self.front_dist = attempts

                attempts += 1

    #load map
    def load_map(map = default_map):

        #read map
        with open(map) as file:
            World.world_data = [s.rstrip() for s in file.readlines()]

        #determine bounds
        World.world_height = len(World.world_data)-1
        World.world_length = len(World.world_data[0])-1

        #locate start position
        start_found = False
        y = 0
        for line in World.world_data:
            x = 0
            for tile in line:
                if tile == '5':
                    start_found = True
                    break
                x += 1
            y += 1
            if start_found: break
        World.start_position = [x,y]

    #simulate car on map
    def simulate(ltc,rtc,ftc,stc,lac,rac,fac,sac, record_frame:pd.DataFrame=None, steps=500, accel=2):
        fitness = 0

        #construct car
        racer = World.Car(ltc,rtc,ftc,stc,lac,rac,fac,sac)
        racer.max_accel = accel

        #run simulator
        timer = 0
        while(timer < steps and int(racer.tile) in World.VALID_TILES):
            old_pos = racer.position
            racer.sim_step()

            #handle checkpoints
            distance = math.sqrt((old_pos[0] - racer.position[0])**2 + (old_pos[1] - racer.position[1])**2)
            if(World.previous_tile == 3 and int(racer.tile) != 3):
                fitness += 30
            elif(int(racer.tile) != 3): #might have jumped over a checkpoint
                x_coords = np.linspace(old_pos[0], racer.position[0], round(distance))
                y_coords = np.linspace(old_pos[1], racer.position[1], round(distance))

                # check tiles in between
                for x, y in zip(x_coords, y_coords):
                    if(World.world_data[int(y)][int(x)] == '3'):
                        fitness += 30
                        break

            #penalize reverse movement
            if(int(racer.left_type) == 2 and int(racer.right_type) == 1):
                fitness -= 30 * distance

            #reward distance moved
            fitness += distance

            #if recording, save this car's path
            if record_frame is not None:
                curr_car = len(record_frame) - 1
                record_frame.at[curr_car,'x'].append(racer.position[0])
                record_frame.at[curr_car,'y'].append(racer.position[1])

            World.previous_tile = racer.tile
            timer += 1

        return [fitness, record_frame]
