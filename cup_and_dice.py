import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
from math import pi

from pdb import set_trace as st

class CupDice:        
    def __init__(self):
        self.collision_types = {
            "cup": 1,
            "dice": 2,
            "table": 3,
        }

        self.running = True
        self.drawing = True
        self.w, self.h = 600,600
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        ### Init pymunk and create space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -50000.0)
        # self.space.gravity = (0.0, 0.0)
        self.space.sleep_time_threshold = 0.3
        
        ### cup
        self.cup_body = pymunk.Body()
        cup_radius = 5
        cup_wall1_length = 170
        cup_wall2_length = 100

        cup_wall1 = pymunk.Segment(self.cup_body, (300,cup_wall1_length + 300),(300,300),cup_radius) #left
        cup_wall2 = pymunk.Segment(self.cup_body, (300,300),(cup_wall2_length + 300,300),cup_radius) #bottom
        cup_wall3 = pymunk.Segment(self.cup_body, (cup_wall2_length + 300,300),(cup_wall2_length + 300,cup_wall1_length + 300),cup_radius) #right
        cup_wall1.mass = 5
        cup_wall2.mass = 5
        cup_wall3.mass = 5
        cup_wall1.collision_type = self.collision_types["cup"]
        cup_wall2.collision_type = self.collision_types["cup"]
        cup_wall3.collision_type = self.collision_types["cup"]

        cup_wall1.friction = 1
        cup_wall2.friction = 1
        cup_wall3.friction = 1
        self.space.add(self.cup_body, cup_wall1, cup_wall2, cup_wall3)
        # self.space.add(self.cup_body, cup_wall1)

        # walls
        wall_left = 5
        wall_right = 595
        wall_top = 595
        wall_bottom = 100

        wall_radius = 5
        wall1_shape = pymunk.Segment(self.space.static_body, (wall_left, wall_bottom), (wall_right,wall_bottom), wall_radius) #bottom
        wall2_shape = pymunk.Segment(self.space.static_body, (wall_left, wall_bottom), (wall_left, wall_top), wall_radius) #left
        wall3_shape = pymunk.Segment(self.space.static_body, (wall_right, wall_bottom), (wall_right, wall_top), wall_radius) #right
        wall4_shape = pymunk.Segment(self.space.static_body, (wall_left, wall_top), (wall_right,wall_top), wall_radius) #top
        
        wall1_shape.friction = 1.0
        wall2_shape.friction = 1.0
        wall3_shape.friction = 1.0
        wall4_shape.friction = 1.0

        wall1_shape.collision_type = self.collision_types["table"]
        wall2_shape.collision_type = self.collision_types["table"]
        wall3_shape.collision_type = self.collision_types["table"]
        wall4_shape.collision_type = self.collision_types["table"]
        self.space.add(wall1_shape, wall2_shape, wall3_shape, wall4_shape)
        
        # dice
        box_pos = Vec2d(100, 150)
        delta_box_pos = Vec2d(50, 0)

        for i in range(3):
            size = 20
            points = [(-size, -size), (-size, size), (size,size), (size, -size)]
            mass = 1.0
            moment = pymunk.moment_for_poly(mass, points, (0,0))
            body = pymunk.Body(mass, moment)
            body.position = box_pos
            shape = pymunk.Poly(body, points)
            shape.collision_type = self.collision_types["dice"]
            shape.friction = 1
            self.space.add(body,shape)
            box_pos = box_pos + delta_box_pos
        
        ### draw options for drawing
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)


    def run(self):
        while self.running:
            self.loop() 



    def loop(self): 
        cup_speed = 17
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self.screen, "box2d_pyramid.png")
            elif event.type == KEYDOWN and event.key == K_d:
                self.drawing = not self.drawing

            elif event.type == KEYDOWN and event.key == K_LEFT:
                self.cup_body.velocity = Vec2d(-600,0) * cup_speed
            elif event.type == KEYUP and event.key == K_LEFT:
                self.cup_body.velocity = Vec2d(0,0)
                
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                self.cup_body.velocity = Vec2d(600,0) * cup_speed
            elif event.type == KEYUP and event.key == K_RIGHT:
                self.cup_body.velocity = Vec2d(0,0)

            elif event.type == KEYDOWN and event.key == K_UP:
                self.cup_body.velocity = Vec2d(0,600) * cup_speed
            elif event.type == KEYUP and event.key == K_UP:
                self.cup_body.velocity = Vec2d(0,0)
                
            elif event.type == KEYDOWN and event.key == K_DOWN:
                self.cup_body.velocity = Vec2d(0,-600) * cup_speed
            elif event.type == KEYUP and event.key == K_DOWN:
                self.cup_body.velocity = Vec2d(0,0)
        
        mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), self.screen )

        cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
        
        cup_orientation = self.cup_body.angle + pi/2
        mouse_to_cup_orientation = (mouse_position - cup_cog_world).angle
        angular_speed = 400
        self.cup_body.angular_velocity = (mouse_to_cup_orientation - cup_orientation) * angular_speed

        cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)
        print(cup_body_reverse_gravity)
        self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)

        self.space.reindex_shapes_for_body(self.cup_body)

        fps = 30.
        dt = 1.0/fps/50        
        self.space.step(dt)
        if self.drawing:
            self.draw()
        
        ### Tick clock and update fps in title
        self.clock.tick(fps)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))
        
    def draw(self):
        ### Clear the screen
        self.screen.fill(THECOLORS["white"])
        
        ### Draw space
        self.space.debug_draw(self.draw_options)

        ### All done, lets flip the display
        pygame.display.flip()        

def main():
    demo = CupDice()
    demo.run()

if __name__ == '__main__':
    main()