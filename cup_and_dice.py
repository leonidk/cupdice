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
        self.start_state = [300,300,0, 100,125,0,0,0,0, 200,125,0,0,0,0, 300,125,0,0,0,0 ]
        self.goal_state  = [300,300,0, 200,125,0,0,0,0, 200,165,0,0,0,0, 200,205,0,0,0,0 ]
        self.running = True
        self.drawing = True
        self.w, self.h = 900,700
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        ### Init pymunk and create space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -80)
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

        cup_wall1.friction = 0.8
        cup_wall2.friction = 0.8
        cup_wall3.friction = 0.8
        self.cup_walls = [cup_wall1,cup_wall2,cup_wall3]
        self.space.add(self.cup_body, cup_wall1, cup_wall2, cup_wall3)
        # self.space.add(self.cup_body, cup_wall1)

        # walls
        wall_left = 5
        wall_right = self.w - 5
        wall_top = self.h - 5
        wall_bottom = 50

        wall_radius = 3
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

        self.dice_bodies = []

        for i in range(3):
            size = 20
            cs = 5
            points = [(-size+cs, -size), (-size, -size+cs),(-size,size-cs),(-size+cs,size),(size-cs,size),(size,size-cs),(size,-size+cs),(size-cs,-size)]
            mass = 1.0
            moment = pymunk.moment_for_poly(mass, points, (0,0))
            body = pymunk.Body(mass, moment)
            self.dice_bodies.append(body)
            body.position = box_pos
            shape = pymunk.Poly(body, points)
            shape.collision_type = self.collision_types["dice"]
            shape.friction = 0.5
            self.space.add(body,shape)
            box_pos = box_pos + delta_box_pos

        self.set_space(self.start_state)

        ### draw options for drawing
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.left_down = False
        self.right_down = False
        self.up_down = False
        self.down_down = False
        self.q_down = False
        self.e_down = False
    def run(self):
        while self.running:
            self.loop()

    def set_space(self, settings):
        assert(len(settings) == 21)

        self.space.remove(self.cup_body)
        for a in self.cup_walls:
            self.space.remove(a)
        self.cup_body = pymunk.Body()
        cup_radius = 5
        cup_wall1_length = 220
        cup_wall2_length = 100
        cup_start_x = settings[0]
        cup_start_y = settings[1]
        splay = 20

        cup_wall1 = pymunk.Segment(self.cup_body, (cup_start_x-splay,cup_wall1_length + cup_start_y),(cup_start_x,cup_start_y),cup_radius) #left
        cup_wall2 = pymunk.Segment(self.cup_body, (cup_start_x,cup_start_y),(cup_wall2_length + cup_start_x,cup_start_y),cup_radius) #bottom
        cup_wall3 = pymunk.Segment(self.cup_body, (cup_wall2_length + cup_start_x,cup_start_y),(cup_wall2_length + cup_start_x + splay,cup_wall1_length + cup_start_y),cup_radius) #right
        cup_wall1.mass = 5
        cup_wall2.mass = 5
        cup_wall3.mass = 5
        cup_wall1.collision_type = self.collision_types["cup"]
        cup_wall2.collision_type = self.collision_types["cup"]
        cup_wall3.collision_type = self.collision_types["cup"]

        cup_wall1.friction = 0.8
        cup_wall2.friction = 0.8
        cup_wall3.friction = 0.8
        self.cup_walls = [cup_wall1,cup_wall2,cup_wall3]
        self.space.add(self.cup_body, cup_wall1, cup_wall2, cup_wall3)
        self.cup_body.angle = settings[2]
        for i in range(3):
            self.dice_bodies[i].position = (settings[3 + i * 6], settings[4 + i * 6])
            self.dice_bodies[i].angle = settings[5 + i * 6]
            self.dice_bodies[i].velocity = (settings[6 + i * 6], settings[7 + i * 6])
            self.dice_bodies[i].angular_velocity = settings[8 + i * 6]
            self.space.reindex_shapes_for_body(self.dice_bodies[i])

        self.space.reindex_shapes_for_body(self.cup_body)
        fps = 30.
        dt = 1.0/fps/50
        self.space.step(dt)

    def loop(self):

        cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)

        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self.running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self.screen, "box2d_pyramid.png")
            elif event.type == KEYDOWN and event.key == K_d:
                self.drawing = not self.drawing
            elif event.type == KEYDOWN and event.key == K_o:
                self.set_space(self.start_state)
            elif event.type == KEYDOWN and event.key == K_i:
                self.set_space(self.goal_state)
            elif event.type == KEYDOWN and event.key == K_LEFT:
                self.left_down = True
            elif event.type == KEYUP and event.key == K_LEFT:
                self.left_down = False
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                self.right_down = True
            elif event.type == KEYUP and event.key == K_RIGHT:
                self.right_down = False
            elif event.type == KEYDOWN and event.key == K_UP:
                self.up_down = True
            elif event.type == KEYUP and event.key == K_UP:
                self.up_down = False
            elif event.type == KEYDOWN and event.key == K_DOWN:
                self.down_down = True
            elif event.type == KEYUP and event.key == K_DOWN:
                self.down_down = False
            elif event.type == KEYDOWN and event.key == K_q:
                self.q_down = True
            elif event.type == KEYUP and event.key == K_q:
                self.q_down = False
            elif event.type == KEYDOWN and event.key == K_e:
                self.e_down = True
            elif event.type == KEYUP and event.key == K_e:
                self.e_down = False

        speed = 30
        
        if self.left_down:
            v = self.cup_body.velocity
            self.cup_body.velocity = (v[0]-speed,v[1])
            #self.cup_body.apply_force_at_world_point((-1000,0),cup_cog_world)
            #self.cup_body.apply_impulse_at_world_point((-1000,0),cup_cog_world)
        if self.right_down:
            v = self.cup_body.velocity
            self.cup_body.velocity = (v[0]+speed,v[1])
            #self.cup_body.apply_force_at_world_point((1000,0),cup_cog_world)
            #self.cup_body.apply_impulse_at_world_point((1000,0),cup_cog_world)

        if self.up_down:
            v = self.cup_body.velocity
            self.cup_body.velocity = (v[0],v[1]+speed)
            #self.cup_body.apply_force_at_world_point((0,1000),cup_cog_world)
            #self.cup_body.apply_impulse_at_world_point((0,1000),cup_cog_world)

        if self.down_down:
            v = self.cup_body.velocity
            self.cup_body.velocity = (v[0],v[1]-speed)
            #self.cup_body.apply_force_at_world_point((0,-1000),cup_cog_world)
            #self.cup_body.apply_impulse_at_world_point((0,-1000),cup_cog_world)
        if self.e_down:
            self.cup_body.angular_velocity -= 0.1
        if self.q_down:
            self.cup_body.angular_velocity += 0.1
        if not self.down_down and not self.up_down:
            v = self.cup_body.velocity
            self.cup_body.velocity = (v[0],0)
        if not self.left_down and not self.right_down:
            v = self.cup_body.velocity
            self.cup_body.velocity = (0,v[1])
        if not self.q_down and not self.e_down:
            self.cup_body.angular_velocity = 0
        #if not evented:
        #    self.cup_body.velocity = (0,0)
        #    self.cup_body.angular_velocity = 0
        mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), self.screen )

        cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)

        cup_orientation = self.cup_body.angle + pi/2
        # print('cup_orientation: {:.2f}'.format(cup_orientation))
        mouse_to_cup_orientation = (mouse_position - cup_cog_world).angle
        # print('mouse_to_cup_orientation: {:.2f}'.format(mouse_to_cup_orientation))
        
        angular_speed = 10
        dist1 = mouse_to_cup_orientation - cup_orientation
        dist2 = dist1 + 2*pi
        if abs(dist1) < abs(dist2):
            self.cup_body.angular_velocity += dist1 * angular_speed
        else:
            self.cup_body.angular_velocity += dist2 * angular_speed

        cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)
        #print(cup_body_reverse_gravity)

        self.space.reindex_shapes_for_body(self.cup_body)
        self.space.iterations = 25

        fps = 30.
        dt = (1/fps)
        steps = 5
        for _ in range(steps):
            self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
            self.space.step(dt/steps)
            #self.cup_body.angular_velocity =0
            #self.cup_body.velocity =(0,0)

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
