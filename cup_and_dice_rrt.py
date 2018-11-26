import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
from math import pi
import numpy as np

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
        self.w, self.h = 600,600
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        ### Init pymunk and create space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -980)
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
        wall_right = 595
        wall_top = 595
        wall_bottom = 100

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

    def step_from_to(self,p1,p2):
        import cma
        # what am I doing? 
        # generating vx,vy, av for the cup! 

        def func(x):
            self.set_space(p1)
            for i in range(100):
                v = self.cup_body.velocity
                self.cup_body.velocity = (v[0]+x[i*3+0]*50,v[1]+x[i*3+1]*50)
                self.cup_body.angular_velocity += x[i*3+2]*1

                cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
                cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)

                fps = 30.
                dt = (1/fps)
                steps = 5
                for _ in range(steps):
                    self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
                    self.space.step(dt/steps)
            new_state = self.get_state()
            return np.linalg.norm(new_state-np.squeeze(p2))
        es = cma.CMAEvolutionStrategy(np.zeros(100*3),1,{'maxfevals':1000,'verbose':-9,'verb_log':0})
        es.optimize(func)

        #print(es.result_pretty())
        x = es.result.xbest

        self.set_space(p1)
        for i in range(100):
            v = self.cup_body.velocity
            self.cup_body.velocity = (v[0]+x[i*3+0]*50,v[1]+x[i*3+1]*50)
            self.cup_body.angular_velocity += x[i*3+2]*1

            cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
            cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)

            fps = 30.
            dt = (1/fps)
            steps = 5
            for _ in range(steps):
                self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
                self.space.step(dt/steps)
        new_state = np.array(self.get_state())
        return new_state, x

    def run(self):
        GOAL_TOL = 1e-3
        self.bounds = [(5,595),(100,595),(0,2*np.pi)]
        for i in range(3):
            self.bounds.append((5,595))
            self.bounds.append((100,595))
            self.bounds.append((0,2*np.pi))
            self.bounds.append((-1000,1000))
            self.bounds.append((-1000,1000))
            self.bounds.append((-10,10))
        self.bounds = np.array(self.bounds)
        self.bounds_min = self.bounds.min(1)
        self.bounds_span = self.bounds.max(1) - self.bounds_min

        nodes = np.array([self.start_state])[:np.newaxis]
        connections = np.array([0])
        forces = np.zeros((1,3*100))
        for i in range(10000):
            rand = (np.random.rand(1,21)*self.bounds_span +self.bounds_min) if np.random.rand() > 0.5 else np.array([self.goal_state])[:np.newaxis]
            dists = np.linalg.norm((nodes-rand)[:,3:],axis=1)
            closest_idx = np.argmin(dists)
            closest = nodes[closest_idx]

            new_node,sequence = self.step_from_to(closest,rand)

            if np.linalg.norm(new_node - self.goal_state) > GOAL_TOL:
                print(nodes.shape,new_node[:,np.newaxis].T.shape,closest_idx,np.linalg.norm((nodes-np.array([self.goal_state])[:np.newaxis])[:,3:],axis=1).min())
                nodes = np.append(nodes,new_node[:,np.newaxis].T,0)
                connections  = np.append(connections,closest_idx)
                forces = np.append(forces,sequence[:,np.newaxis].T,0)
            else:
                import pickle
                pickle.dump( nodes, open( "nodes.p", "wb" ) )
                pickle.dump( connections, open( "connections.p", "wb" ) )
                pickle.dump( sequence, open( "sequence.p", "wb" ) )
                break
            if i > 0 and i % 100 == 0:
                print(nodes.shape,connections.shape,forces.shape)
                print(nodes,connections,forces)
                while self.running:
                    self.set_space(self.start_state)
                    rand = np.array([self.goal_state])[:np.newaxis]
                    dists = np.linalg.norm(nodes-rand,axis=1)
                    closest_idx = np.argmin(dists)
                    closest = nodes[closest_idx]
                    path_node = closest_idx
                    final_nodes = [closest]
                    while path_node != 0:
                        self.set_space(nodes[path_node])
                        self.draw()
                        self.clock.tick(30)
                        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))
                        path_node = connections[path_node]
                        final_nodes.append(nodes[path_node])
                    final_nodes.append(nodes[0])
                    final_nodes = list(reversed(final_nodes))
                    print(len(final_nodes))
                    self.loop(final_nodes)
    def get_state(self):
        settings = [self.cup_body.position[0],self.cup_body.position[1],self.cup_body.angle]
        for i in range(3):
            settings.append(self.dice_bodies[i].position[0])
            settings.append(self.dice_bodies[i].position[1])
            settings.append(self.dice_bodies[i].angle)
            settings.append(self.dice_bodies[i].velocity[0])
            settings.append(self.dice_bodies[i].velocity[1])
            settings.append(self.dice_bodies[i].angular_velocity)

        return settings
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

    def loop(self,states):
        for s in states:
            self.set_space(s)
            cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
            
            for event in pygame.event.get():
               pass
            mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), self.screen )

            cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
            
            cup_orientation = self.cup_body.angle + pi/2
            mouse_to_cup_orientation = (mouse_position - cup_cog_world).angle
            angular_speed = 100
            #self.cup_body.angular_velocity += (mouse_to_cup_orientation - cup_orientation) * 0.2

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