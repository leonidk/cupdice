#!/usr/bin/env python3
import pygame
from pygame.constants import *
from pygame.color import *

import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

import numpy as np
from numpy import pi

import argparse
import sys
import os
import math
import pickle

class ImitationModel:
    def __init__(self,model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.regressor = model[0]
        self.dataset_mean = model[1]
        self.dataset_std = model[2]

    def pred(self, state):
        # input: N x 24 state
        # output: N x 3 velocity diffs

        state_nmlz = (state - self.dataset_mean[0:24]) / self.dataset_std[0:24]
        velocity_diff_pred_nmlz = self.regressor.predict(state_nmlz)

        velocity_diff_pred = \
            velocity_diff_pred_nmlz * self.dataset_std[24:27] + self.dataset_mean[24:27]

        return velocity_diff_pred


class CupDice:
    def __init__(self,args):
        self.collision_types = {
            "cup": 1,
            "dice": 2,
            "table": 3, 
        }
        self.start_state = [400,190,pi, 408,125,0,0,0,0, 451,125,0,0,0,0, 493,125,0,0,0,0, 0,0,0]
        self.goal_state  = [400,190,pi, 451,125,0,0,0,0, 451,165,0,0,0,0, 451,205,0,0,0,0, 0,0,0]
        self.cost_std = [242.479,209.024,0.489,130.643,64.346,1.886,491.836,103.828,2.239,125.137,54.608,1.979,468.258,98.720,2.186,126.970,62.748,2.546,471.087,102.686,2.435,497.239,109.929,1.728]
        self.cost_std = np.array(self.cost_std)
        self.running = True
        self.drawing = True
        self.w, self.h = 900,700
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.use_mouse = (args.m != 0)
        self.cup_body = None
        self.args = args
        self.dataset = []

        ### Init pymunk and create space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -1600*args.gm)
        # self.space.gravity = (0.0, 0.0)
        self.space.sleep_time_threshold = 0.3

        # walls
        wall_left = 5
        wall_right = self.w - 5
        wall_top = self.h - 5
        wall_bottom = 60

        wall_radius = 40
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
            shape.friction = 0.6
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
        self.set_space(self.start_state)
        print(self.start_state)
        print(self.get_state())
        print(np.linalg.norm((self.get_state()[3:-3]-np.squeeze(self.start_state)[3:-3])/self.cost_std[3:-3]))
        if self.args.policy == 'play':
            while self.running:
                self.loop()
        elif self.args.policy == 'model':
            model = ImitationModel(self.args.model)
            self.set_space(self.start_state)
            itr = 0
            while self.running:
                state = np.array(self.get_state()).reshape((1,-1))
                action = model.pred(state)[0]
                self.loop(action)
        elif self.args.policy == 'cma' or self.args.policy == 'de':
            policy_length = self.args.pl
            if self.args.pv > 0.5:
                xC = 100*self.args.gm # 100,100, 0.5
                yC = 10*self.args.gm # 1500,20,5
                aC = 0.5*self.args.gm
            else:
                xC = 800*self.args.gm # 100,100, 0.5
                yC = 40*self.args.gm # 1500,20,5
                aC = 2*self.args.gm
            feval_max = self.args.maxf
            def func(x):
                err = 0
                for n in range(self.args.n):
                    self.set_space(self.start_state)
                    for i in range(policy_length):
                        v = self.cup_body.velocity
                        self.cup_body.velocity = (self.args.pv*v[0] + x[i*3+0]*xC, self.args.pv*v[1] + x[i*3+1]*yC)
                        self.cup_body.angular_velocity = self.args.pv*self.cup_body.angular_velocity + x[i*3+2]*aC

                        cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
                        cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)

                        fps = 30.
                        dt = (1/fps)
                        steps = 5
                        for _ in range(steps):
                            self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
                            self.space.step(dt/steps)
                        discount = (self.args.discount ** (policy_length-i-1))
                        state_err = np.linalg.norm((self.get_state()[3:-3]-np.squeeze(self.goal_state)[3:-3])/self.cost_std[3:-3])
                        err += state_err * discount

                new_state = self.get_state()
                return err#np.linalg.norm( (new_state[3:]-np.squeeze(self.goal_state)[3:])/self.cost_std[3:]) 
            if self.args.policy == 'cma':
                import cma
                popsize = 4+int(3*np.log(3*policy_length))
                es = cma.CMAEvolutionStrategy(np.zeros(policy_length*3),0.5,{'popsize': popsize, 'maxfevals':feval_max,'verb_log':0})
                es.optimize(func)

                print(es.result_pretty())
                x = es.result.xbest
            else:
                import scipy.optimize as opt
                popsize = 15
                maxiter = int(round(feval_max/(popsize*(3*policy_length))))
                print(maxiter)
                res = opt.differential_evolution(func,
                                    bounds=[(-1,1) for _ in range(3*policy_length)],
                                    maxiter=maxiter,polish=False,tol=0.001,popsize=popsize,
                                    disp=True)
                x = res.x
            print(func(np.zeros(policy_length*3)),func(x))
            self.set_space(self.start_state)
            itr = 0
            while self.running:
                if itr % policy_length == 0:
                    self.set_space(self.start_state)
                mi = itr % policy_length
                self.loop(x[mi:mi+3] * np.array([xC,yC,aC]))
                itr += 1
            
    def get_state(self):
        settings = [self.cup_body.position[0],self.cup_body.position[1],self.cup_body.angle]
        for i in range(3):
            settings.append(self.dice_bodies[i].position[0])
            settings.append(self.dice_bodies[i].position[1])
            settings.append(self.dice_bodies[i].angle)
            settings.append(self.dice_bodies[i].velocity[0])
            settings.append(self.dice_bodies[i].velocity[1])
            settings.append(self.dice_bodies[i].angular_velocity)
        settings.append(self.cup_body.velocity[0])
        settings.append(self.cup_body.velocity[1])
        settings.append(self.cup_body.angular_velocity)
        return settings

    def save_dataset(self):
        if (self.args.r != 0):
            base_name = "imitate_{}.csv"
            i = 0
            while True:
                if not os.path.exists(base_name.format(i)):
                    break
                i+=1
            recording_name = base_name.format(i)
            np.savetxt(recording_name, np.array(self.dataset), delimiter=",")

    def set_space(self, settings):
        assert(len(settings) == 24)

        if self.cup_body is not None:
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

        cup_wall1.friction = 0.4
        cup_wall2.friction = 0.4
        cup_wall3.friction = 0.4
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
        self.space.step(dt*0.001)

    def loop(self, action_vector = None):
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
                self.dataset = []
            elif event.type == KEYDOWN and event.key == K_s:
                self.save_dataset()
                self.set_space(self.start_state)
                self.dataset = []
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
            elif event.type == KEYUP and event.key == K_m:
                self.use_mouse = not self.use_mouse

        speed = 100*self.args.gm
        key_ang_speed = 0.5*self.args.gm
        if action_vector is not None:
            v = self.cup_body.velocity
            self.cup_body.velocity = ( self.args.pv*v[0]+action_vector[0], self.args.pv*v[1]+action_vector[1])
            self.cup_body.angular_velocity = self.args.pv*self.cup_body.angular_velocity + action_vector[2]
        else:
            if self.left_down:
                v = self.cup_body.velocity
                self.cup_body.velocity = (v[0]-speed,v[1])
            if self.right_down:
                v = self.cup_body.velocity
                self.cup_body.velocity = (v[0]+speed,v[1])
            if self.up_down:
                v = self.cup_body.velocity
                self.cup_body.velocity = (v[0],v[1]+speed)
            if self.down_down:
                v = self.cup_body.velocity
                self.cup_body.velocity = (v[0],v[1]-speed)
            if self.e_down:
                self.cup_body.angular_velocity -= key_ang_speed
            if self.q_down:
                self.cup_body.angular_velocity += key_ang_speed
            if not self.down_down and not self.up_down:
                v = self.cup_body.velocity
                self.cup_body.velocity = (v[0],0)
            if not self.left_down and not self.right_down:
                v = self.cup_body.velocity
                self.cup_body.velocity = (0,v[1])
            if not self.q_down and not self.e_down:
                self.cup_body.angular_velocity = 0
            if self.use_mouse:
                mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), self.screen )

                cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)

                cup_orientation =self.cup_body.angle + pi/2
                mouse_to_cup_orientation = (mouse_position - cup_cog_world).angle
                
                angular_speed = 10
                dist1 = mouse_to_cup_orientation - cup_orientation
                #print([_ for _ in [self.cup_body.angle,cup_orientation,mouse_to_cup_orientation,self.cup_body.angle]])
                dist2 = dist1 + 2*pi
                dist3 = dist1 - 2*pi
                dists = [dist1,dist2]
                dists = sorted([(abs(_),_) for _ in dists])
                self.cup_body.angular_velocity += dists[0][1] * angular_speed
                #print(dists[0][1])
                #print(math.fmod(mouse_to_cup_orientation,pi),math.fmod(cup_orientation,pi),mouse_to_cup_orientation,dist1,dist2,dist3)
                #print(dist1,dist2,dist3)
        cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)

        self.space.reindex_shapes_for_body(self.cup_body)
        self.space.iterations = 25

        fps = 30.
        dt = (1/fps)
        steps = 5
        for _ in range(steps):
            self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
            self.space.step(dt/steps)
        if self.drawing:
            self.draw()
        if (self.args.r != 0):
            self.dataset.append(self.get_state())

        ### Tick clock and update fps in title
        self.clock.tick(fps)

    def draw(self):
        ### Clear the screen
        self.screen.fill(THECOLORS["white"])

        font = pygame.font.Font('RobotoMono-Regular.ttf', 30)
        text =  "move with the arrow keys\n"+\
                "q: rotate ccw\n" +\
                "e: rotate cw\n" +\
                "m: mouse control\n"+\
                "o: start state\n"+\
                "i: final state"
        y = 65
        for line in text.splitlines():
            text = font.render(line, 1,THECOLORS["grey"])
            self.screen.blit(text, (65,y))
            y += 30

        ### Draw space
        self.space.debug_draw(self.draw_options)

        ### All done, lets flip the display
        pygame.display.flip()

def main(args):
    pygame.init()
    demo = CupDice(args)
    demo.run()
    pygame.display.quit()
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--m", action="store_true",
                        help="use mouse input")
    parser.add_argument("--gm", type=float,default=1.0,
                        help="change gravity and forces")
    parser.add_argument("--pv", type=float,default=1.0,
                        help="amount of previous velocity to use")
    parser.add_argument("--discount", type=float,default=0.1,
                        help="change discount factor")
    parser.add_argument("--maxf", type=int,default=5000,
                        help="maximum number of function evaluations")
    parser.add_argument("--n", type=int,default=1,
                        help="number of iterations for optimization")
    parser.add_argument("--pl", type=int,default=50,
                        help="length of learned policy")
    parser.add_argument("--r", action="store_true",
                        help="record data")
    parser.add_argument("--model", type=str,default='model.pkl',
                        help="modefile to evaluate")
    parser.add_argument('policy', nargs='?', default='play',choices=['play','cma','de','model'])
    args = parser.parse_args()
    main(args)
