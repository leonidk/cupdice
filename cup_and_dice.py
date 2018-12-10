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
import fnmatch

from pdb import set_trace as st

class ImitationModel:
    def __init__(self,model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.regressor = model[0]
        self.dataset_mean = model[1]
        self.dataset_std = model[2]

        self.dataset = []
        for fname in sorted(os.listdir('data')):
            if fnmatch.fnmatch(fname, "imitate*.csv"):
                d = np.loadtxt(os.path.join('data',fname) ,delimiter=',')
                self.dataset.append(d)
        self.dataset = np.vstack(self.dataset)

    def pred(self, state):
        # input: N x 24 state
        # output: N x 3 velocity diffs

        state_nmlz = (state - self.dataset_mean[:-3]) / self.dataset_std[:-3]
        velocity_diff_pred_nmlz = self.regressor.predict(state_nmlz)

        velocity_diff_pred = \
            velocity_diff_pred_nmlz * self.dataset_std[-3:] + self.dataset_mean[-3:]

        return velocity_diff_pred
    
    def partial_fit(self):
        dataset_nmlz = (self.dataset - self.dataset_mean) / self.dataset_std
        state = dataset_nmlz[:,:-3]
        velocity_diff = dataset_nmlz[:,-3:]
        self.regressor.fit(state,velocity_diff)


class CupDice:
    def __init__(self,args):
        self.collision_types = {
            "cup": 1,
            "dice": 2,
            "table": 3, 
        }
        self.start_state = [400,190,pi, 408,125,0,0,0,0, 451,125,0,0,0,0, 493,125,0,0,0,0, 0,0,0]
        self.goal_state  = [400,190,pi, 451,125,0,0,0,0, 451,165,0,0,0,0, 451,205,0,0,0,0, 0,0,0]
       
        #self.start_state = [400,190,pi, 408,125,0,0,0,0, 451,125,0,0,0,0, 493,125,0,0,0,0, 0,0,0]
        #self.goal_state  = [400,190,pi, 451,125,0,0,0,0, 451,165,0,0,0,0, 451,205,0,0,0,0, 0,0,0]
        self.cost_std = [242.479,209.024,0.489,130.643,64.346,1.886,491.836,103.828,2.239,125.137,54.608,1.979,468.258,98.720,2.186,126.970,62.748,2.546,471.087,102.686,2.435,497.239,109.929,1.728]
        self.cost_std = [166.918,196.105,0.344,77.405,44.492,1.140,274.843,88.957,2.186,83.403,30.683,1.127,298.744,96.776,1.551,94.562,50.006,1.890,320.849,102.481,2.163,411.637,133.321,1.145,143.081,74.641,0.275]
        self.cost_std = np.array(self.cost_std)
        self.cost_avg = [889.190,564.406,3.106,441.687,156.780,-0.032,4.804,2.866,0.038,452.212,165.340,-0.705,-2.379,4.799,-0.109,462.736,174.106,0.547,-9.477,6.885,0.152,-5.317,30.342,0.008,2.428,9.099,-0.012]
        self.cost_avg = np.array(self.cost_avg)
        self.running = True
        self.drawing = True
        self.w, self.h = 900,700
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.use_mouse = (args.m != 0)
        self.cup_body = None
        self.args = args
        self.dataset = []
        self.recording = (self.args.r != 0)

        ### Init pymunk and create space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -400*args.gm)
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

        self.set_space(self.start_state)
        self.true_start = self.get_state()
        self.set_space(self.goal_state)
        self.true_goal = self.get_state()


    def run(self):
        self.set_space(self.start_state)
        print(self.start_state)
        print(self.get_state())
        print(np.linalg.norm((self.get_state()[3:]-np.squeeze(self.start_state)[3:])/self.cost_std[3:-3]))

        if self.args.pv > 0.5:
            xC = 100*self.args.gm # 100,100, 0.5
            yC = 10*self.args.gm # 1500,20,5
            aC = 0.5*self.args.gm
        else:
            xC = 800*self.args.gm # 100,100, 0.5
            yC = 40*self.args.gm # 1500,20,5
            aC = 2*self.args.gm
        if self.args.policy == 'play':
            while self.running:
                self.loop()
        elif self.args.policy == 'model':
            model = ImitationModel(self.args.model)
            self.model = model
            self.set_space(self.start_state)
            itr = 0
            while self.running:
                state = np.array(self.get_state()).reshape((1,-1))
                action = model.pred(state)[0]
                self.loop(action)
        elif self.args.policy == 'rrt':
            policy_length = self.args.pl
            def step_from_to(s1,s2):
                def func(x):
                    err = 0
                    for n in range(self.args.n):
                        self.set_space(s1)
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
                                self.space.step(dt/steps)
                            discount = (self.args.discount ** (policy_length-i-1))
                            err_vec = (self.get_state()[3:]-np.squeeze(s2)[3:])/self.cost_std[3:-3]
                            state_err = np.linalg.norm(err_vec)
                            #state_err = np.linalg.norm([err_vec[0],err_vec[1],err_vec[6],err_vec[7],err_vec[12],err_vec[13]])
                            err += state_err * discount

                    return err
                import cma
                popsize = 4+int(3*np.log(3*policy_length))
                x0 = np.zeros(3*policy_length)
                x0 = np.random.rand(3*policy_length)*2 - 1

                es = cma.CMAEvolutionStrategy(x0,0.25,{'verbose':-9,'popsize': popsize, 'maxfevals':self.args.maxf,'verb_log':0})
                es.optimize(func)

                x = es.result.xbest
                self.set_space(s1)
                for i in range(policy_length):
                    v = self.cup_body.velocity
                    self.cup_body.velocity = (self.args.pv*v[0] + x[i*3+0]*xC, self.args.pv*v[1] + x[i*3+1]*yC)
                    self.cup_body.angular_velocity = self.args.pv*self.cup_body.angular_velocity + x[i*3+2]*aC

                    fps = 30.
                    dt = (1/fps)
                    steps = 5
                    for _ in range(steps):
                        self.space.step(dt/steps)
                new_state = np.array(self.get_state())
                return new_state, x
            nodes = np.array([self.start_state])[:np.newaxis]
            connections = np.array([0])
            forces = np.zeros((1,3*self.args.pl))
            for i in range(10000):
                rand = (np.random.rand(1,24)*3*self.cost_std[:-3] + self.cost_avg[:-3]) if np.random.rand() > 0.25 else np.array([self.true_goal])[:np.newaxis]

                err_vec = ((nodes-rand)[:,3:])/self.cost_std[3:-3]
                #state_err = np.linalg.norm(err_vec)
                dists = np.linalg.norm(np.vstack([err_vec[:,0],err_vec[:,1],err_vec[:,6],err_vec[:,7],err_vec[:,12],err_vec[:,13]]).T,axis=1)
                #print(dists)
                closest_idx = np.argmin(dists)
                closest = nodes[closest_idx]

                new_node,sequence = step_from_to(closest,rand)

                if np.linalg.norm(new_node - self.true_goal) > 0.01:
                    print(nodes.shape,new_node[:,np.newaxis].T.shape,closest_idx,np.linalg.norm((nodes-np.array([self.true_goal])[:np.newaxis])[:,3:]/self.cost_std[3:-3],axis=1).min())
                    #print(np.linalg.norm((nodes-np.array([self.goal_state])[:np.newaxis])/self.cost_std[:-3],axis=1))
                    nodes = np.append(nodes,new_node[:,np.newaxis].T,0)
                    connections  = np.append(connections,closest_idx)
                    forces = np.append(forces,sequence[:,np.newaxis].T,0)
                else:
                    import pickle
                    pickle.dump( nodes, open( "nodes.p", "wb" ) )
                    pickle.dump( connections, open( "connections.p", "wb" ) )
                    pickle.dump( sequence, open( "sequence.p", "wb" ) )
                    break
                if False and i > 0 and i % 5 == 0:
                    while self.running:
                    #print(nodes.shape,connections.shape,forces.shape)
                    #print(nodes,connections,forces)

                        self.set_space(self.start_state)
                        rand = np.array([self.true_goal])[:np.newaxis]
                        dists = np.linalg.norm(nodes-rand,axis=1)
                        closest_idx = np.argmin(dists)
                        closest = nodes[closest_idx]
                        path_node = closest_idx
                        final_nodes = [closest]
                        while path_node != 0:
                            for i in range(self.args.pl):
                                self.loop(forces[path_node][3*i:3*i+3])
                            path_node = connections[path_node]
                            final_nodes.append(nodes[path_node])
                        final_nodes.append(nodes[0])
                        final_nodes = list(reversed(final_nodes))
        elif self.args.policy == 'pg':
            H = 32
            W1 = (np.random.randn(24,H))/np.sqrt(24/2)*0.1
            W2 = (np.random.randn(H,H))/np.sqrt(H/2)*0.1
            W3 = (np.random.randn(H,3))/np.sqrt(H/2)*0.1

            pl = self.args.pl
            r_max = None
            LR = 1e-6
            g_array = np.ones(pl)
            for i in range(pl):
                g_array[i] = 0.99 ** float(i)
            for runs in range(200):
                rewards = 0
                grad_W1 = np.zeros_like(W1)
                grad_W2 = np.zeros_like(W2)
                grad_W3= np.zeros_like(W3)

                for i_episode in range(16):
                    self.set_space(self.start_state)
                    epi_grad_W1 = np.zeros([pl] + list(W1.shape))
                    epi_grad_W2 = np.zeros([pl] + list(W2.shape))
                    epi_grad_W3 = np.zeros([pl] + list(W3.shape))

                    epi_rewards = np.zeros(pl)
                    alpha = 0.05
                    for t in range(pl):
                        features = (np.array(self.get_state())-self.cost_avg[:-3])/self.cost_std[:-3]

                        h1 = np.dot(features,W1)
                        h2 = np.maximum(h1,alpha*h1)
                        h3 = np.dot(h2,W2)
                        h4 = np.maximum(h3,alpha*h3)
                        h5 = np.dot(h4,W3)
                        h6 = np.tanh(h5) + np.random.randn(3)/2
                        x = (h6)
                        grad_log = ((h5 - h6)*((1/2)**2))*(1-np.tanh(h5)**2)

                        v = self.cup_body.velocity
                        self.cup_body.velocity = (v[0]+x[0]*xC,v[1]+x[1]*yC)
                        self.cup_body.angular_velocity += x[2]*aC


                        fps = 30.
                        dt = (1/fps)
                        steps = 5
                        for _ in range(steps):
                            self.space.step(dt/steps)
                        state = np.array(self.get_state()) 
                        
                        discount = (self.args.discount ** (pl-t-1))
                        err_vec = (self.get_state()[3:]-np.squeeze(self.true_goal)[3:])#/self.cost_std[3:-3]
                        #err_vec /= self.cost_std[3:-3]
                        #state_err = np.linalg.norm(err_vec)
                        state_err = np.linalg.norm([err_vec[0],err_vec[1],err_vec[6],err_vec[7],err_vec[12],err_vec[13]])
                        #reward = -np.linalg.norm((state - np.array(self.goal_state))/self.bounds_span)
                        ##reward = -np.linalg.norm(np.array([state[3],state[9],state[15]]) - 200)
                        reward = -discount*state_err
                        epi_rewards[t] = reward
                        rewards += reward

            
                        grad_log_W3 = np.dot(h4[:,np.newaxis],grad_log[:,np.newaxis].T)
                        grad_log_h4 = np.dot(grad_log,W3.T)
                        grad_log_h4[ h4 <= 0] = alpha*grad_log_h4[ h4 <= 0]
                        grad_log_W2 = np.dot(h2[:,np.newaxis],grad_log_h4[:,np.newaxis].T)
                        grad_log_h2 = np.dot(grad_log_h4,W2.T)
                        grad_log_h2[ h2 <= 0] = alpha*grad_log_h2[ h2 <= 0]
                        grad_log_W1 = np.dot(features[:,np.newaxis],grad_log_h2[:,np.newaxis].T)

                        epi_grad_W3[t,:,:] = grad_log_W3
                        epi_grad_W2[t,:,:] = grad_log_W2
                        epi_grad_W1[t,:,:] = grad_log_W1
                    scale = 1
                    if r_max is None:
                        r_max = np.zeros_like(epi_rewards)
                        scale = 10
                    epi_sums = np.zeros_like(epi_rewards)
                    for t in range(pl):
                        r = np.sum(epi_rewards[t:] * g_array[:pl-t])
                        if scale == 10:
                            r_max[t] = r
                        grad_W3 += epi_grad_W3[t]*(r-r_max[t])
                        grad_W2 += epi_grad_W2[t]*(r-r_max[t])
                        grad_W1 += epi_grad_W1[t]*(r-r_max[t])
                        r_max[t] = 0.99*r_max[t] + 0.01*r
                        epi_sums[t] = r
                #r_max = 0.9*r_max + 0.1*(rewards/64.0)

                W1= W1 + LR*(grad_W1) 
                W2= W2 + LR*(grad_W2)
                W3= W3 + LR*(grad_W3)

                print('R: {0:6.1f}\tB: {3:6.1f}\t\t|W1| {1:.3f} |W2| {2:.3f}'.format(epi_sums.sum(),np.linalg.norm(W1),np.linalg.norm(W2),r_max.sum()))
            while self.running:
                self.set_space(self.start_state)

                for i in range(self.args.pl):
                    features = (np.array(self.get_state())-self.cost_avg[:-3])/self.cost_std[:-3]

                    h1 = np.dot(features,W1)
                    h2 = np.maximum(h1,alpha*h1)
                    h3 = np.dot(h2,W2)
                    h4 = np.maximum(h3,alpha*h3)
                    h5 = np.dot(h4,W3)
                    h6 = np.tanh(h5) + np.random.randn(3)/2
                    x = (h6)
                    self.loop([x[0]*xC,x[1]*yC,x[2]*aC])
        elif self.args.policy == 'cma' or self.args.policy == 'de' or self.args.policy == 'opt':
            policy_length = self.args.pl

            feval_max = self.args.maxf
            def func(x):
                err = 0
                for n in range(self.args.n):
                    self.set_space(self.start_state)
                    for i in range(policy_length+3):
                        v = self.cup_body.velocity
                        if i < policy_length:
                            self.cup_body.velocity = (self.args.pv*v[0] + x[i*3+0]*xC, self.args.pv*v[1] + x[i*3+1]*yC)
                            self.cup_body.angular_velocity = self.args.pv*self.cup_body.angular_velocity + x[i*3+2]*aC

                        cup_cog_world = self.cup_body.local_to_world(self.cup_body.center_of_gravity)
                        cup_body_reverse_gravity = -(self.cup_body.mass * self.space.gravity)

                        fps = 30.
                        dt = (1/fps)
                        steps = 5
                        for _ in range(steps):
                            #self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
                            self.space.step(dt/steps)
                        discount = (self.args.discount ** (policy_length-i-1))
                        err_vec = (self.get_state()[3:]-np.squeeze(self.true_goal)[3:])#/self.cost_std[3:-3]
                        err_vec /= self.cost_std[3:-3]
                        state_err = np.linalg.norm(err_vec)
                        #state_err = np.linalg.norm([err_vec[0],err_vec[1],err_vec[6],err_vec[7],err_vec[12],err_vec[13]])
                        #print(i,state_err)
                        err += state_err * discount

                new_state = self.get_state()
                return err#np.linalg.norm( (new_state[3:]-np.squeeze(self.goal_state)[3:])/self.cost_std[3:]) 
            if self.args.policy == 'cma':
                import cma
                popsize = 4+int(3*np.log(3*policy_length))
                x0 = np.random.rand(3*policy_length)*2 - 1

                es = cma.CMAEvolutionStrategy(x0,0.25,{'popsize': popsize, 'maxfevals':feval_max,'verb_log':0})
                es.optimize(func)

                print(es.result_pretty())
                x = es.result.xbest
            elif self.args.policy == 'opt':
                import scipy.optimize as opt
                x0 = np.random.rand(3*policy_length)*2 - 1
                res = opt.basinhopping(func,x0,disp=True)
                x = res.x
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
            
            base_name = "{}_{}.csv"
            i = 0
            while True:
                if not os.path.exists(base_name.format(self.args.policy,i)):
                    break
                i+=1
            recording_name = base_name.format(self.args.policy,i)
            itr = 0
            self.dataset = []
            saved = False
            self.recording = True
            while self.running:
                if itr % policy_length == 0:
                    self.set_space(self.start_state)
                    if itr > 0 and (saved == False):
                        saved = True
                        np.savetxt(recording_name, np.array(self.dataset), delimiter=",")
                        self.dataset = []
                        self.recording = False
                mi = itr % policy_length
                self.loop(x[3*mi:3*mi+3] * np.array([xC,yC,aC]))
                itr += 1
        elif self.args.policy == 'replay':
            while self.running:
                self.set_space(self.start_state)
                replay_states = np.loadtxt(args.imitate_file, delimiter=',')
                num_states = replay_states.shape[0]

                for i in range(num_states):
                    action = replay_states[i, -3:]
                    self.loop(action)
            
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
        if self.recording:
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
                if self.args.policy == 'model':
                    print(self.model.dataset.shape,np.array(self.dataset).shape)
                    self.model.dataset = np.vstack([self.model.dataset,np.array(self.dataset)])
                    self.model.partial_fit()
                self.dataset = []
            elif event.type == KEYDOWN and event.key == K_u:
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
        olv = self.cup_body.velocity
        oav = self.cup_body.angular_velocity
        
        speed = 100*self.args.gm
        key_ang_speed = 0.5*self.args.gm
        if action_vector is not None:
            v = self.cup_body.velocity
            self.cup_body.velocity = ( self.args.pv*v[0]+action_vector[0], self.args.pv*v[1]+action_vector[1])
            self.cup_body.angular_velocity = self.args.pv*self.cup_body.angular_velocity + action_vector[2]
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
        if self.args.policy == 'play':
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
        #print(self.recording,len(self.dataset))
        if self.recording or self.args.policy == 'model':
            nlv = self.cup_body.velocity
            nav = self.cup_body.angular_velocity
            self.dataset.append(self.get_state() + [nlv[0]-olv[0], nlv[1]-olv[1], nav-oav])
        err_vec = (self.get_state()[3:]-np.squeeze(self.goal_state)[3:])#/self.cost_std[3:-3]
        err_vec /= self.cost_std[3:-3]
        state_err = np.linalg.norm(err_vec)
        #state_err = np.linalg.norm([err_vec[0],err_vec[1],err_vec[6],err_vec[7],err_vec[12],err_vec[13]])
        #print(state_err)
        self.space.reindex_shapes_for_body(self.cup_body)
        self.space.iterations = 25

        fps = 30.
        dt = (1/fps)
        steps = 5
        for _ in range(steps):
            #self.cup_body.apply_force_at_world_point(cup_body_reverse_gravity,cup_cog_world)
            self.space.step(dt/steps)
        if self.drawing:
            self.draw()

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
    parser.add_argument("--imitate-file", default="data/imitate_0.csv",
                        help="file to imitate")
    parser.add_argument("--model", type=str,default='model.pkl',
                        help="modefile to evaluate")
    parser.add_argument('policy', nargs='?', default='play',choices=['play','cma','de','model','replay','opt','pg','rrt'])
    args = parser.parse_args()
    main(args)
