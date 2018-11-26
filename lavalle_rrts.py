#!/usr/bin/env python

# rrt.py
# This program generates a simple rapidly
# exploring random tree (RRT) in a rectangular region.
#
# Written by Steve LaValle
# May 2011

import sys, random, math, pygame
from pygame.locals import *
from math import sqrt,cos,sin,atan2
import heapq
import numpy as np

#constants
XDIM = 500
YDIM = 500
WINSIZE = np.array([XDIM, YDIM])
MAX_STEP_SIZE = 12
NUMNODES = 5000
NUM_OBSTACLES = 25
OBSTACLE_WIDTH = 100
OBSTACLE_HEIGHT = 100
RAND_SEARCH_PROB = 0.25
GOAL_TOL = 1e-3

start = WINSIZE/2
goal1 = np.zeros((1,2))
goal2 = WINSIZE.reshape((1,2))



def step_from_to(p1,p2):
    if np.linalg.norm(p1-p2) < MAX_STEP_SIZE:
        return p2
    else:
        diff = p2-p1
        return p1 + MAX_STEP_SIZE*diff/np.linalg.norm(diff)

def main():
    #initialize and prepare screen
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRT      S. LaValle    May 2011')
    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(black)

    obstacles = []
    for _ in range(NUM_OBSTACLES):
        rand_rect = np.random.rand(4)*np.array([XDIM,YDIM,OBSTACLE_WIDTH,OBSTACLE_HEIGHT]) + np.ones(4)*MAX_STEP_SIZE
        if (rand_rect[:2] < start).all() and (rand_rect[:2]+rand_rect[2:] > start).all():
            print('skip!')
            continue
        if (rand_rect[:2] < goal1).all() and (rand_rect[:2]+rand_rect[2:] > goal1).all():
            print('skip!')
            continue
        if (rand_rect[:2] < goal2).all() and (rand_rect[:2]+rand_rect[2:] > goal2).all():
            print('skip!')
            continue
        obstacles.append(rand_rect)
    for idx,o in enumerate(obstacles):
        weight = idx/(len(obstacles)-1)
        color = (240-240*weight,128,40+(255-40)*weight)
        screen.fill(color,o)

    nodes = np.array([start])[:np.newaxis]
    connections = np.array([0])
    print(nodes.shape,connections.shape)
    for goal in [goal1,goal2]:
        searching = True
        for i in range(NUMNODES):
            if searching:
                # get a random configuration
                #valid = False
                #while not valid:
                rand = np.random.rand(1,2)*WINSIZE if np.random.rand() > RAND_SEARCH_PROB else goal
                    #valid = True
                    #for o in obstacles:
                        #if (o[:2] < rand[0]).all() and (o[:2]+o[2:] > rand[0]).all():
                            #valid = False
                            #break

                dists = np.linalg.norm(nodes-rand,axis=1)
                #print(dists)
                closest_idx = np.argmin(dists)
                closest = nodes[closest_idx]
                new_node = step_from_to(closest,rand)
                valid_new_node = True
                for o in obstacles:
                    if (o[:2] < new_node[0]).all() and (o[:2]+o[2:] > new_node[0]).all():
                        valid_new_node = False
                        break
                if valid_new_node:
                    if np.linalg.norm(new_node - goal) > GOAL_TOL:
                        #print(goal,new_node)

                        nodes = np.append(nodes,new_node,0)
                        connections  = np.append(connections,closest_idx)
                        #print(np.linalg.norm(new_node - goal),nodes.shape,connections.shape)

                        pygame.draw.line(screen,white,np.squeeze(closest),np.squeeze(new_node))
                    else:
                        print(new_node,goal)
                        path_node = closest_idx
                        while path_node != 0:
                            print(path_node,end=',',flush=True)
                            path_node = connections[path_node]
                        print(0)
                        searching = False
                        break
            pygame.display.update()
            #print i, "    ", nodes

            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    sys.exit("Leaving because you requested it.")
        
    while True:
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Leaving because you requested it.")
# if python says run, then we should run
if __name__ == '__main__':
    main()

