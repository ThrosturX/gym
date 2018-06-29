# -*- coding: utf-8 -*-
"""
@author: Jordi Bieger

Simple Arrival Control environment, adapted from /classic_control/continuous_mountain_car.py
"""

import math
import gym
import gym.spaces
from gym.utils import seeding
import numpy as np

class ACSimpleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.nr_aircraft = 5
        self.separation = 100 # seconds
        self.dt = 1 # seconds
        self.time_step = 0

        self.viewer = None

        self.action_space = gym.spaces.MultiDiscrete([self.nr_aircraft,2,2]) # aircraft id, speed adjustment true/false, speed increase/decrease
        aircraft_observation_lo = np.repeat([0, 0, 0, -10000], self.nr_aircraft) # aircraft id, distance, speed, arrival time
        aircraft_observation_hi = np.repeat([self.nr_aircraft-1, 2000000, 340, 10000], self.nr_aircraft) # aircraft id, distance, speed, arrival time
        self.observation_space = gym.spaces.Box(aircraft_observation_lo, aircraft_observation_hi, dtype=np.float32)

        self.seed(1)
        self.target_time = self.np_random.uniform(low=500, high=1000)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = 0

        for i in range(self.nr_aircraft):
            if action[0] == i and action[1] != 0: # Plane==i and has some action
                reward -= 1
                if action[2] > 0:
                    speed_change = 1
                    color = Color.GEEN
                else:
                    speed_change = -1
                    color = Color.RED
                self.aircraft[i].adjust_speed(speed_change)
                self.ttas[i] = self.aircraft[i].tta
                self.order = np.argsort(self.ttas)
            else:
                color = Color.BLACK
            self.aircraft[i].color = color

        for i in range(self.nr_aircraft):
            j = self.order[i]
            
            if self.aircraft[j].update(self.dt): # airplane landing
                # The planes must not land too soon after each other
                sep = abs(self.last_landing - self.time_step)
                if sep < self.separation: # separation violated
                    reward -= 100 # todo: make larger errors weigh heavier
                else:
                    reward += 10
                self.last_landing = self.time_step
            
            self.ttas[j] = self.aircraft[j].tta
            self.state[i*4+0] = self.aircraft[j].id
            self.state[i*4+1] = self.aircraft[j].distance
            self.state[i*4+2] = self.aircraft[j].velocity
            self.state[i*4+3] = self.aircraft[j].tta
        
        done = self.state[self.nr_aircraft*4-1] <= 0
        self.time_step += 1

        return self.state, reward, done, {}

    def reset(self, target_time=None):
        self.last_landing = -self.separation
        self.aircraft = np.empty(self.nr_aircraft, dtype=object)
        self.ttas = np.empty(self.nr_aircraft)
        for i in range(0, self.nr_aircraft):
            self.aircraft[i] = Aircraft(i)
            self.aircraft[i].distance = self.np_random.uniform(low=(i+1)*100000, high=(i+3)*100000)
            if self.target_time is None:
                self.aircraft[i].velocity = self.np_random.uniform(low=100, high=340)
            else:
                self.aircraft[i].velocity = self.aircraft[i].distance / self.target_time
            self.aircraft[i].tta = self.aircraft[i].seconds_to_arrival()
            self.ttas[i] = self.aircraft[i].tta
            
        self.order = np.argsort(self.ttas)
        self.state = np.empty((self.nr_aircraft*4))
        for i in range(0, self.nr_aircraft):
            j = self.order[i]
            self.state[i*4+0] = self.aircraft[j].id
            self.state[i*4+1] = self.aircraft[j].distance
            self.state[i*4+2] = self.aircraft[j].velocity
            self.state[i*4+3] = self.aircraft[j].tta
        
        return np.array(self.state)


    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 1000000 # meters distance
        xscale = screen_width/world_width
        world_height = 20000 # seconds to arrival
        yscale = screen_height/world_height


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            self.viewer.add_geom(rendering.Line((0, screen_height/2), (screen_width, screen_height/2)))
            
            self.icons = np.empty(self.nr_aircraft, dtype=object)
            self.icontrans = np.empty(self.nr_aircraft, dtype=object)
            for i in range(0, self.nr_aircraft):
                transform = rendering.Transform()
                icon = rendering.make_circle(2)
                icon.add_attr(transform)
                self.viewer.add_geom(icon)

                self.icons[i] = icon
                self.icontrans[i] = transform

        for i in range(0, self.nr_aircraft):
            self.icons[i].set_color(*self.aircraft[i].color)
            self.icontrans[i].set_translation(self.aircraft[i].distance*xscale, ((self.aircraft[i].distance / 100) + world_height/2) * yscale)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

class ACSimpleSolver():
    def __init__(self, env):
        self.predictor = ACSimple_PredictArrivalSolver(env)
        self.detector = ACSimple_CheckConflictSolver(env)
        self.nr_aircraft = env.unwrapped.nr_aircraft
        self.seed(env.unwrapped.np_random.tomaxint())
        return
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, observation, reward, done):
        nr_inputs = 3
        self.aircraft = np.empty(self.nr_aircraft, dtype=object)
        for i in range(0, self.nr_aircraft):
            self.aircraft[i] = Aircraft(observation[4*i+0])
            self.aircraft[i].distance = observation[4*i+1]
            self.aircraft[i].velocity = observation[4*i+2]
            if nr_inputs == 4:
                self.aircraft[i].tta = observation[4*i+3]
            else:
                self.aircraft[i].tta = self.predictor.step([self.aircraft[i].distance, self.aircraft[i].velocity], 0, False)
            if i > 0 and self.aircraft[i].tta > 0 and self.detector.step([self.aircraft[i-1].tta, self.aircraft[i].tta], 0, False):
                if self.np_random.uniform() < 0.5: # TODO: fix greedy algorithm
                    return [self.aircraft[i-1].id, 1] # make first aircraft go faster
                else:
                    return [self.aircraft[i].id, -1] # make second aircraft go slower
        
        return [0, 0]
        
class ACSimple_PredictArrivalEnv(gym.Env): # in: (distance, speed), out: tta
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.viewer = None

        self.action_space = gym.spaces.Box(-10000, 10000) # seconds to arrival
        aircraft_observation_lo = np.repeat([0, 0], 1) # id, distance, speed
        aircraft_observation_hi = np.repeat([2000000, 340], 1) # id, distance, speed
        self.observation_space = gym.spaces.Box(aircraft_observation_lo, aircraft_observation_hi)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = action[0] - self.aircraft.tta
        reward = -(reward*reward) # squared error
        
        done = True

        return self.state, reward, done, {}

    def reset(self):
        self.aircraft = Aircraft(0)
        self.aircraft.velocity = self.np_random.uniform(low=100, high=340)
        self.aircraft.distance = self.np_random.uniform(low=10, high=2000000)
        self.aircraft.tta = self.aircraft[i].seconds_to_arrival()
        
        self.state = np.array([self.aircraft.distance, self.aircraft.velocity])
        return np.array(self.state)

class ACSimple_PredictArrivalSolver():
    def __init__(self, env):
        return
        
    def step(self, observation, reward, done):
        distance, velocity = observation
        return distance / velocity
        
class ACSimple_CheckConflictEnv(gym.Env): # in: (tta, tta), out: yes/no
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.separation = 60
        self.viewer = None

        self.action_space = gym.spaces.Box(0, 1) # conflict or not
        aircraft_observation_lo = np.repeat([-10000], 2) # seconds to arrival
        aircraft_observation_hi = np.repeat([10000], 2) # seconds to arrival
        self.observation_space = gym.spaces.Box(aircraft_observation_lo, aircraft_observation_hi)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        conflict = abs(self.state[0] - self.state[1]) >= self.separation
        if conflict and action[0] == 0: # false negative
            reward = -10
        elif not conflict and action[0] == 1: # false positive
            reward = -1
        else:
            reward = 1
        
        done = True

        return self.state, reward, done, {}

    def reset(self):
        conflict = self.np_random.uniform() > 0.8
        if conflict:
            tta1 = self.np_random.uniform(low=-self.separation+1, high=9999-self.separation)
            tta2 = self.np_random.uniform(low=tta1, high=tta1+self.separation-1)
        else:
            tta1 = self.np_random.uniform(low=-self.separation+1, high=9999-self.separation)
        
        
        
        self.aircraft = Aircraft(0)
        self.aircraft.velocity = self.np_random.uniform(low=100, high=340)
        self.aircraft.distance = self.np_random.uniform(low=10, high=2000000)
        self.aircraft.tta = self.aircraft[i].seconds_to_arrival()
        
        self.state = np.array([self.aircraft.distance, self.aircraft.velocity])
        return np.array(self.state)

class ACSimple_CheckConflictSolver():
    def __init__(self, env):
        self.separation = env.unwrapped.separation
        
    def step(self, observation, reward, done):
        tta1, tta2 = observation
        print("{} - {} = {} <> {}".format(tta1, tta2, abs(tta1 - tta2), self.separation))
        
        return abs(tta1 - tta2) <= self.separation
        
class Aircraft():
    def __init__(self, id):
        self.id = id
        self.velocity = 0.0         # in m/s
        self.distance = 0.0         # in meters
        self.tta = 0.0              # time to arrival in seconds
        self.min_velocity = 100
        self.max_velocity = 340
        self.color = Color.BLACK    # Aircraft color (R, B, G, Transparency)
        
    def seconds_to_arrival(self):
        return self.distance / self.velocity
    
    def adjust_speed(self, adjustment):
        if self.distance <= 0:
            return
        if adjustment == -1:
            self.velocity = max(self.min_velocity, self.velocity*0.9)
        elif adjustment == 1:
            self.velocity = min(self.max_velocity, self.velocity*1.1)
        else:
            return
        self.tta = np.floor(self.seconds_to_arrival())
    
    def update(self, dt):
        self.tta -= dt
        if self.distance <= 0:
            return False
        self.distance = self.distance - dt*self.velocity
        if self.distance <= 0:
            self.distance = 0
            self.velocity = 0
            return True
        return False

class Color():
    RED = (1, 0, 0)
    GEEN = (0, 1, 0)
    BLACK = (0, 0, 0)