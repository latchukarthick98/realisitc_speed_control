from multiprocessing import set_forkserver_preload
import os
from random import random, randrange,choice
import sys
from traceback import print_tb
from turtle import done
from typing_extensions import Self
import numpy as np
from torch import rand
import sumolib
from gym import spaces
from gym.utils import seeding
import logging

G = 9.81  # gravity
FR = 0.015  # friction
RHO = 1.2  # density air, kg/m³
M_S_MPH = 0.44704  # 1 mph = 0.44704 m/s
KM_H_M_S = 3.6  # 1 m/s = 3.6 km/h

logging.basicConfig(level=logging.DEBUG)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
# logging.setLevel(logging.INFO)
class SumoEnvSafe:
    place_len = 7.5
    place_offset = 8.50
    lane_len = 10
    lane_ids = ['E8', '-E8', 'E4', 'E5', 'E9', 'E13', 'E12']
    # dis_map = {
    #     'E8': 295,
    #     '-E8': 295,
    #     'E4': 395,
    #     'E5': 495,
    #     'E9': 295,
    #     'E13': 295,
    #     'E12': 295
    # }

    
    def __init__(self, label='default', gui_f=False, agent_id = "veh0", current_target = "E9", config="data4/quickstart.sumocfg", netPath = "data4/quickstart.net.xml", speed_mode = 31, typeId = 'CarA', time_factor=0.1, disregardTime = False,
                 TTC_threshold = 4.001, route="", terminationType="destination", targetDistance = 1000):
        self.label = label
        self.wt_last = 0.
        self.ncars = 0
        self.time_target = 12 + (randrange(13) * 100)
        self.distance_to_goal = 0
        self.distance_left = 0
        self.agent_id = agent_id
        self.current_target = current_target
        self.time_pass = 0
        self.elapsed_time_ratio = 0
        self.target_speed = 0
        self.real_speed = 0
        self.instantaneous_speed = 0
        self.count = 0
        self.done = False
        self.reward = 0
        self.velocity = 0
        self.acceleration = 0
        self.prev_acceleration = 0
        self.tls = ()
        self.loopIds = []
        self.config = config 
        self.reachable = set()
        self.dis_map = {}
        self.net = sumolib.net.readNet(netPath)
        self.edges = self.net.getEdges()
        self.speed_mode = speed_mode
        self.specs = {}
        self.jerk = 0.0
        # self.dt = 0.1  # in s
        self.dt = time_factor
        self.reward_weights = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0] # weights of reward_forward, reward_jerk, reward_shock, reward_speed_limit, reward_TTC, reward_headway
        self.specs = self.get_vehicle_spec(self.agent_id)
        self.init = 0
        self.typeId = typeId
        self.current_speed_limit = 0
        self.safe_speed = 0
        self.future_speed_limits = []
        self.future_speed_limits_distances = []
        self.current_route = []
        self.sensor_range = 150.0
        self.steps = 0

        self.leader = None
        self.follower = None
        self.leader_speed = 0
        self.leader_gap = self.sensor_range
        self.current_gap = self.sensor_range
        self.follower_gap = self.sensor_range
        self.collisions = None
        self.isCollided = False
        self.min_gap = 25 # (in meters) minimum gap between vehicles
        self.desired_gap = 25
        self.isTLSRed = False
        self.signal_dis = self.sensor_range
        self.desired_speed = 0
        self.speed_threshold = 10
        self.disregardTime = disregardTime

        self.route = route

        self.termination_type = terminationType
        self.target_distance = targetDistance
        #TTC
        self.rel_speed = 0
        self.TTC = -1
        self.headway = 1.2
        self.TTC_threshold = TTC_threshold
        self._max_episode_steps = 1000
        self.dsafe = 0
        self.drac = 0
        self.rtime = 1 # reaction time
        self.telemetry = {
            'acc': [],
            'vel': [],
            'steps': [],
            'speed_lim': [],
            'rewards': [],
            'desired_speed': [],
            'desired_gap': [],
            'leader_speed': [],
            'leader_gap': [],
            'current_gap': [],
            'jerk': [],
            'headway': [],
            'ttc': [],
            'acc_traci': []
        }

        self.vizualize = False
        self.viz_path = None
        self.state_max = np.hstack(
            (self.specs['velocity_limits'][1],
             self.specs['velocity_limits'][1],
             self.specs['acceleration_limits'][1],
            #  self.specs['velocity_limits'][1],
             self.specs['velocity_limits'][1] * np.ones(2),
             self.sensor_range * np.ones(2),
            #  1,
             self.sensor_range,
             self.specs['velocity_limits'][1],
            #  self.specs['velocity_limits'][1],
            #  self.specs['velocity_limits'][1],
            #  self.sensor_range,
            #  1

            #  100000
            ))
        self.state_min = np.hstack(
            (self.specs['velocity_limits'][0],
             self.specs['velocity_limits'][0],
             self.specs['acceleration_limits'][0],
            #  self.specs['velocity_limits'][0],
             self.specs['velocity_limits'][0] * np.ones(2),
             np.zeros(2),
            #  0,
             0,
            #  -self.specs['velocity_limits'][1],
             0,
            #  self.specs['velocity_limits'][0],
            #  0,
            #  0
            #  0.0
            ))

        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(1, ),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=self.state_min,
                                            high=self.state_max,
                                            dtype=np.float64)

        '''
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
		'''
        #exe = 'sumo-gui.exe' if gui_f else 'sumo.exe'
        exe = 'sumo-gui' if gui_f else 'sumo'
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', exe)
        #sumoBinary = checkBinary('sumo')
        self.sumoCmd = [sumoBinary, 
                        '-c', self.config,
                        '--step-length', str(self.dt),
                        '--default.action-step-length', '0.1',
                        '--collision.stoptime', '0.1',
                        '--fcd-output.distance',
                        '--fcd-output.acceleration',
                        '--fcd-output.max-leader-distance', '150',
                        '--fcd-output', './dumps/op.xml'
                        ]

        return

    def enable_viz(self, path):
        self.vizualize = True
        self.viz_path = path
    def disable_viz(self):
        self.vizualize = False
    def track_veh(self):
        traci.gui.track(objID=self.agent_id)
        traci.gui.setZoom(viewID='View #0', zoom=1000.0)
        return
    def get_telemetry(self):
        return self.telemetry
    def clear_telemetry(self):
        self.telemetry = {
            'acc': [],
            'acc_traci':[],
            'vel': [],
            'steps': [],
            'speed_lim': [],
            'rewards':[],
            'desired_speed': [],
            'desired_gap': [],
            'leader_speed': [],
            'leader_gap': [],
            'current_gap': [],
            'jerk': [],
            'headway': [],
            'ttc': []
        }
    def calc_acceleration_from_power(self, velocity, power):
        """
        Physically calculates the corresonding acceleration for a specific
        power at a specific velocity.
        :param vel: velocity
        :param P: power
        :return: acceleration
        """
        # TODO: make formular readable - comment!
        acceleration = (power / (velocity * self.specs['mass']) * 1000 - (
            self.specs['cW'] * self.specs['frontal_area'] * RHO * velocity**2 /
            (2 * self.specs['mass']) + FR * G))
        return acceleration
    
    def action_scaling_vecs(self):
        vel_vec = np.arange(1, self.specs['velocity_limits'][1] + 1, 1)

        acc_pos_vec = self.calc_acceleration_from_power(
            vel_vec, self.specs['power_limits'][1])
        acc_neg_vec = self.calc_acceleration_from_power(
            vel_vec, self.specs['power_limits'][0])
        acc_0_vec = self.calc_acceleration_from_power(vel_vec, 0)

        acc_pos_vec = np.min([
            acc_pos_vec,
            np.ones(len(acc_pos_vec)) * self.specs['acceleration_limits'][1]
        ],
                             axis=0)
        acc_neg_vec = np.max([
            acc_neg_vec,
            np.ones(len(acc_neg_vec)) * self.specs['acceleration_limits'][0]
        ],
                             axis=0)

        # TODO: Find better solution :)
        # This is kind of a workaround. Roman got the values for 0 from the
        # data, which seems difficult to implement here. So the added 1.0 in
        # acc_pos_vec is handcrafted.
        self.vel_vec = np.append(0, vel_vec)
        self.acc_pos_vec = np.append(1.0, acc_pos_vec)
        self.acc_neg_vec = np.append(0.0, acc_neg_vec)
        self.acc_0_vec = np.append(0.0, acc_0_vec)

    def get_vehicle_spec(self, agentId):
        # vType = traci.vehicle.getTypeID(agentId)
        return {
            # 'acceleration_limits': [traci.vehicle.getDecel(vType), traci.vehicle.getAccel(vType)],
            # 'velocity_limits': [0, traci.vehicle.getMaxSpeed(vType)],
            'acceleration_limits': [-3, 3],
            # 'acceleration_limits': [-4.5, 2.6],
            'velocity_limits': [0, 50],
            'mass': 1443,
            'frontal_area': 2.38,
            'cW': 0.29,
            'power_limits': [-50, 75]
        }
    def get_acceleration_from_action(self, velocity, action):
        action_min = np.interp(velocity,
                               self.vel_vec,
                               self.acc_neg_vec,
                               left=0,
                               right=-1e-6)
        action_0 = np.interp(velocity,
                             self.vel_vec,
                             self.acc_0_vec,
                             left=0,
                             right=-1e-6)
        action_max = np.interp(velocity,
                               self.vel_vec,
                               self.acc_pos_vec,
                               left=0.6258544444444445,
                               right=-1e-6)

        action_lim = action_max if action > 0 else action_min
        acceleration = (action_lim - action_0) * abs(action) + action_0
        return acceleration

    def dfs(self, root, dist):
        if root is None:
            return
        if root in self.reachable:
            return
        curr_edge = self.net.getEdge(root)
        nextEdges = curr_edge.getOutgoing()
        # print(nextEdges)
        length = curr_edge.getLength()
        self.reachable.add(root)
        self.dis_map[root] = length + dist
        for edge in nextEdges:
            self.dfs(edge.getID(), length + dist)

    def getDetectorPos(self):
        # print(f"********** Detector POS ********")
        # print(traci.inductionloop.getIDCount())
        # print(self.loopIds)
        for id in self.loopIds:
            if traci.inductionloop.getLastStepVehicleNumber(id) > 0:
                vehIds = traci.inductionloop.getLastStepVehicleIDs(id)
                # print(f"************ Loop Id: {id} ********")
                # print(f"Veh Ids: {vehIds}")
                if self.agent_id in vehIds:
                    return id
        return None

        
        # print(f"***********************************")
 
    
    def get_state(self):
        #real_speed is not negative
        # return np.asarray([self.real_speed/10, self.target_speed/10, self.elapsed_time_ratio, self.distance_to_goal/100], dtype=np.float32)
        agentIsLead = 1 if self.leader is None else 0
        print(f'''Inside get state:\n {self.real_speed},{self.prev_acceleration},{self.safe_speed},\nSpeed Limits: {self.future_speed_limits},{self.future_speed_limits_distances}
                \nTLS: {self.isTLSRed}, {self.signal_dis} \nLeader: Speed: {self.leader_speed}, gap: {self.leader_gap}, agentIsLead: {agentIsLead}''')
        return np.hstack(
            (self.velocity,
             self.desired_speed,
             self.prev_acceleration,
            #  self.current_speed_limit, 
             self.future_speed_limits,
             self.future_speed_limits_distances,
            #  1 if self.isTLSRed else 0, # TLS status
            #  self.leader_gap,
             self.current_gap,
             self.rel_speed,
            #  self.leader_speed,
            #  self.leader_speed,
            #  self.signal_dis,
            #  agentIsLead, # If agent is leading
            #  self.distance_to_goal
            )
        )

    def get_real_speed(self):
        return traci.vehicle.getSpeed(self.agent_id)
    def update_agent_speed(self, acceleration):
        logging.debug(f'Acceleration change: {acceleration} m/s')
        curr_speed =  traci.vehicle.getSpeedWithoutTraCI(self.agent_id)
        new_speed = curr_speed + acceleration
        traci.vehicle.setSpeed(self.agent_id, speed=new_speed)
        # traci.vehicle.slowDown(self.agent_id, new_speed, 1)
        # traci.vehicle.moveToXY()
        # traci.vehicle.setSpeed(self.agent_id, acceleration)
        # current = traci.vehicle.getAcceleration(self.agent_id)
        # traci.vehicle.setAcceleration(vehID=self.agent_id,acceleration=acceleration, duration=10)
        return

    def get_agent_speed(self):
        return traci.vehicle.getSpeed(self.agent_id)

    
    def get_surrounding(self):
        return self.leader, self.follower, self.collisions
    
    def lmap(self, v, x, y):
        """Linear map of value v with range x to desired range y."""
        return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    
    def get_reward(self):
        # # safe_speed = min(self.target_speed, self.current_speed_limit)
        # safe_speed = self.current_speed_limit

        reward_forward = abs(self.velocity - self.desired_speed)
        # desired_speed = speed of vehicle in front if leader
        forward_max = self.desired_speed if self.desired_speed > 0 else 1
        # forward_max = self.specs['velocity_limits'][1]
        reward_forward /= forward_max
        reward_forward = pow(reward_forward, 2)
        reward_forward = self.lmap(reward_forward, [1,0], [0,1])

        reward_speed_limit = abs(self.velocity - self.current_speed_limit)
        reward_speed_limit /= self.current_speed_limit
        reward_speed_limit = pow(reward_speed_limit, 2)
        reward_speed_limit = self.lmap(reward_speed_limit, [1,0], [0, 1])
        
        reward_jerk = self.jerk
        jerk_max = np.diff(self.specs['acceleration_limits'])[0] / self.dt
        reward_jerk /= jerk_max

        reward_shock = 1 if self.velocity > self.desired_speed else 0
        
        # TTC reward calculation
        if self.leader is not None:
            cur_speed = 0.00001 if self.velocity <= 0 else self.velocity
            self.rel_speed = self.leader_speed - cur_speed
            relative_speed = 0.1 if self.rel_speed == 0 else self.rel_speed
            gap = self.leader_gap
            headway = self.leader_gap / cur_speed
            self.headway = headway
            # self.rel_speed = 0.0001 if self.rel_speed == 0 else self.rel_speed
            self.TTC = -gap / relative_speed
            
        else:
            self.TTC = -1
            headway = None
            self.headway = 1.2
            self.rel_speed = 0
            
        reward_TTC = 0

        if self.TTC >=0 and self.TTC <= self.TTC_threshold:
            reward_TTC = np.log(self.TTC / self.TTC_threshold)
            # reward_TTC = (self.TTC - self.TTC_threshold)
            # reward_TTC = -pow(reward_TTC, 2)

        # headway reward
        mu = 0.422618  
        sigma = 0.43659
        if headway is None:
            reward_headway = 0
        elif headway <= 0:
            reward_headway = -1
        else:
            reward_headway = (np.exp(-(np.log(headway) - mu) ** 2 / (2 * sigma ** 2)) / (headway * sigma * np.sqrt(2 * np.pi)))

        reward_forward = self.lmap(reward_forward, [0, 1], [0, 0.6])
        reward_headway = np.clip(reward_headway, -1, 0.65)
        reward_headway = self.lmap(reward_headway, [0, 0.65], [0, 0.4])
        # reward_TTC = np.clip(reward_TTC, -4, 0)
        reward_TTC = self.lmap(reward_TTC, [-4, 0], [-1, 0])
        print(f'TTC: {self.TTC}, Reward TTC: {reward_TTC}')
        print(f'Headway: {headway}, Reward headway: {reward_headway}')
        # reward for maintaining a safe distance between the leader vehicle
        # self.desired_gap = self.min_gap if self.leader_gap < self.min_gap else self.leader_gap
        reward_gap = abs(self.current_gap - self.desired_gap)
        reward_gap /= self.min_gap
        reward_gap = pow(reward_gap, 2)
        # reward_gap = 1 if self.current_gap < self.desired_gap else 0

        reward_list = [
            reward_forward, -reward_jerk, -reward_shock, reward_speed_limit, reward_TTC, reward_headway
        ]

        return reward_list

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def feature_scaling(self, state):
        """
        Min-Max-Scaler: scale X' = (X-Xmin) / (Xmax-Xmin)
        :param state:
        :return: scaled state
        """
        
        # print("Feature scaling Start................")
        # print(state)
        # print(state.shape)
        # print(self.state_min)
        # print(self.state_min.shape)
        # print(self.state_max)
        # print(self.state_max.shape)
        # print("End................")
        # try:
        return (state - self.state_min) / (self.state_max - self.state_min)
     
            # raise Exception("Sorry, no numbers below zero")
    def set_accel(self, acceleration):
        self.acceleration = acceleration

    def get_speed(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration

    def get_current_speed_limit(self):
        return self.current_speed_limit

    def get_target_speed(self):
        return self.target_speed


    def sensor(self):
        current_speed_limit = traci.vehicle.getAllowedSpeed(self.agent_id)
        traffic_phase = self.tls[0][3] if len(self.tls) > 0 else 'G'
        self.signal_dis = self.sensor_range
        self.current_gap = self.sensor_range
        if len(self.tls) > 0:
            self.signal_dis = self.tls[0][2] if self.tls[0][2] < self.sensor_range else self.sensor_range
            # self.current_gap = self.signal_dis

        # if (len(self.tls) > 0) and self.tls[0][3] == 'r':
        #     self.isTLSRed = True
        #     isStopRed = traci.vehicle.isStopped(self.agent_id)
        # else:
        #     self.isTLSRed = False
        
        self.isTLSRed = False
        self.desired_gap =  self.sensor_range
        self.leader_speed = 0
        leader_decel = self.specs['acceleration_limits'][1]
        # self.dsafe = self.sensor_range
        # when there is a vehicle in front
        if self.leader is not None:
            leader_id = self.leader[0]
            self.leader_speed = traci.vehicle.getSpeed(leader_id)
            self.current_gap = self.leader_gap if self.leader_gap < self.sensor_range else self.sensor_range
            # self.current_gap = self.leader_gap
            gap = self.leader_gap if self.leader_gap > 0 else 0.1
            self.drac = 0.5 * (pow(self.leader_speed - self.velocity, 2)) / gap
            leader_decel = traci.vehicle.getDecel(leader_id)
            if self.leader_gap < self.min_gap:
                self.desired_speed = self.leader_speed
                self.desired_gap = self.min_gap
            else:
                self.desired_speed = self.current_speed_limit + self.speed_threshold # make the desired speed a little above the speed limit
                self.desired_gap = self.leader_gap if self.leader_gap < self.sensor_range else self.sensor_range
        else:
            self.drac = 0
            self.leader_speed = self.velocity
            match traffic_phase:
                case 'r':
                    self.desired_speed = 0 if self.signal_dis < self.min_gap else self.desired_speed
                    self.desired_gap = self.min_gap if self.signal_dis < self.min_gap else self.signal_dis
                    # current_speed_limit = 0 if self.signal_dis < self.min_gap else self.current_speed_limit
                    self.current_gap = self.signal_dis
                    self.isTLSRed = True
                case 'G' | 'g':
                    self.desired_speed = self.current_speed_limit + self.speed_threshold
                case _:
                    self.desired_speed = self.current_speed_limit + self.speed_threshold

        print(f'Desired Gap: {self.desired_gap}, Current gap: {self.current_gap}')
        print(f'Desired speed: {self.desired_speed}, Speed Limit: {self.current_speed_limit}, Current speed: {self.velocity}')

        print(f'TLS: {self.tls}')
        self.desired_speed = min(self.desired_speed, self.specs['velocity_limits'][1])
        # safe distance calculation'
        self.dsafe = (self.velocity * self.rtime) + (pow(self.velocity, 2) / (2 * (-self.specs['acceleration_limits'][0]))) - (pow(self.leader_speed, 2) / (2 * leader_decel))
        # self.dsafe = min(self.dsafe, self.sensor_range)
        # self.dsafe += 10
        self.dsafe = np.clip(self.dsafe, 2.5, self.sensor_range)

        
    
        #old sensor data
        if(current_speed_limit <= 0 ):
            current_speed_limit = self.current_speed_limit
        current_edge_idx = traci.vehicle.getRouteIndex(self.agent_id)
        future_speed_limits = []
        future_speed_limits_distances = []
        next_2_edge_idx = self.current_route[current_edge_idx+1 : current_edge_idx+2]
        for i in next_2_edge_idx:
            speed_limit = traci.lane.getMaxSpeed(i+"_0")
            if(speed_limit <=0 ):
                speed_limit = current_speed_limit
            distance = traci.vehicle.getDrivingDistance(self.agent_id, i, 0)
            future_speed_limits.append(speed_limit)
            future_speed_limits_distances.append(min(self.sensor_range, distance))
        if(len(future_speed_limits) == 1):
            future_speed_limits.append(current_speed_limit)
            future_speed_limits_distances.append(self.sensor_range)
        elif(len(future_speed_limits) == 0):
            future_speed_limits = [current_speed_limit, current_speed_limit]
            future_speed_limits_distances = [self.sensor_range, self.sensor_range]

        return (current_speed_limit, future_speed_limits, future_speed_limits_distances)
    
    def step(self, action):
        try:
            if self.done == True:
                # self.close()
                self.reset()
                state = self.feature_scaling(self.get_state())
                return state, self.reward, self.done, self.time_pass
            
            # self.current_speed_limit = traci.vehicle.getAllowedSpeed(self.agent_id)
            # self.sensor()
            
            self.isCollided = False
            (self.current_speed_limit, self.future_speed_limits, 
            self.future_speed_limits_distances) = self.sensor()
            # safe_speed = min(self.target_speed, self.current_speed_limit)
            self.safe_speed = self.current_speed_limit
            action = np.clip(action, -1, 1)
            action_by_model = action
            if self.current_gap < self.dsafe:
                logging.debug(f'Correcting action, applying brakes')
                action = np.clip([-3.0] , -1, 1)
                action_by_model = action
            assert self.action_space.contains(action),\
                f'{action} ({type(action)}) invalid shape or bounds'
            self.acceleration = self.get_acceleration_from_action(
                self.velocity, action)[0]
            
            # disabling for training
            if self.leader_gap < self.dsafe:
                # self.drac = np.clip([self.drac], 0,6.0)[0]
                # self.acceleration = -self.drac
                # self.acceleration = self.specs['acceleration_limits'][0]
                # self.acceleration = -6.5
                self.acceleration = -4.5
                # s = 0
            # self.set_accel(action)
            # self.real_speed = traci.vehicle.getSpeedWithoutTraCI(self.agent_id)
            logging.debug(f'Real speed: {self.real_speed}')


            logging.debug(f"Acceleration: {self.acceleration}")

            logging.debug(f"instantaneous_speed: {self.instantaneous_speed}")

            # traci.vehicle.setAcceleration(self.agent_id, self.acceleration, self.dt)
            
            traci.simulationStep()
            self.acceleration = traci.vehicle.getAcceleration(self.agent_id)
            self.collisions = traci.simulation.getCollisions()

            if(len(self.collisions) > 0):
                matched = [x for x in self.collisions if x.collider == self.agent_id]
                collision = matched[0]
                victim = collision.victim
                if len(matched) > 0:
                    self.done = True
                    self.isCollided = True
                
                

            self.leader = traci.vehicle.getLeader(self.agent_id)
            self.follower = traci.vehicle.getFollower(self.agent_id)
            
            if(self.leader is not None):
                self.leader_gap = self.leader[1]
            else:
                self.leader_gap = self.sensor_range
            

            self.steps += 1
            self.distance_to_goal = traci.vehicle.getDrivingDistance(self.agent_id, self.current_target, 100)
            # if(self.distance_to_goal < 0) :
            #     traci.close()
            #     self.reset()
        
            
            self.real_speed = traci.vehicle.getSpeed(self.agent_id)
            self.velocity = self.real_speed
            # print(f'New speed: {new_speed}')
            logging.debug(f'Updated agent speed: {self.real_speed}')
        
            self.time_pass += self.dt
            self.jerk = abs((self.acceleration - self.prev_acceleration)) / self.dt
            self.prev_acceleration = self.acceleration
            logging.debug("Time pass: %d \t Time target: %d" % (self.time_pass, self.time_target))
            self.elapsed_time_ratio = self.time_pass / self.time_target
            logging.debug(f"Current target: {self.current_target}")

            logging.debug("Elapsed time ratio %d \n Time pass: %d" % (self.elapsed_time_ratio, self.time_pass))
            logging.debug("Distance to Goal left: %d" % (self.distance_to_goal))
            logging.debug(f"Safe distance: {self.dsafe}")
            self.tls = traci.vehicle.getNextTLS(self.agent_id)
            reward_list = self.get_reward()
            

        
            self.reward = np.array(reward_list).dot(np.array(self.reward_weights))

            # penalize heavily for collision irrespective of speed and other factors
            if(self.isCollided):
                self.reward -= 1

            if(self.termination_type == "destination"):
                if(self.distance_to_goal < 50):
                    self.done = True
                    self.reward += 1
            else:
                if(traci.vehicle.getDistance(self.agent_id) >= self.target_distance):
                    self.done = True
                    self.reward += 1

            # self.reward = np.clip(self.reward, -5000, 5000)
            self.reward = self.lmap(self.reward, [-1, 1], [0, 1])
            self.reward = np.clip(self.reward, 0, 1)

            state = self.feature_scaling(self.get_state())
            logging.debug(f'Reward: {self.reward}')
            logging.debug(f"Next TLS: {self.tls} , {len(self.tls)}")
            logging.debug(f"Done: {self.done}")
            # minTTC = traci.vehicle.getParameter(self.agent_id, "device.ssm.minTTC")
            # print(f'Min TTC: {minTTC}, type:{type(minTTC)}')
            # if(len(self.tls) > 0):
            #     print(self.tls[1])
            # state = np.asarray([self.real_speed/10, self.target_speed/10, self.elapsed_time_ratio, self.distance_to_goal/100], dtype=np.float32)

            # detectorId = self.getDetectorPos()
            detectorId = None
            # mark done if hit the time horizon
            if self.disregardTime == False and self.steps >= self._max_episode_steps:
                self.done = True
            
            #sample video frame every second
            if self.steps % 10 == 0:
                if self.vizualize:
                    img_path = f'{self.viz_path}/{self.steps//10}.png'
                    traci.gui.screenshot(viewID='View #0', filename=img_path)
                
                self.telemetry['acc'].append(self.acceleration)
                self.telemetry['vel'].append(self.velocity)
                self.telemetry['speed_lim'].append(self.current_speed_limit)
                self.telemetry['steps'].append(self.steps//10)
                self.telemetry['rewards'].append(self.reward)
                self.telemetry['desired_speed'].append(self.desired_speed)
                self.telemetry['desired_gap'].append(self.desired_gap)
                self.telemetry['leader_speed'].append(self.leader_speed)
                self.telemetry['leader_gap'].append(self.leader_gap)
                self.telemetry['current_gap'].append(self.current_gap)
                self.telemetry['jerk'].append(self.jerk)
                self.telemetry['headway'].append(self.headway)
                self.telemetry['ttc'].append(self.TTC)
            
            self.telemetry['acc_traci'].append(self.acceleration)

            info = {
                'position': self.distance_to_goal,
                # 'velocity': self.velocity * 2.23694,
                'velocity': self.velocity,
                'acceleration': self.acceleration,
                'jerk': self.jerk,
                'time': self.time_pass,
                'detectorId': detectorId,
                'lead_veh_gap': self.leader_gap,
                'is_collided': self.isCollided,
                'action': action_by_model
            }
            return state, self.reward, self.done, info

        except:
            logging.error('Exception')
            self.done = True
            info = {
                'position': self.distance_to_goal,
                # 'velocity': self.velocity * 2.23694,
                'velocity': self.velocity,
                'acceleration': self.acceleration,
                'jerk': self.jerk,
                'time': self.time_pass,
                'detectorId': None,
                'lead_veh_gap': self.leader,
                'is_collided': self.isCollided,
                'action': action_by_model
            }
            # self.reset()
            state = self.feature_scaling(self.get_state())
            return state, self.reward, self.done, info

    def reset(self):
        self.wt_last = 0.
        self.ncars = 0
        self.time_pass = 0
        # self.count = 0
        self.done = False
        self.real_speed = 5
        self.instantaneous_speed = 5
        self.reward = 0

        self.steps = 0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.jerk = 0.0
        self.prev_acceleration = 0.0
        self.dis_map.clear()
        self.reachable.clear()
        self.current_target = 'E9'


        self.leader = None
        self.follower = None
        self.leader_speed = self.specs['velocity_limits'][1]
        self.current_gap = self.sensor_range
        self.leader_gap = self.sensor_range
        self.follower_gap = self.sensor_range
        self.collisions = None
        self.isCollided = False
        self.desired_gap = 25
        self.isTLSRed = False
        self.signal_dis = self.sensor_range
        self.desired_speed = 0
        self.TTC = -1
        self.rel_speed = 0
        self.dsafe = 0
        self.drac = 0
        # commenting for testing
        self.clear_telemetry()
        logging.debug("Before reset")
        if self.init == 0:
            traci.start(self.sumoCmd, label=self.label, traceFile="./debug/traci.log")
            # traci.init(port=19080, traceFile="./debug/traci.log")
            # if(traci.isLoaded()):
            #     print("Connected")
            #     traci.load(['-c', self.config,
            #                 '--step-length', '0.1',
            #                 '--default.action-step-length', '0.1'
            #                 ])
            self.init = 1
        else:
            traci.load(['-c', self.config,
                        '--step-length', str(self.dt),
                        '--default.action-step-length', '0.1',
                        '--collision.stoptime', '0.1',
                        '--fcd-output.distance',
                        '--fcd-output.acceleration',
                        '--fcd-output.max-leader-distance', '150',
                        '--fcd-output', './dumps/op.xml'
                        ])
        
        # traci.trafficlight.setProgram('gneJ00', '0')
        logging.debug("Reset:\n")
        
        self.loopIds = traci.inductionloop.getIDList()
        # self.edges = traci.edge.getIDList()
        
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()

        
        # print(f'Random: {self.np_random.random()}')
        rand_step = int(self.np_random.rand(1)[0]*1000)
        # rand_step = int(randrange(1,500,1))
        # print(f'Rand: {self.np_random.rand()}')
        # rand_step = 1000
        print(f'Rand step: {rand_step}')
        traci.simulationStep(rand_step)

        
        # for i in range(1000):
        #     # traci.vehicle.add(vehID=f"lead_car_{i}",routeID="", typeID=self.typeId)
        #     traci.simulationStep()
        # typeID="passenger"
        # traci.vehicle.add(vehID=f"lead_car_1",routeID="", departSpeed='0',typeID="CarB")
        # traci.simulationStep(10)

        traci.vehicle.add(vehID=self.agent_id, routeID=self.route, departSpeed='0', typeID=self.typeId)
        traci.simulationStep()

        
        
        veh_lane = traci.vehicle.getLaneID(self.agent_id)

        while veh_lane == '':
            veh_lane = traci.vehicle.getLaneID(self.agent_id)
            traci.simulationStep()

        self.current_source = veh_lane.split("_")[0]
        logging.debug(f"Current Source: {self.current_source}, lane: {veh_lane}")
        self.dfs(self.current_source, 0)
        # print(f"Edges: {self.edges} \n")
 
        self.current_target = choice(tuple(self.reachable))
        # traci.vehicle.changeTarget(self.agent_id, self.current_target)
        self.current_route = traci.vehicle.getRoute(self.agent_id)
        self.current_target = self.current_route[-1]
        self.action_scaling_vecs()
        # traci.vehicle.setSpeedMode(self.agent_id, 32)
        # traci.vehicle.setSpeedMode(self.agent_id, 0)
        traci.vehicle.setSpeedMode(self.agent_id, self.speed_mode)
        dis = traci.lane.getLength(self.current_target + "_0")
        self.distance_to_goal = traci.vehicle.getDrivingDistance(self.agent_id, self.current_target,dis)
        logging.debug("Distance to goal for %s : %d" % (self.current_target, self.distance_to_goal))
        self.time_target = int((self.distance_to_goal/100) * 5) + (randrange(3))
        self.target_speed = self.distance_to_goal / self.time_target
        # self.real_speed = traci.vehicle.getSpeed(self.agent_id)
        self.real_speed = traci.vehicle.getSpeedWithoutTraCI(self.agent_id)
        self.instantaneous_speed = self.real_speed
        # self.sensor()
        (self.current_speed_limit, self.future_speed_limits, 
         self.future_speed_limits_distances) = self.sensor()
        self.safe_speed = self.current_speed_limit
        logging.debug("Target speed: %d" % (self.target_speed))
        state = self.feature_scaling(self.get_state())
        
        return state
        # return self.get_state_d()

    def close(self):
        traci.close()
        return None
