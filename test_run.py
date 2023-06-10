import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import os
import time
import arguments
import sumo_env


args = arguments.get_args()
seed = args.seed

layout = {
    "Test Results": {
        "Experiment/speed": ["Multiline", ["speed/current_speed", "speed/speed_limit", "speed/target_speed"]],
        "Experiment/acceleration": ["Multiline", ["acceleration/current_acceleration"]]
    },
}
# sys.stderr.write("\x1b[2J\x1b[H")
writer = SummaryWriter()
writer.add_custom_scalars(layout)

# env = sumo_env.SumoEnv(gui_f=True, config="road_network/data4/quickstart.sumocfg", netPath="road_network/data4/quickstart.net.xml", speed_mode=0)
env = sumo_env.SumoEnv(gui_f=True, config="road_network/data5/osm.sumocfg", netPath="road_network/data5/osm.net.xml.gz", speed_mode=0, typeId='veh_passenger')
env.seed(seed=seed)
env.action_space.seed(seed=seed)

np.random.seed(seed)

# ckpt_path = f'out/sumo/id_3/seed2.tar'
ckpt_path = args.ckpt_path
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_checkpoint(ckpt_path=ckpt_path, evaluate=True)

done = False



def do_visualization(record=False, save_dname=None, writer=None):
    state = env.reset()
    if record:
        from visualize import Visualize
        import os
        from time import time
        save_dname = os.path.join(save_dname,
                                    f'videos/{args.test_id}')
        if not os.path.exists(save_dname):
            os.makedirs(save_dname)
        env.enable_viz(f'{save_dname}/snaps')
        viz = Visualize(save_name=save_dname, test_id = args.test_id)
        env.track_veh()
    episode_return = 0
    while True:
        action = agent.select_action(state=state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        episode_return += reward

        state = next_state
        # env.render()
        current_speed = env.get_speed()
        current_speed_limit = env.get_current_speed_limit()
        target_speed = env.get_target_speed()
        current_acceleration = env.get_acceleration()

        if writer:
            writer.add_scalar('speed/current_speed', current_speed, info['time'])
            writer.add_scalar('speed/speed_limit', current_speed_limit, info['time'])
            writer.add_scalar('speed/target_speed', target_speed, info['time'])
            writer.add_scalar('acceleration/current_acceleration', current_acceleration, info['time'])
        
        if done:
            if record:
                telemetry = env.get_telemetry()
                viz.set_history(telemetry)
                viz.overlay()
                viz.save()
            np.save(os.path.join(save_dname, f'telemetry.npy'), telemetry)
            # state = env.reset()
            break
    env.close()

# while not done:
#      action = agent.select_action(state=state, evaluate=True)
#      next_state, reward, done, _ = env.step(action) # Step
#      state = next_state

def main():
    
    print('started')
   
    save_path = os.path.join(os.path.dirname(__file__),'test_runs')
    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    do_visualization(writer=writer,record=True, save_dname=save_path)
    writer.close()
    return None


if __name__ == '__main__':
    main()