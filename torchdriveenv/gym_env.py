import os
import logging
import math
import inspect
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import time
import torch
import gymnasium as gym
from invertedai.common import AgentState, Point, AgentAttributes, RecurrentState

from torchdrivesim.behavior.iai import IAIWrapper
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.rendering.base import RendererConfig
from torchdrivesim.utils import Resolution
from torchdrivesim.lanelet2 import find_lanelet_directions
from torchdrivesim.map import find_map_config, traffic_controls_from_map_config
from torchdrivesim.traffic_lights import current_light_state_tensor_from_controller
from torchdrivesim.simulator import TorchDriveConfig, SimulatorInterface, \
    BirdviewRecordingWrapper, Simulator, HomogeneousWrapper, CollisionMetric

from torchdriveenv.helpers import save_video, set_seeds
from torchdriveenv.iai import iai_conditional_initialize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EnvConfig:
    ego_only: bool = False
    max_environment_steps: int = 200
    frame_stack: int = 3
    waypoint_bonus: float = 100.
    heading_penalty: float = 25.
    distance_bonus: float = 1.
    distance_cutoff: float = 0.5
    use_background_traffic: bool = True
    terminated_at_infraction: bool = True
    seed: Optional[int] = None
    simulator: TorchDriveConfig = field(default_factory=lambda:TorchDriveConfig(renderer=RendererConfig(left_handed_coordinates=True,
                                                                           highlight_ego_vehicle=True),
                                                   collision_metric=CollisionMetric.nograd,
                                                   left_handed_coordinates=True))
    render_mode: Optional[str] = "rgb_array"
    video_filename: Optional[str] = "rendered_video.mp4"
    video_res: Optional[int] = 1024
    video_fov: Optional[float] = 500
    device: Optional[str] = None
    num_of_agents: Optional[list] = None
    num_of_agents_timestep: Optional[list] = None
    difficulty_sets: Optional[list] = None
    eval_mode: Optional[dict] = None
@dataclass
class Scenario:
    agent_states: List[List[float]] = None
    agent_attributes: List[List[float]] = None
    recurrent_states: List[List[float]] = None


@dataclass
class WaypointSuite:
    locations: List[str] = None
    waypoint_suite: List[List[List[float]]] = None
    car_sequence_suite: List[Optional[Dict[int, List[List[float]]]]] = None
    scenarios: List[Optional[Scenario]] = None


class GymEnv(gym.Env):

    metadata = {
        "render_modes": ["video", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self, cfg: EnvConfig, simulator: SimulatorInterface):
        if cfg.render_mode is not None and cfg.render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError
        self.render_mode = cfg.render_mode

        acceleration_range = (-1.0, 1.0)
        steering_range = (-0.3, 0.3)
        action_range = np.ndarray(shape=(2, 2), dtype=np.float32)
        action_range[:, 0] = acceleration_range
        action_range[:, 1] = steering_range
        self.max_environment_steps = cfg.max_environment_steps
        self.environment_steps = 0
        self.action_space = gym.spaces.Box(
            low=action_range[0],
            high=action_range[1],
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

        self.reward_range = (- float('inf'), float('inf'))
        self.collision_threshold = 0.0
        self.offroad_threshold = 0.0

        self.config = cfg
        self.simulator = simulator
        self.current_action = None

        self.last_birdview = None

    # TODO: use the seed
    # TODO: return the reset info
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.simulator = self.start_sim.copy()
        self.environment_steps = 0
        self.last_birdview = None
        return self.get_obs(), {}

    def step(self, action: np.array):
        self.environment_steps += 1
        self.simulator.step(action)
        self.last_action = self.current_action if self.current_action is not None else action
        self.current_action = action
        return self.get_obs(), self.get_reward(), self.is_terminated(), self.is_truncated(), self.get_info()

    def get_obs(self):
        birdview = self.simulator.render_egocentric().cpu().numpy().astype(np.uint8)
        return birdview

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        r = np.zeros(x.shape)
        return r

    def is_done(self):
         return self.is_truncated() or self.is_terminated()

    def is_truncated(self):
        return self.environment_steps >= self.max_environment_steps

    def is_terminated(self):
        return False

    def get_info(self):
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
        )
        return self.info

    def seed(self, seed=None):
        pass

    def render(self):
        if self.render_mode == 'rgb_array':
            birdview = self.simulator.render_egocentric().cpu().numpy()
            return np.transpose(birdview.squeeze(), axes=(1, 2, 0))
        else:
            raise NotImplementedError

    def mock_step(self):
        obs = np.zeros((1, 3, 64, 64)) # self.last_obs
        reward = 0
        terminated = False
        truncated = True
        info = dict(
            offroad=torch.Tensor([[0]]),
            collision=torch.Tensor([[0]]),
            traffic_light_violation=torch.Tensor([[0]]),
            is_success=False,
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        if isinstance(self.simulator, BirdviewRecordingWrapper):
            bvs = self.simulator.get_birdviews()
            if len(bvs) > 1:
                save_video(bvs, self.config.video_filename)


def build_simulator(cfg: EnvConfig, map_cfg, device, ego_state, scenario=None, car_sequences=None, waypointseq=None,num_agent_dict={}):
    # print("Entering build_simulator")
    with torch.no_grad():
        max_num_agents = num_agent_dict['max'] + 1 # +1 to consider ego vehicle
        min_num_agents = num_agent_dict['min'] + 1 # +1 to consider ego vehicle
        # print("build_simulator including ego, max_num_agents:", max_num_agents)
        # print("build_simulator including ego, min_num_agents: ", min_num_agents)
        # print(f" len(num_agent_dict['map_list_all_density_and_seeds']) {len(num_agent_dict['map_list_all_density_and_seeds'])}")
        traffic_light_controller = map_cfg.traffic_light_controller
        initial_light_state_name = traffic_light_controller.current_state_with_name
        traffic_light_ids = [stopline.actor_id for stopline in map_cfg.stoplines if stopline.agent_type == 'traffic_light']
        driving_surface_mesh = map_cfg.road_mesh


        traffic_controls = traffic_controls_from_map_config(map_cfg)
        traffic_controls = {key: traffic_controls[key] for key in traffic_controls}
        traffic_controls['traffic_light'].set_state(current_light_state_tensor_from_controller(traffic_light_controller, traffic_light_ids).unsqueeze(0))
        st = time.time()
        while True:
            st1 = time.time()
            if cfg.ego_only or max_num_agents == 1:
                agent_states = torch.Tensor([ego_state[0], ego_state[1], ego_state[2], ego_state[3]]).unsqueeze(0)
                length = np.random.random() * (5.5 - 4.8) + 4.8
                width = np.random.random() * (2.2 - 1.8) + 1.8
                rear_axis_offset = np.random.random() * (0.97 - 0.82) + 0.82
                agent_attributes = torch.Tensor([length, width, rear_axis_offset]).unsqueeze(0)
                recurrent_states = torch.Tensor([0] * 132).unsqueeze(0)
            else:
                if cfg.use_background_traffic:
                    background_traffic_dir = os.path.join(
                        os.path.dirname(os.path.realpath(
                            __file__)), f"resources/background_traffic")
                    
                    background_traffic_file_name = random.choice(num_agent_dict["map_list_all_density_and_seeds"])
                    background_traffic_file = os.path.join(background_traffic_dir, background_traffic_file_name)
                    with open(background_traffic_file, "r") as f:
                        background_traffic_json = json.load(f)
                    background_traffic = {}
                    background_traffic['location'] = background_traffic_json['location']
                    background_traffic['agent_density'] = background_traffic_json['agent_density']
                    background_traffic['random_seed'] = background_traffic_json['random_seed']
                    background_traffic['agent_states'] = [AgentState.model_validate(agent_state) for agent_state in background_traffic_json['agent_states']]
                    background_traffic['agent_attributes'] = [AgentAttributes.model_validate(agent_attribute) for agent_attribute in background_traffic_json['agent_attributes']]
                    background_traffic['recurrent_states'] = [RecurrentState.model_validate(recurrent_state) for recurrent_state in background_traffic_json['recurrent_states']]

                    if len(background_traffic["agent_states"]) + background_traffic["agent_density"] > 100:
                        continue

                    remain_agent_states = [AgentState(center=Point(x=ego_state[0], y=ego_state[1]), orientation=ego_state[2], speed=ego_state[3])]
                    remain_agent_attributes = [background_traffic["agent_attributes"][0]]
                    remain_recurrent_states = [background_traffic["recurrent_states"][0]]
                    if scenario is not None:
                        for agent_state in scenario.agent_states:
                            remain_agent_states.append(AgentState(center=Point(x=agent_state[0], y=agent_state[1]), orientation=agent_state[2], speed=agent_state[3]))
                        for agent_attribute in scenario.agent_attributes:
                            remain_agent_attributes.append(AgentAttributes(length=agent_attribute[0], width=agent_attribute[1], rear_axis_offset=agent_attribute[2]))
                        for recurrent_state in scenario.recurrent_states:
                            remain_recurrent_states.append(background_traffic["recurrent_states"][0])

                    for i in range(len(background_traffic["agent_states"])):
                        agent_state = background_traffic["agent_states"][i]
                        if math.dist(ego_state[:2], (agent_state.center.x, agent_state.center.y)) > 100:
                            remain_agent_states.append(agent_state)
                            remain_agent_attributes.append(background_traffic["agent_attributes"][i])
                            remain_recurrent_states.append(background_traffic["recurrent_states"][i])
                    st2 = time.time()
                    agent_attributes, agent_states, recurrent_states = iai_conditional_initialize(location=map_cfg.iai_location_name,
                        agent_count=max(95 - len(remain_agent_states), background_traffic["agent_density"]), agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
                        center=tuple(ego_state[:2]), traffic_light_state_history=[initial_light_state_name])
                    # print("iai_conditional_initialize time: ", time.time()-st2)
            agent_attributes, agent_states = agent_attributes.unsqueeze(
                0), agent_states.unsqueeze(0)
            agent_attributes, agent_states = agent_attributes.to(torch.float32), agent_states.to(
                torch.float32)
            # BxAxNxMx2
            agent_num = agent_states.shape[1]
            # print(f"agent_num: {agent_num}")
            # print(f"max_num_agents: {max_num_agents}")
            # print(f"min_num_agents: {min_num_agents}")
            # print("while loop single itr time: ", time.time()-st1)
            if agent_num > max_num_agents:
                continue
            elif agent_num < min_num_agents:
                continue
            break
        et = time.time()
        # print(f"build_simulator while loop time: {et-st}")
        kinematic_model = KinematicBicycle()
        kinematic_model.set_params(lr=agent_attributes[..., 2])
        kinematic_model.set_state(agent_states)
        renderer = renderer_from_config(
            cfg.simulator.renderer, static_mesh=driving_surface_mesh)


        waypoints = dict(vehicle=torch.Tensor(waypointseq).unsqueeze(-2).unsqueeze(0).unsqueeze(0).expand(-1, agent_num, -1, -1, -1))
        # BxAxNxM
        mask = dict(vehicle=torch.tensor([False] + [True] * (len(waypointseq) - 1)).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(-1, agent_num, -1, -1))
        waypoint_goals = WaypointGoal(waypoints, mask)
        st = time.time()
        simulator = Simulator(
            cfg=cfg.simulator, road_mesh=driving_surface_mesh,
            kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
            initial_present_mask=dict(vehicle=torch.ones_like(
                agent_states[..., 0], dtype=torch.bool)),
            renderer=renderer,
            traffic_controls=traffic_controls,
            waypoint_goals=waypoint_goals
        )
        end = time.time()
        # print(f"Simulator time: {end-st}")
        st = time.time()
        simulator = HomogeneousWrapper(simulator)
        # print(f"HomogeneousWrapper time: {time.time()-st}")
        npc_mask = torch.ones(
            agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
        npc_mask[0] = False

        if not cfg.ego_only:

            if car_sequences is not None and len(car_sequences) > 0:
                T = max([len(car_seq) for car_seq in car_sequences.values()])
                replay_states = torch.zeros((1, agent_num, T, 4))
                replay_mask = torch.zeros(replay_states.shape[:3], dtype=torch.bool)
                replay_states[:, list(car_sequences.keys()), :, :] = torch.Tensor(list(car_sequences.values())).unsqueeze(0)
                replay_mask[:, list(car_sequences.keys()), :] = True
            else:
                replay_states = None
                replay_mask = None
            st = time.time()
            simulator = IAIWrapper(
                simulator=simulator, npc_mask=npc_mask, recurrent_states=[
                    recurrent_states],
                rear_axis_offset=agent_attributes[..., 2:3], locations=[
                    map_cfg.iai_location_name],
                traffic_light_controller=traffic_light_controller,
                traffic_light_ids=traffic_light_ids,
                replay_states=replay_states,
                replay_mask=replay_mask
            )
            end = time.time()
            # print(f"IAIWrapper time: {end-st}")
        if cfg.render_mode == "video":
            simulator = BirdviewRecordingWrapper(
                simulator, res=Resolution(cfg.video_res, cfg.video_res), fov=cfg.video_fov, to_cpu=True)
        simulator.to(device)

        return simulator,agent_num,background_traffic['agent_density']


class WaypointSuiteEnv(GymEnv):
    def __init__(self, cfg: EnvConfig, data: WaypointSuite):
        self.config = cfg
        if cfg.device is None:
            self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.torch_device = torch.device(cfg.device)

        set_seeds(self.config.seed, logger)
        self.locations = data.locations
        self.map_cfgs = [find_map_config(f"carla_{location}") for location in data.locations]
        
        self.waypoint_suite = data.waypoint_suite
        self.car_sequence_suite = data.car_sequence_suite
        self.scenarios = data.scenarios
        print("Initializing WaypointSuiteEnv")
        if self.config.num_of_agents is None:
            self.cur_max_num_of_agents = np.inf
            self.cur_min_num_of_agents = 0
            self.allowed_maps = None
        else:
            self.cur_max_num_of_agents = 100
            self.cur_min_num_of_agents = 0
            self.allowed_maps = None
        super().__init__(cfg=self.config, simulator=None)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        print("Entering WaypointSuiteEnv reset")
        print(f"self.cur_max_num_of_agents: {self.cur_max_num_of_agents}")
        print(f"self.cur_min_num_of_agents: {self.cur_min_num_of_agents}")
        
        num_agent_dict = {'max': self.cur_max_num_of_agents, 'min': self.cur_min_num_of_agents}
        
        self.allowed_maps = []
        if self.config.difficulty_sets is not None:
            for i in range(len(self.config.difficulty_sets)):
                difficulty_set = self.config.difficulty_sets[i]
                if difficulty_set['max_num'] <= self.cur_max_num_of_agents:
                    if difficulty_set['min_num'] > self.cur_min_num_of_agents:
                        self.allowed_maps.extend(difficulty_set['map_names'])
        
        if self.allowed_maps is None:        
            self.current_waypoint_suite_idx = np.random.randint(len(self.waypoint_suite))
            map_cfg = self.map_cfgs[self.current_waypoint_suite_idx]
        else:
            count =0
            while True:
                #find a map_name that is in the allowed_maps   
                self.current_waypoint_suite_idx = np.random.randint(len(self.waypoint_suite))
                map_name = self.locations[self.current_waypoint_suite_idx]
                map_list_all_density_and_seeds = []
                for allowed_map_name in self.allowed_maps:
                    # print(f"map_name: {map_name} allowed_map_name: {allowed_map_name}")
                    if map_name+"_" in allowed_map_name:
                        map_list_all_density_and_seeds.append(allowed_map_name)
                # print(f"count: {count} map_name {map_name} len(map_list_all_density_and_seeds): {len(map_list_all_density_and_seeds)}")
                if len(map_list_all_density_and_seeds) == 0:
                    continue
                map_cfg = self.map_cfgs[self.current_waypoint_suite_idx]
                break
                
        num_agent_dict['map_list_all_density_and_seeds'] = map_list_all_density_and_seeds
        self.lanelet_map = map_cfg.lanelet_map

        self.set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypoint_suite[self.current_waypoint_suite_idx][self.current_target_idx]

        ego_state = (self.start_point[0], self.start_point[1], self.start_orientation, self.start_speed)

        self.last_x = None
        self.last_y = None
        self.last_psi = None

        self.last_obs = None
        self.last_reward = None
        self.last_info = None

        self.reached_waypoint_num = 0
        self.environment_steps = 0

        st = time.time()
        self.simulator,agent_num,agent_density = build_simulator(self.config,
                                         map_cfg=map_cfg,
                                         ego_state=ego_state,
                                         scenario=self.scenarios[self.current_waypoint_suite_idx],
                                         car_sequences=self.car_sequence_suite[self.current_waypoint_suite_idx],
                                         waypointseq=self.waypoint_suite[self.current_waypoint_suite_idx],
                                         device=self.torch_device,num_agent_dict=num_agent_dict)
        end = time.time()
        # print(f"build_simulator time: {end-st}")
        self.agent_num = agent_num
        self.agent_density = agent_density
        return self.get_obs(), {}
    def set_max_num_of_agents(self, max_num_of_agents):
        self.cur_max_num_of_agents = max_num_of_agents
    def set_min_num_of_agents(self, min_num_of_agents):
        self.cur_min_num_of_agents = min_num_of_agents
    def set_start_pos(self):
        self.waypoints = self.waypoint_suite[self.current_waypoint_suite_idx]
        p0 = np.array(self.waypoints[0])
        p1 = np.array(self.waypoints[1])
        # in case the start_point is offroad
        try:
            self.start_point = p0 + np.random.rand() * (p1 - p0)
            self.start_speed = np.random.rand() * 10
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map,
                                                                   x=self.start_point[0], y=self.start_point[1])[0]) \
                                     + np.random.normal(0, 0.1)
        except Exception as e:
            self.start_point = p0
            self.start_speed = np.random.rand() * 10
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map,
                                                                   x=self.start_point[0], y=self.start_point[1])[0]) \
                                     + np.random.normal(0, 0.1)

    def step(self, action: np.array):
#        try:
        st = time.time()
        state = self.simulator.get_state()
        self.last_x = state[..., 0]
        self.last_y = state[..., 1]
        self.last_psi = state[..., 2]
        self.last_speed = state[..., 3]

        obs, reward, terminated, truncated, info = super().step(action)
        if self.check_reach_target():
            self.current_target_idx += 1
            if self.current_target_idx < len(self.waypoints):
                self.current_target = self.waypoints[self.current_target_idx]
            else:
                self.current_target = None
        self.last_obs = obs
        self.last_reward = reward
        self.last_info = info
        
        # print(f"step time: {time.time()-st}")
#        except Exception as e:
#            obs, reward, terminated, truncated, info = self.mock_step()
        return obs, reward, terminated, truncated, info

    def check_reach_target(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        return (self.current_target is not None) and (math.dist((x, y), self.current_target) < 3)

    def get_reward(self):
        st = time.time()
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]

        d = math.dist((x, y), (self.last_x, self.last_y)) if (self.last_x is not None) and (self.last_y is not None) else 0
        distance_reward = self.config.distance_bonus if d > self.config.distance_cutoff else 0
        psi_reward = (1 - math.cos(psi - self.last_psi)) * (- self.config.heading_penalty) if (self.last_psi is not None) else 0
        if self.check_reach_target():
            reach_target_reward = self.config.waypoint_bonus
            self.reached_waypoint_num += 1
        else:
            reach_target_reward = 0
        r = torch.zeros_like(x)
        r += reach_target_reward + distance_reward + psi_reward
        # print(f"get_reward time: {time.time()-st}")
        return r.item()

    def is_terminated(self):
        if self.config.terminated_at_infraction:
            return ((self.simulator.compute_offroad() > 0) or (self.simulator.compute_collision() > 0) or ((self.simulator.compute_traffic_lights_violations()) > 0)).item()
        else:
            return False

    def get_info(self):
        st = time.time()
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        d = math.dist((x, y), (self.last_x, self.last_y)) if (self.last_x is not None) and (self.last_y is not None) else 0
        reached_waypoint_num = self.reached_waypoint_num
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
            reached_waypoint_num=reached_waypoint_num,
            psi_smoothness=((self.last_psi - psi) / 0.1).norm(p=2).item(),
            psi_reward=(1 - math.cos(psi - self.last_psi)) * (- self.config.heading_penalty),
            dist_reward=self.config.distance_bonus if d > self.config.distance_cutoff else 0,
            speed_smoothness=((self.last_speed - speed) / 0.1).norm(p=2).item()
        )
        # print(f"get_info time: {time.time()-st}")
        return self.info


class SingleAgentWrapper(gym.Wrapper):
    """
    Removes batch and agent dimensions from the environment interface.
    Only safe if those dimensions are both singletons.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        return self.transform_out(obs), _

    def step(self, action: np.array):
        action = torch.Tensor(action).unsqueeze(0).unsqueeze(0).to(self.torch_device)
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.transform_out(obs)
        reward = self.transform_out(reward)
        terminated = self.transform_out(terminated)
        info = self.transform_out(info)

        return obs, reward, terminated, truncated, info

    def transform_out(self, x):
        if torch.is_tensor(x):
            t = x.squeeze(0).squeeze(0).cpu()
        elif isinstance(x, dict):
            t = {k: self.transform_out(v) for (k, v) in x.items()}
        elif isinstance(x, np.ndarray):
            t = self.transform_out(torch.tensor(x)).cpu().numpy()
        else:
            t = x
        return t

    def transform_in(self, x):
        if torch.is_tensor(x):
            t = x.unsqueeze(0).unsqueeze(0)
        elif isinstance(x, dict):
            t = {k: self.transform_in(v) for (k, v) in x.items()}
        else:
            t = x
        return t

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
