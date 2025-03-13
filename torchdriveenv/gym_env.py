"""
An example showing how to define an OpenAI gym environment based on TorchDriveSim.
It uses the IAI API to provide behaviors for other vehicles and requires an access key to run.
"""
import os
import logging
import math
import inspect
import json
import pickle
import random
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Tuple

import gymnasium as gym
import torch
from torch import Tensor
from invertedai.common import AgentState, Point, AgentAttributes, RecurrentState

from torchdrivesim.behavior.iai import IAIWrapper, iai_drive
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import BicycleNoReversing, KinematicBicycle
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.rendering.base import RendererConfig
from torchdrivesim.utils import Resolution
from torchdrivesim.lanelet2 import find_lanelet_directions
from torchdrivesim.map import find_map_config, traffic_controls_from_map_config
from torchdrivesim.traffic_lights import current_light_state_tensor_from_controller
from torchdrivesim.simulator import TorchDriveConfig, SimulatorInterface, \
    BirdviewRecordingWrapper, Simulator, HomogeneousWrapper, CollisionMetric

from torchdriveenv.helpers import save_video, set_seeds, sample_waypoints_from_graph, convert_to_json
from torchdriveenv.iai import iai_conditional_initialize, iai_blame
from torchdriveenv.record_data import OfflineDataRecordingWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaselineAlgorithm(Enum):
    """
    Method used to calculate collisions between agents.
    """
    sac = 'sac'
    ppo = 'ppo'
    a2c = 'a2c'
    td3 = 'td3'


@dataclass
class EnvConfig:
    ego_only: bool = False
    max_environment_steps: int = 200
    use_background_traffic: bool = True
    terminated_at_infraction: bool = True
    use_expert_action: bool = True
    seed: Optional[int] = None
    simulator: TorchDriveConfig = TorchDriveConfig(renderer=RendererConfig(left_handed_coordinates=True,
                                                                           highlight_ego_vehicle=True),
                                                   collision_metric=CollisionMetric.nograd,
                                                   left_handed_coordinates=True)
    render_mode: Optional[str] = "rgb_array"
    video_filename: Optional[str] = "rendered_video.mp4"
    video_res: Optional[int] = 1024
    video_fov: Optional[float] = 500
    record_episode_data: bool = False
    record_replay_data: bool = False
    terminated_at_blame: bool = False
    log_blame: bool = False
    train_replay_data_path: Optional[str] = None
    val_replay_data_path: Optional[str] = None


@dataclass
class RlTrainingConfig:
    algorithm: BaselineAlgorithm = None
    parallel_env_num: Optional[int] = 2
    env: EnvConfig = EnvConfig()


@dataclass
class Scenario:
    agent_states: List[List[float]] = None
    agent_attributes: List[List[float]] = None
    recurrent_states: List[List[float]] = None


@dataclass
class Node:
    id: int
    point: Tuple[float]
    next_node_ids: List[int]
    next_edges: List[float]


@dataclass
class WaypointSuite:
    locations: List[str] = None
    waypoint_suite: List[List[List[float]]] = None
    car_sequence_suite: List[Optional[Dict[int, List[List[float]]]]] = None
    scenarios: List[Optional[Scenario]] = None
    waypoint_graphs: List[List[Node]] = None


@dataclass
class StepData:
    obs_birdview: List
    ego_action: Tuple
    ego_state: List
    recurrent_states: List[List[float]]
    reward: float
    info: Dict
    waypoint: Tuple
#    q: float


@dataclass
class EpisodeData:
    location: str
    step_data: List[StepData]


@dataclass
class ReplayRecord:
    location: str
    agent_attributes: List
    traffic_light_ids: List
    agent_states: List
    traffic_light_state_history: List
    waypoint_seq: List


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
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

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

    def step(self, action: Tensor):
        self.environment_steps += 1
        self.simulator.step(action)
        self.last_action = self.current_action if self.current_action is not None else action
        self.current_action = action
        return self.get_obs(), self.get_reward(), self.is_terminated(), self.is_truncated(), self.get_info()

    def get_obs(self):
        birdview = self.simulator.render_egocentric().cpu().numpy()
#        birdview[0, 0, :, 32:, :] = 0.0
        return birdview

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        r = torch.zeros_like(x)
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
#            birdview[0, 0, :, 32:, :] = 0.0
            return np.transpose(birdview.squeeze(), axes=(1, 2, 0))
        else:
            raise NotImplementedError

    def mock_step(self):
        obs = np.zeros((1, 3, 64, 64))  # self.last_obs
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
            print("is_birdview")
            bvs = self.simulator.get_birdviews()
            if len(bvs) > 1:
                save_video(bvs, self.config.video_filename)

        if isinstance(self.simulator.inner_simulator, BirdviewRecordingWrapper):
            print("is_inner.birdview")
            bvs = self.simulator.inner_simulator.get_birdviews()
            if len(bvs) > 1:
                save_video(bvs, self.config.video_filename)


def build_simulator(cfg: EnvConfig, map_cfg, ego_state, scenario=None, car_sequences=None, waypointseq=None):
    with torch.no_grad():
        device = torch.device("cuda")
        traffic_light_controller = map_cfg.traffic_light_controller
        initial_light_state_name = traffic_light_controller.current_state_with_name
        traffic_light_ids = [
            stopline.actor_id for stopline in map_cfg.stoplines if stopline.agent_type == 'traffic-light']
        driving_surface_mesh = map_cfg.road_mesh.to(device)

        traffic_controls = traffic_controls_from_map_config(map_cfg)
        traffic_controls = {key: traffic_controls[key].to(
            device) for key in traffic_controls}
        traffic_controls['traffic_light'].set_state(current_light_state_tensor_from_controller(
            traffic_light_controller, traffic_light_ids).unsqueeze(0).to(device))

        if cfg.ego_only:
            agent_states = torch.Tensor(
                [ego_state[0], ego_state[1], ego_state[2], ego_state[3]]).unsqueeze(0)
            length = np.random.random() * (5.5 - 4.8) + 4.8
            width = np.random.random() * (2.2 - 1.8) + 1.8
            rear_axis_offset = np.random.random() * (0.97 - 0.82) + 0.82
            agent_attributes = torch.Tensor(
                [length, width, rear_axis_offset]).unsqueeze(0)
            recurrent_states = torch.Tensor([0] * 132).unsqueeze(0)
        else:
            if cfg.use_background_traffic:
                background_traffic_dir = os.path.join(
                    os.path.dirname(os.path.realpath(
                        __file__)), f"resources/background_traffic")
                while True:
                    background_traffic_file = os.path.join(background_traffic_dir, random.choice(list(filter(
                        lambda x: x.split("_")[1] == map_cfg.name[6:], os.listdir(background_traffic_dir)))))
                    with open(background_traffic_file, "r") as f:
                        background_traffic_json = json.load(f)
                    background_traffic = {}
                    background_traffic['location'] = background_traffic_json['location']
                    background_traffic['agent_density'] = background_traffic_json['agent_density']
                    background_traffic['random_seed'] = background_traffic_json['random_seed']
                    background_traffic['agent_states'] = [AgentState.model_validate(
                        agent_state) for agent_state in background_traffic_json['agent_states']]
                    background_traffic['agent_attributes'] = [AgentAttributes.model_validate(
                        agent_attribute) for agent_attribute in background_traffic_json['agent_attributes']]
                    background_traffic['recurrent_states'] = [RecurrentState.model_validate(
                        recurrent_state) for recurrent_state in background_traffic_json['recurrent_states']]

                    if len(background_traffic["agent_states"]) + background_traffic["agent_density"] < 100:
                        break

                remain_agent_states = [AgentState(center=Point(
                    x=ego_state[0], y=ego_state[1]), orientation=ego_state[2], speed=ego_state[3])]
                remain_agent_attributes = [
                    background_traffic["agent_attributes"][0]]
                remain_recurrent_states = [
                    background_traffic["recurrent_states"][0]]
                if scenario is not None:
                    for agent_state in scenario.agent_states:
                        remain_agent_states.append(AgentState(center=Point(
                            x=agent_state[0], y=agent_state[1]), orientation=agent_state[2], speed=agent_state[3]))
                        remain_recurrent_states.append(
                            background_traffic["recurrent_states"][0])
                    for agent_attribute in scenario.agent_attributes:
                        remain_agent_attributes.append(AgentAttributes(
                            length=agent_attribute[0], width=agent_attribute[1], rear_axis_offset=agent_attribute[2]))

#                for i in range(len(background_traffic["agent_states"])):
#                    agent_state = background_traffic["agent_states"][i]
#                    if math.dist(ego_state[:2], (agent_state.center.x, agent_state.center.y)) > 100:
#                        remain_agent_states.append(agent_state)
#                        remain_agent_attributes.append(background_traffic["agent_attributes"][i])
#                        remain_recurrent_states.append(background_traffic["recurrent_states"][i])
                agent_attributes, agent_states, recurrent_states = iai_conditional_initialize(location=map_cfg.iai_location_name,
                       agent_count=min(30, max(95 - len(remain_agent_states), background_traffic["agent_density"])), agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
                       center=tuple(ego_state[:2]), traffic_light_state_history=[initial_light_state_name])

        agent_attributes, agent_states = agent_attributes.unsqueeze(
            0), agent_states.unsqueeze(0)
        agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(
            torch.float32)
        kinematic_model = BicycleNoReversing()
        kinematic_model.set_params(lr=agent_attributes[..., 2])
        kinematic_model.set_state(agent_states)
        renderer = renderer_from_config(
            cfg.simulator.renderer, static_mesh=driving_surface_mesh)

        # BxAxNxMx2
        agent_num = agent_states.shape[1]
        waypoints = dict(vehicle=torch.Tensor(waypointseq).unsqueeze(-2).unsqueeze(
            0).unsqueeze(0).expand(-1, agent_num, -1, -1, -1).to(device))
        # BxAxNxM
        mask = dict(vehicle=torch.tensor([False] + [True] * (len(waypointseq) - 1)).unsqueeze(-1).unsqueeze(
            0).unsqueeze(0).expand(-1, agent_num, -1, -1).to(device))
        waypoint_goals = WaypointGoal(waypoints, mask)

        simulator = Simulator(
            cfg=cfg.simulator, road_mesh=driving_surface_mesh,
            kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
            initial_present_mask=dict(vehicle=torch.ones_like(
                agent_states[..., 0], dtype=torch.bool)),
            renderer=renderer,
            traffic_controls=traffic_controls,
            waypoint_goals=None
        )
#            waypoint_goals=waypoint_goals
        simulator = HomogeneousWrapper(simulator)
        npc_mask = torch.ones(
            agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
        npc_mask[0] = False

        if car_sequences is not None and len(car_sequences) > 0:
            T = max([len(car_seq) for car_seq in car_sequences.values()])
            replay_states = torch.zeros((1, agent_num, T, 4))
            replay_mask = torch.zeros(
                replay_states.shape[:3], dtype=torch.bool)
            replay_states[:, list(car_sequences.keys()), :, :] = torch.Tensor(
                list(car_sequences.values())).unsqueeze(0)
            replay_mask[:, list(car_sequences.keys()), :] = True
        else:
            replay_states = None
            replay_mask = None

        if not cfg.ego_only:
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
        if cfg.render_mode == "video":
            simulator = BirdviewRecordingWrapper(
                simulator, res=Resolution(cfg.video_res, cfg.video_res), fov=cfg.video_fov, to_cpu=True)
        if cfg.record_replay_data or cfg.terminated_at_blame or cfg.log_blame:
            simulator = OfflineDataRecordingWrapper(simulator)

        return simulator


class WaypointSuiteEnv(GymEnv):
    def __init__(self, cfg: EnvConfig, data: WaypointSuite):
        self.config = cfg
        set_seeds(self.config.seed, logger)
        self.map_cfgs = [find_map_config(
            f"carla_{location}") for location in data.locations]

        self.waypoint_suite = data.waypoint_suite
        self.waypoint_graphs = data.waypoint_graphs
        self.car_sequence_suite = data.car_sequence_suite
        self.scenarios = data.scenarios
        super().__init__(cfg=cfg, simulator=None)
        if self.config.use_expert_action:
            self.expert_kinematic_model = KinematicBicycle(left_handed=True)
        if self.config.record_episode_data:
            self.episode_data = None
            self.episode_data_dir = f"offline_datasets/episode_data_{datetime.now().strftime('%Y%m%d-%H%M')}"
            if not os.path.exists(self.episode_data_dir):
                os.mkdir(self.episode_data_dir)
#                link_path = "offline_datasets/latest_episode_data"
#                os.unlink(link_path)
#                os.symlink(self.episode_data_dir, link_path)
        if self.config.record_replay_data:
            self.replay_data = None
            self.replay_data_dir = f"offline_datasets/replay_data_{datetime.now().strftime('%Y%m%d-%H%M')}"
            if not os.path.exists(self.replay_data_dir):
                os.mkdir(self.replay_data_dir)
                link_path = "offline_datasets/latest_replay_data"
                os.unlink(link_path)
                os.symlink(self.replay_data_dir, link_path)
        self.data_index = -1

        logger.info(inspect.getsource(WaypointSuiteEnv.get_reward))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if isinstance(self.simulator, BirdviewRecordingWrapper):
            print("is_birdview")
            bvs = self.simulator.get_birdviews()
            if len(bvs) > 1:
                save_video(bvs, self.config.video_filename)

        if self.simulator is not None and isinstance(self.simulator.inner_simulator, BirdviewRecordingWrapper):
            print("is_inner.birdview")
            bvs = self.simulator.inner_simulator.get_birdviews()
            if len(bvs) > 1:
                save_video(bvs, self.config.video_filename)

        if self.config.record_episode_data:
            if self.data_index >= 0:
                self.episode_data.location = self.location
                if len(self.episode_data.step_data) > 0:
#                    with open(f"{self.episode_data_dir}/episode_{self.data_index}_{random.randint(0, 100000)}.pkl", "wb") as f:
#                        pickle.dump(self.episode_data, f)
                    episode_dict = convert_to_json(asdict(self.episode_data))

#                    print("episode_dict")
#                    print(episode_dict)

                    # Save to JSON
                    with open(f"{self.episode_data_dir}/episode_{self.data_index}_{random.randint(0, 100000)}.pkl", "w") as f:
                        json.dump(episode_dict, f, indent=4)

            self.episode_data = EpisodeData(location="", step_data=[])

        if self.config.record_replay_data:
            if self.data_index >= 0:
                records = self.simulator.get_records()
                iai_simulator = self.simulator.inner_simulator
                if not isinstance(iai_simulator, IAIWrapper):
                    iai_simulator = iai_simulator.inner_simulator
                traffic_light_ids = iai_simulator._traffic_light_ids
                agent_attributes = iai_simulator._agent_attributes
                self.replay_data = ReplayRecord(location=self.location, agent_attributes=agent_attributes, traffic_light_ids=traffic_light_ids, agent_states=records[
                                                "agent_states"], traffic_light_state_history=records["traffic_light_state_history"], waypoint_seq=self.waypoint_suite[self.current_waypoint_suite_idx])
                with open(f"{self.replay_data_dir}/replay_{self.data_index}_{random.randint(0, 100000)}.pkl", "wb") as f:
                    pickle.dump(self.replay_data, f)

        self.data_index += 1

        self.current_waypoint_suite_idx = np.random.randint(
            len(self.waypoint_suite))
#        self.current_waypoint_suite_idx = 4
        self.map_cfg = self.map_cfgs[self.current_waypoint_suite_idx]
        self.location = self.map_cfgs[self.current_waypoint_suite_idx].name
#        print(self.location)
#        while self.location != "carla_Town10HD":
#            self.current_waypoint_suite_idx = np.random.randint(len(self.waypoint_suite))
#    #        self.current_waypoint_suite_idx = 4
#            self.map_cfg = self.map_cfgs[self.current_waypoint_suite_idx]
#            self.location = self.map_cfgs[self.current_waypoint_suite_idx].name

        self.lanelet_map = self.map_cfg.lanelet_map
        if (self.waypoint_graphs is not None) and (self.waypoint_graphs[self.current_waypoint_suite_idx] is not None):
            self.waypoint_suite[self.current_waypoint_suite_idx] = sample_waypoints_from_graph(self.waypoint_graphs[self.current_waypoint_suite_idx])

        self.set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypoint_suite[self.current_waypoint_suite_idx][self.current_target_idx]

        ego_state = (self.start_point[0], self.start_point[1],
                     self.start_orientation, self.start_speed)

        self.last_x = None
        self.last_y = None
        self.last_psi = None

        self.last_reward = None
        self.last_info = None

        self.last_colliding_agent = None
        self.last_colliding_step = -100

        self.reached_waypoint_num = 0
        self.environment_steps = 0

        self.simulator = build_simulator(self.config,
                                         map_cfg=self.map_cfg,
                                         ego_state=ego_state,
                                         scenario=self.scenarios[self.current_waypoint_suite_idx],
                                         car_sequences=self.car_sequence_suite[self.current_waypoint_suite_idx],
                                         waypointseq=self.waypoint_suite[self.current_waypoint_suite_idx])
        obs = self.get_obs()
        self.last_obs = obs

        return obs, {}

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

    def expert_prediction(self):
#        location = f'carla:{":".join(self.map_cfgs[self.current_waypoint_suite_idx].name.split("_"))}'
        location = self.map_cfgs[self.current_waypoint_suite_idx].iai_location_name
        agent_states = self.simulator.get_innermost_simulator().get_state()["vehicle"].squeeze(0).cpu().numpy()
        iai_simulator = self.simulator
        if not isinstance(iai_simulator, IAIWrapper):
            iai_simulator = iai_simulator.inner_simulator
            if not isinstance(iai_simulator, IAIWrapper):
                iai_simulator = iai_simulator.inner_simulator
#        print("simulator")
#        print(self.simulator)
#        print(self.simulator.inner_simulator)
#        print(iai_simulator)
        agent_attributes = iai_simulator._agent_attributes
#        recurrent_states = self.simulator._recurrent_states
        traffic_lights_states = iai_simulator._traffic_light_controller.current_state_with_name
        waypoint_for_ego = self.current_target
#               "recurrent_states": recurrent_states[0],
        obs = {"location": location,
               "agent_states": agent_states,
               "agent_attributes": agent_attributes[0],
               "recurrent_states": None,
               "traffic_lights_states": traffic_lights_states,
               "waypoint_for_ego": waypoint_for_ego}
        states, iai_recurrent_states = iai_drive(location=obs["location"],
                                             agent_states=obs["agent_states"],
                                             agent_attributes=obs["agent_attributes"],
                                             recurrent_states=obs["recurrent_states"],
                                             traffic_lights_states=obs["traffic_lights_states"],
                                             waypoint_for_ego=obs["waypoint_for_ego"])

        current_state = torch.Tensor(obs["agent_states"][0])

        action = self.expert_kinematic_model.fit_action(
                    future_state=states[0], current_state=current_state).to(torch.device("cuda"))
        recurrent_states = [recurrent_state.packed for recurrent_state in iai_recurrent_states]
        action = action.unsqueeze(0).unsqueeze(0)
#        print("action")
#        print(action)
        return action, recurrent_states, current_state


    def step(self, action: Tensor):
#        try:
#@dataclass
#class StepData:
#    obs_birdview: List
#    ego_action: Tuple
#    reward: float
#    info: Dict
#    waypoint: Tuple
#
#
#@dataclass
#class EpisodeData:
#    location: str
#    step_data: List[StepData]
#        print("action from RL")
#        print(action)
        if self.config.use_expert_action:
#            self.expert_action = self.expert_prediction()
#            self.action = action
            action, recurrent_states, ego_state = self.expert_prediction()
        else:
            recurrent_states = None
            agent_states = self.simulator.get_innermost_simulator().get_state()["vehicle"].squeeze(0).cpu().numpy()
            ego_state = torch.Tensor(agent_states[0])

        state = self.simulator.get_state()
        self.last_x = state[..., 0]
        self.last_y = state[..., 1]
        self.last_psi = state[..., 2]
        self.last_speed = state[..., 3]

        obs, reward, terminated, truncated, info = super().step(action)
        if self.config.record_episode_data:
            step_data = StepData(obs_birdview=self.last_obs,
                                 ego_action=action,
                                 ego_state=ego_state,
                                 recurrent_states=recurrent_states,
                                 reward=reward,
                                 info=info,
                                 waypoint=self.current_target)
            self.episode_data.step_data.append(step_data)

        if self.check_reach_target():
            self.current_target_idx += 1
            if self.current_target_idx < len(self.waypoints):
                self.current_target = self.waypoints[self.current_target_idx]
            else:
                self.current_target = None
        self.last_obs = obs
        self.last_reward = reward
        self.last_info = info

#        if self.config.record_episode_data and (terminated or truncated):
#            step_data = StepData(obs_birdview=self.last_obs,
#                                 ego_action=None,
#                                 reward=self.last_reward,
#                                 info=self.last_info,
#                                 waypoint=self.current_target)
#            self.episode_data.step_data.append(step_data)
#        except Exception as e:
#            obs, reward, terminated, truncated, info = self.mock_step()
        return obs, reward, terminated, truncated, info

    def check_reach_target(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        return (self.current_target is not None) and (math.dist((x, y), self.current_target) < 5)

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]

        d = math.dist((x, y), (self.last_x, self.last_y)) if (self.last_x is not None) and (self.last_y is not None) else 0
        distance_reward = 1 if d > 0.1 else 0
        psi_reward = (1 - math.cos(psi - self.last_psi)) * (-20.0) if (self.last_psi is not None) else 0
#        speed_reward = -1 if d > 0.5 else 0
        if self.check_reach_target():
            reach_target_reward = 10
            self.reached_waypoint_num += 1
        else:
            reach_target_reward = 0
        r = torch.zeros_like(x)
        r += reach_target_reward + distance_reward + psi_reward
#        r += reach_target_reward + psi_reward
#        r += distance_reward
        return r

    def is_terminated(self):
#                   ((self.simulator.compute_traffic_lights_violations()) > 0)
        if self.config.terminated_at_infraction:
            return (self.simulator.compute_offroad() > 0) or \
                   (self.simulator.compute_collision() > 0 and \
                           ((not self.config.terminated_at_blame) or (self.config.terminated_at_blame and self.check_blame()))) # or \
        else:
            return False

    def check_blame(self):
        iai_simulator = self.simulator.inner_simulator
        if not isinstance(iai_simulator, IAIWrapper):
            iai_simulator = iai_simulator.inner_simulator
        agent_attributes = iai_simulator._agent_attributes
        agent_state_history = self.simulator.records["agent_states"]
        distances = [math.dist(agent_state_history[-1][0, 0, :2], agent_state_history[-1][0, i, :2]) for i in range(1, agent_state_history[-1].shape[-2])]
        colliding_agent = np.argmin(distances) + 1
        if (self.environment_steps > self.last_colliding_step + 1) or (self.last_colliding_agent != colliding_agent):
            blame_result = (0 in iai_blame(self.map_cfg.iai_location_name, colliding_agents=(0, colliding_agent), agent_state_history=agent_state_history, agent_attributes=agent_attributes.squeeze(), traffic_light_state_history=self.simulator.records["traffic_light_state_history"]))
        else:
            blame_result = False
        self.last_colliding_step = self.environment_steps
        self.last_colliding_agent = colliding_agent
        return blame_result


    def get_info(self):
        psi = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        reached_waypoint_num = self.reached_waypoint_num
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
            reached_waypoint_num=reached_waypoint_num,
            psi_smoothness=((self.last_psi - psi) / 0.1).norm(p=2).item(),
            speed_smoothness=((self.last_speed - speed) / 0.1).norm(p=2).item()
        )
        if (self.info["collision"] > 0) and (self.config.log_blame):
            self.info["blame"] = self.check_blame()
        else:
            self.info["blame"] = None
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

    def step(self, action: Tensor):
        action = torch.Tensor(action).unsqueeze(0).unsqueeze(0).to("cuda")
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
