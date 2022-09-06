import carla
from navigation.basic_agent import BasicAgent
from navigation.local_planner import RoadOption
from tools.misc import get_speed


class CustomAgent(BasicAgent):
    def __init__(self, vehicle):
        super(CustomAgent, self).__init__(vehicle)

        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self.max_speed = 70
        self.speed_lim_dist = 3
        self._sampling_resolution = 4.5
        self._min_proximity_threshold = 10
        self.braking_distance = 10

        self._tailgate_counter = 0

    def _update_information(self):
        """
        Actually, this is mainly just local planner waypoint stuff.
        Obstacles are not really dealt with here
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option

        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int(self._speed_limit / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)

        if self._incoming_direction is None:
            self.incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def emergency_stop(self):
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def collision_and_car_avoid_manager(self, waypoint):
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v): return v.get_location().distance(waypoint.transform.location)

        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            vehicle_list,
            max(
                self._min_proximity_threshold,
                self._speed_limit / 3
            ),
            up_angle_th=30
        )

        return vehicle_state, vehicle, distance

    def run_step(self, adv_cam_loc, adv_dist, debug=False):
        self._update_information()

        control = None

        if self._tailgate_counter > 0:
            self._tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red Lights and stop behaviour
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.2: Car Following Behaviours
        # vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        # If there is some sort of collision potential
        if adv_dist is not None and adv_dist < self.braking_distance:
            return self.emergency_stop()
        else:
            target_speed = min([
                self.max_speed,
                self._speed_limit - self.speed_lim_dist])
            self._local_planner.set_speed(target_speed)

            control = self._local_planner.run_step(debug=debug)

        return control
