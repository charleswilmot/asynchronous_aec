from os.path import abspath
from os import listdir
from pyrep import PyRep
from pyrep.const import TextureMappingMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
import numpy as np
import RESSOURCES


deg = np.rad2deg
rad = np.deg2rad


class SquaredPlane(Shape):
    def __init__(self, size):
        self.size = size
        d = size / 2
        vertices = [-d, 0.0, -d, d, 0.0, -d, d, 0.0, d, -d, 0.0, d]
        indices = [0, 1, 2, 0, 2, 3]
        shape = Shape.create_mesh(vertices, indices)
        self._handle = shape.get_handle()

    def set_texture(self, texture):
        super().set_texture(texture, TextureMappingMode.CUBE, interpolate=False, uv_scaling=[self.size, self.size])


class RandomScreen(SquaredPlane):
    def __init__(self, min_distance, max_distance, max_speed_in_deg, textures_list):
        super().__init__(1.5)
        self.textures_list = textures_list
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_speed = rad(max_speed_in_deg)
        self.episode_reset()

    def set_texture(self):
        super().set_texture(self.textures_list[np.random.randint(len(self.textures_list))])

    def episode_reset(self):
        self.distance = np.random.uniform(self.min_distance, self.max_distance)
        self.speed = np.random.uniform(0, self.max_speed)
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.set_texture()
        self.set_episode_iteration(0)

    def set_episode_iteration(self, it):
        self._episode_iteration = 0
        self.set_position(self.position)

    def increment_iteration(self):
        self.set_episode_iteration(self._episode_iteration + 1)

    def _get_position(self):
        cos_speed = np.cos(self._episode_iteration * self.speed)
        sin_speed = np.sin(self._episode_iteration * self.speed)
        cos_dir = np.cos(self.direction)
        sin_dir = np.sin(self.direction)
        x = self.distance * sin_speed * cos_dir
        y = self.distance * cos_speed
        z = self.distance * sin_speed * sin_dir
        return [x, y, z]

    def _get_tilt_pan_speed(self):
        return [deg(self.speed * np.sin(self.direction)), deg(self.speed * np.cos(self.direction))]

    position = property(_get_position)
    tilt_pan_speed = property(_get_tilt_pan_speed)


class Environment:
    def __init__(self, headless=True):
        self.pyrep = PyRep()
        self.pyrep.launch("/home/aecgroup/aecdata/Software/vrep_scenes/stereo_vision_robot.ttt", headless=headless)
        min_distance = 0.5
        max_distance = 5
        max_speed = 0
        path = "/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_bmp_selection/"
        textures_names = listdir(path)
        textures_list = [self.pyrep.create_texture(path + name)[1] for name in textures_names]
        self.screen = RandomScreen(min_distance, max_distance, max_speed, textures_list)
        self.robot = StereoVisionRobot(min_distance, max_distance)
        self.pyrep.start()

    def step(self):
        # move screen
        self.screen.increment_iteration()
        # step simulation
        self.pyrep.step()

    def episode_reset(self):
        # reset screen
        self.screen.episode_reset()
        # reset robot
        self.robot.episode_reset()
        self.pyrep.step()

    def close(self):
        self.pyrep.stop()
        self.pyrep.shutdown()


class StereoVisionRobot:
    def __init__(self, min_distance, max_distance, joint_limit_type="stick"):
        self.cam_left = VisionSensor("vs_cam_left#")
        self.cam_right = VisionSensor("vs_cam_right#")
        self.tilt_left = Joint("vs_eye_tilt_left#")
        self.tilt_right = Joint("vs_eye_tilt_right#")
        self.pan_left = Joint("vs_eye_pan_left#")
        self.pan_right = Joint("vs_eye_pan_right#")
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.joint_limit_type = joint_limit_type
        self._pan_max = rad(20)
        self._tilt_max = rad(20)
        self._tilt_speed = 0
        self._pan_speed = 0
        self.episode_reset()

    def episode_reset(self):
        self._episode_iteration = 0
        ### reset joints position / speeds
        self._tilt_speed = 0
        self._pan_speed = 0
        self.pan_right.set_joint_position(0)
        self.pan_left.set_joint_position(0)
        self.tilt_right.set_joint_position(0)
        self.tilt_left.set_joint_position(0)
        fixation_distance = np.random.uniform(self.min_distance, self.max_distance)
        random_vergence = np.arctan(RESSOURCES.Y_EYES_DISTANCE / (2 * fixation_distance))
        self.pan_left.set_joint_position(random_vergence / 2)
        self.pan_right.set_joint_position(-random_vergence / 2)

    def _check_pan_limit(self, left, right):
        if self.joint_limit_type == "stick":
            return np.clip(left, 0, self._pan_max), np.clip(right, -self._pan_max, 0)

    def _check_tilt_limit(self, left, right):
        if self.joint_limit_type == "stick":
            return np.clip(left, -self._tilt_max, self._tilt_max), np.clip(right, -self._tilt_max, self._tilt_max)

    def reset_vergence_position(self):
        mean = (self.pan_right.get_joint_position() + self.pan_left.get_joint_position()) / 2
        left, right = self._check_pan_limit(mean, mean)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_delta_vergence_position(self, delta):
        rad_delta = rad(delta)
        left, right = self._check_pan_limit(
            self.pan_left.get_joint_position() + rad_delta / 2,
            self.pan_right.get_joint_position() - rad_delta / 2)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_delta_pan_speed(self, delta):
        self._pan_speed += rad(delta)
        left, right = self._check_pan_limit(
            self.pan_left.get_joint_position() + self._pan_speed,
            self.pan_right.get_joint_position() + self._pan_speed)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_delta_tilt_speed(self, delta):
        self._tilt_speed += rad(delta)
        left, right = self._check_tilt_limit(
            self.tilt_left.get_joint_position() + self._tilt_speed,
            self.tilt_right.get_joint_position() + self._tilt_speed)
        self.tilt_left.set_joint_position(left)
        self.tilt_right.set_joint_position(right)

    def get_tilt_position(self):
        return deg(self.tilt_right.get_joint_position())

    def get_pan_position(self):
        return deg((self.pan_right.get_joint_position() + self.pan_left.get_joint_position()) / 2)

    def get_vergence_position(self):
        return deg(self.pan_left.get_joint_position() - self.pan_right.get_joint_position())

    def get_tilt_speed(self):
        return deg(self._tilt_speed)

    def get_pan_speed(self):
        return deg(self._pan_speed)

    def set_action(self, action):
        self.set_delta_vergence_position(action[0])
        self.set_delta_pan_speed(action[1])
        self.set_delta_tilt_speed(action[2])

    def get_vision(self):
        return self.cam_left.capture_rgb(), self.cam_right.capture_rgb()

    tilt_position = property(get_tilt_position)
    pan_position = property(get_pan_position)
    vergence_position = property(get_vergence_position)
    tilt_speed = property(get_tilt_speed)
    pan_speed = property(get_pan_speed)


if __name__ == "__main__":
    import time

    env = Environment(False)
    env.step()
    env.robot.reset_vergence_position()
    env.step()
    env.robot.set_delta_vergence_position(30)
    env.step()
    print("measured vergence: {}".format(env.robot.get_vergence_position()))
    t0 = time.time()
    for i in range(10):
        env.episode_reset()
        for i in range(10):
            env.robot.set_delta_vergence_position(1)
            env.step()
            env.robot.get_vision()
    t1 = time.time()
    env.close()
    print("About {} sec per episode".format((t1 - t0) / 10))
    print("About {} iterations per sec".format(100 / (t1 - t0)))