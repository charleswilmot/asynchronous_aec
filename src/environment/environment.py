from os import listdir
from pyrep import PyRep
from pyrep.const import TextureMappingMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
import numpy as np
from helper.utils import deg, rad, to_angle


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

    def set_texture(self, index=None):
        if index is None:
            super().set_texture(self.textures_list[np.random.randint(len(self.textures_list))])
        else:
            super().set_texture(self.textures_list[index])

    def episode_reset(self, preinit=False):
        self.distance = np.random.uniform(self.min_distance, self.max_distance)
        #self.speed = np.random.uniform(rad(1), rad(1.5))
        self.speed = rad(0.7)
        #self.direction = 0
        #self.direction = np.random.choice([0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        self.direction = np.random.choice([rad(45), rad(195)])
        #self.direction = np.random.uniform(rad(0), rad(360))
        #self.direction = np.random.uniform(0, 2 * np.pi)
        self.set_texture()
        self.set_episode_iteration(-1 if preinit else 0)

    def set_trajectory(self, distance, tilt_speed_deg, pan_speed_deg, preinit=False):
        self.distance = distance
        tilt_speed = rad(tilt_speed_deg)
        pan_speed = rad(pan_speed_deg)
        self.speed = np.sqrt(tilt_speed ** 2 + pan_speed ** 2)
        if self.speed > 0:
            self.direction = np.arccos(pan_speed / self.speed)
        else:
            self.direction = 0.0
        if tilt_speed < 0:
            self.direction = -self.direction
        self.set_episode_iteration(-1 if preinit else 0)

    def set_episode_iteration(self, it):
        self._episode_iteration = it
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
    def __init__(self, texture="/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_bmp_selection/",
                 scene="/home/aecgroup/aecdata/Software/vrep_scenes/stereo_vision_robot.ttt", headless=True):
        self.pyrep = PyRep()
        self.pyrep.launch(scene, headless=headless)
        min_distance = 0.5
        max_distance = 5
        max_speed = 0.5
        path = texture
        textures_names = listdir(path)
        textures_list = [self.pyrep.create_texture(path + name)[1] for name in textures_names]
        self.screen = RandomScreen(min_distance, max_distance, max_speed, textures_list)
        self.robot = StereoVisionRobot(min_distance, max_distance)
        self.pyrep.start()

    def step(self):
        # step simulation
        self.pyrep.step()
        # move screen
        self.screen.increment_iteration()

    def episode_reset(self, preinit=False):
        # reset screen
        self.screen.episode_reset(preinit=preinit)
        # reset robot
        self.robot.episode_reset()
        self.robot.set_vergence_position(to_angle(self.screen.distance))
        self.pyrep.step()

    def close(self):
        self.pyrep.stop()
        self.pyrep.shutdown()


class StereoVisionRobot:
    def __init__(self, min_distance, max_distance, default_joint_limit_type="none"):
        self.cam_left = VisionSensor("vs_cam_left#")
        self.cam_right = VisionSensor("vs_cam_right#")
        self.tilt_left = Joint("vs_eye_tilt_left#")
        self.tilt_right = Joint("vs_eye_tilt_right#")
        self.pan_left = Joint("vs_eye_pan_left#")
        self.pan_right = Joint("vs_eye_pan_right#")
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.default_joint_limit_type = default_joint_limit_type
        self._pan_max = rad(4)
        self._tilt_max = rad(10)
        self._tilt_speed = 0
        self._pan_speed = 0
        self.episode_reset()

    def episode_reset(self):
        ### reset joints position / speeds
        self.reset_speed()
        self.tilt_right.set_joint_position(0)
        self.tilt_left.set_joint_position(0)
        fixation_distance = np.random.uniform(self.min_distance, self.max_distance)
        random_vergence = rad(to_angle(fixation_distance))
        self.pan_left.set_joint_position(random_vergence / 2)
        self.pan_right.set_joint_position(-random_vergence / 2)

    def reset_speed(self):
        self._tilt_speed = 0
        self._pan_speed = 0

    def _check_pan_limit(self, left, right, joint_limit_type=None):
        joint_limit_type = self.default_joint_limit_type if joint_limit_type is None else joint_limit_type
        if joint_limit_type == "stick":
            return np.clip(left, 0, self._pan_max), np.clip(right, -self._pan_max, 0)
        if joint_limit_type == "none":
            return left, right

    def _check_tilt_limit(self, left, right, joint_limit_type=None):
        joint_limit_type = self.default_joint_limit_type if joint_limit_type is None else joint_limit_type
        if joint_limit_type == "stick":
            return np.clip(left, -self._tilt_max, self._tilt_max), np.clip(right, -self._tilt_max, self._tilt_max)
        if joint_limit_type == "none":
            return left, right

    def reset_vergence_position(self, joint_limit_type=None):
        mean = (self.pan_right.get_joint_position() + self.pan_left.get_joint_position()) / 2
        left, right = self._check_pan_limit(mean, mean, joint_limit_type)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_vergence_position(self, alpha, joint_limit_type=None):
        rad_alpha = rad(alpha)
        mean = (self.pan_right.get_joint_position() + self.pan_left.get_joint_position()) / 2
        left, right = self._check_pan_limit(mean + rad_alpha / 2, mean - rad_alpha / 2, joint_limit_type)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_delta_vergence_position(self, delta, joint_limit_type=None):
        rad_delta = rad(delta)
        left, right = self._check_pan_limit(
            self.pan_left.get_joint_position() + rad_delta / 2,
            self.pan_right.get_joint_position() - rad_delta / 2,
            joint_limit_type)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_pan_position(self, alpha, joint_limit_type=None):
        rad_alpha = rad(alpha)
        vergence = self.pan_left.get_joint_position() - self.pan_right.get_joint_position()
        left, right = self._check_pan_limit((vergence/2) + rad_alpha, -(vergence/2) + rad_alpha, joint_limit_type)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_delta_pan_speed(self, delta, joint_limit_type=None):
        self._pan_speed += rad(delta)
        left, right = self._check_pan_limit(
            self.pan_left.get_joint_position() + self._pan_speed,
            self.pan_right.get_joint_position() + self._pan_speed,
            joint_limit_type)
        self.pan_left.set_joint_position(left)
        self.pan_right.set_joint_position(right)

    def set_tilt_position(self, alpha, joint_limit_type=None):
        rad_alpha = rad(alpha)
        left, right = self._check_tilt_limit(rad_alpha, rad_alpha, joint_limit_type)
        self.tilt_left.set_joint_position(left)
        self.tilt_right.set_joint_position(right)

    def set_delta_tilt_speed(self, delta, joint_limit_type=None):
        self._tilt_speed += rad(delta)
        left, right = self._check_tilt_limit(
            self.tilt_left.get_joint_position() + self._tilt_speed,
            self.tilt_right.get_joint_position() + self._tilt_speed,
            joint_limit_type)
        self.tilt_left.set_joint_position(left)
        self.tilt_right.set_joint_position(right)

    def get_tilt_position(self):
        return deg(self.tilt_right.get_joint_position())

    def get_pan_position(self):
        return deg((self.pan_right.get_joint_position() + self.pan_left.get_joint_position()) / 2)

    def get_vergence_position(self):
        return deg(self.pan_left.get_joint_position() - self.pan_right.get_joint_position())

    def get_position(self):
        return self.tilt_position, self.pan_position, self.vergence_position

    def get_vergence_error(self, other_distance):
        return to_angle(other_distance) - self.vergence_position

    def get_tilt_speed(self):
        return deg(self._tilt_speed)

    def get_pan_speed(self):
        return deg(self._pan_speed)

    def get_speed(self):
        return self.tilt_speed, self.pan_speed

    def set_action(self, action, joint_limit_type=None):
        self.set_delta_tilt_speed(float(action[0]), joint_limit_type)
        self.set_delta_pan_speed(float(action[1]), joint_limit_type)
        self.set_delta_vergence_position(float(action[2]), joint_limit_type)

    def set_position(self, position, joint_limit_type):
        self.set_tilt_position(position[0], joint_limit_type)
        self.set_pan_position(position[1], joint_limit_type)
        self.set_vergence_position(position[2], joint_limit_type)

    def get_vision(self):
        return self.cam_left.capture_rgb(), self.cam_right.capture_rgb()

    tilt_position = property(get_tilt_position)
    pan_position = property(get_pan_position)
    vergence_position = property(get_vergence_position)
    position = property(get_position)
    tilt_speed = property(get_tilt_speed)
    pan_speed = property(get_pan_speed)
    speed = property(get_speed)


if __name__ == "__main__":
    import time

    env = Environment(False)
    env.step()
    env.robot.set_position([0, 0, 0], joint_limit_type="none")
    env.step()
    time.sleep(2)
    t0 = time.time()
    for i in range(10):
        env.episode_reset()
        for j in range(15):
            time.sleep(0.1)
            env.step()
    # for i in range(15):
    #     env.robot.set_delta_vergence_position(1)
    #     print(env.robot.get_vergence_position())
    #     time.sleep(0.5)
    # for i in range(30):
    #     env.robot.set_delta_vergence_position(-1)
    #     print(env.robot.get_vergence_position())
    #     time.sleep(0.5)
    # for i in range(10):
    #     env.robot.set_delta_vergence_position(1)
    #     print(env.robot.get_vergence_position())
    #     time.sleep(0.5)
    # for i in range(10):
    #     env.robot.set_pan_position(i)
    #     print(env.robot.get_pan_position())
    #     time.sleep(0.5)
        # env.episode_reset()
        # for i in range(10):
        #     env.robot.set_delta_vergence_position(1)
        #     env.step()
        #     env.robot.get_vision()
    t1 = time.time()
    env.close()
    print("About {} sec per episode".format((t1 - t0) / 10))
    print("About {} iterations per sec".format(100 / (t1 - t0)))
