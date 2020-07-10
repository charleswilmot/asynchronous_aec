from os import listdir
from pyrep import PyRep
from pyrep.const import TextureMappingMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
import numpy as np
#from helper.utils import deg, rad, to_angle

# interocular distance [m]
Y_EYES_DISTANCE = 0.034000 + 0.034000

deg = np.rad2deg
rad = np.deg2rad

def to_angle(other_distance):
    return deg(2 * np.arctan2(Y_EYES_DISTANCE, 2 * other_distance))

def vergence_error(eyes_positions, object_distances):
    vergences = eyes_positions[..., -1]
    return to_angle(object_distances) - vergences


class RandomScreen(Shape):
    def __init__(self, min_distance, max_distance, max_speed_in_deg, textures_list, id=0):
        if not id:
            shape = Shape("vs_screen#")
        else:
            shape = Shape("vs_screen_{}#".format(id))
        self.id = id
        self._handle = shape.get_handle()
        self.size = 1.5
        self.textures_list = textures_list
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_speed = rad(max_speed_in_deg)
        self.episode_reset()

    def set_texture(self, index=None):
        if index is None:
            super().set_texture(self.textures_list[np.random.randint(len(self.textures_list))],
                                    TextureMappingMode.CUBE, interpolate=False, uv_scaling=[self.size, self.size])
        else:
            super().set_texture(self.textures_list[index],
                                    TextureMappingMode.CUBE, interpolate=False, uv_scaling=[self.size, self.size])

    def episode_reset(self, preinit=False):
        np.random.seed(None)
        self.start_distance = np.random.uniform(self.min_distance, self.max_distance)
        self.distance = self.start_distance
        self.delta_distance = np.random.choice([-0.00, 0.00]) #np.random.uniform(-0.035, 0.035)
        #self.delta_distance = np.random.uniform(-0.025, 0.025)
        # self.angle = np.random.uniform(rad(0.0), rad(40))
        # self.speed = np.random.uniform(rad(0.0), rad(1.125))
        self.angle = np.random.uniform(rad(10.0), rad(30))
        self.speed = np.random.uniform(rad(0.0), rad(0.0))
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.delta_direction = np.random.uniform(-2 * np.pi, 2 * np.pi) * 0.01
        self.set_texture()
        self.set_episode_iteration(-1 if preinit else 0)

    def set_trajectory(self, distance, tilt_speed_deg, pan_speed_deg, depth_speed, preinit=False):
        self.start_distance = distance
        self.distance = self.start_distance
        self.delta_distance = depth_speed
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
        if self.id:
            self.set_position(self.position_circular)
        else:
            self.set_position(self.position)
        #self.set_orientation(self.orientation)

    def increment_iteration(self):
        self.set_episode_iteration(self._episode_iteration + 1)

    def _get_circular_movement_position(self):
        cos_speed = np.cos(self.angle)
        sin_speed = np.sin(self.angle)
        cos_dir = np.cos(self.direction + self._episode_iteration * self.delta_direction)
        sin_dir = np.sin(self.direction + self._episode_iteration * self.delta_direction)
        self.distance = self.start_distance + (self._episode_iteration * self.delta_distance)
        # if self.distance > self.max_distance or self.distance < self.min_distance:
        #     self.delta_distance = -self.delta_distance
        x = self.distance * sin_speed * cos_dir
        y = self.distance * cos_speed
        z = self.distance * sin_speed * sin_dir
        return [x, y, z]

    def _get_position(self):
        cos_speed = np.cos(self._episode_iteration * self.speed)
        sin_speed = np.sin(self._episode_iteration * self.speed)
        cos_dir = np.cos(self.direction)
        sin_dir = np.sin(self.direction)
        self.distance = self.start_distance + (self._episode_iteration * self.delta_distance)
        x = self.distance * sin_speed * cos_dir
        y = self.distance * cos_speed
        z = self.distance * sin_speed * sin_dir
        return [x, y, z]

    def _get_orientation(self):
        tilt_speed, pan_speed = self.tilt_pan_speed
        tilt, pan = tilt_speed * self._episode_iteration, pan_speed * self._episode_iteration
        x = rad(90 + tilt)
        y = rad(-pan)
        z = 0
        return [x, y, z]

    def _get_tilt_pan_speed(self):
        return [deg(self.speed * np.sin(self.direction)), deg(self.speed * np.cos(self.direction))]

    orientation = property(_get_orientation)
    position = property(_get_position)
    position_circular = property(_get_circular_movement_position)
    tilt_pan_speed = property(_get_tilt_pan_speed)


class Environment:
    def __init__(self,
                 texture="/home/aecgroup/aecdata/Textures/mcgillManMade_600x600_png_selection/",
                 scene="/home/aecgroup/aecdata/Software/vrep_scenes/stereo_vision_robot_collection_mt.ttt",
                 headless_scene="/home/aecgroup/aecdata/Software/vrep_scenes/stereo_vision_robot_collection_mt.ttt",
                 num_screens=2,
                 headless=True
                 ):
        self.pyrep = PyRep()
        if headless:
            #self.pyrep.launch(headless_scene, headless=headless)
            self.pyrep.launch(scene, headless=headless)
        else:
            self.pyrep.launch(scene, headless=headless)
        min_distance = 3
        max_distance = 6
        max_speed = 0.5
        path = texture
        textures_names = listdir(path)
        textures_list = [self.pyrep.create_texture(path + name)[1] for name in textures_names]
        max_num_screens = 6
        if num_screens > max_num_screens:
            num_screens = max_num_screens
            print("Only {} screens supported! Reducing number of screens!".format(max_num_screens))
        self.screens = [RandomScreen(min_distance, max_distance, max_speed, textures_list, id=id) for id in range(num_screens)]
        self.robot = StereoVisionRobot(min_distance, max_distance)
        self.pyrep.start()

    def step(self):
        # step simulation
        self.pyrep.step()
        # move screen
        for screen in self.screens:
            screen.increment_iteration()

    def episode_reset(self, preinit=False):
        # reset screen
        for screen in self.screens:
            screen.episode_reset(preinit=preinit)
        # reset robot
        self.robot.episode_reset()
        # self.robot.set_vergence_position(to_angle(self.screen.distance))
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
        self._pan_max = rad(10)
        self._tilt_max = rad(10)
        self._tilt_speed = 0
        self._pan_speed = 0
        self.episode_reset()

    # def episode_reset(self, vergence=True):
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
            self.tilt_left.get_joint_position() - self._tilt_speed,  # minus to get natural upward movement
            self.tilt_right.get_joint_position() - self._tilt_speed,  # minus to get natural upward movement
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
        return self.vergence_position - to_angle(other_distance)

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

def select_random_point(array):
    slice_width = 4
    rnd_index_x = np.random.randint(slice_width, array.shape[0]-slice_width)
    rnd_index_y = np.random.randint(slice_width, array.shape[1]-slice_width)
    array[rnd_index_x-slice_width:rnd_index_x+slice_width, rnd_index_y-slice_width:rnd_index_y+slice_width, ...] = [1, 0, 0]
    return array, rnd_index_x, rnd_index_y

def distance_center(x, y, width, height):
    center_x = width//2
    center_y = height//2
    x_target = x-center_x
    y_target = y - center_y
    return x_target, y_target

def pan_tilt_deg(x, y):
    deg_per_pix = 90/256
    return x*deg_per_pix, y*deg_per_pix

def lines(array):
    array[:, 64:66, ...] = [0,0.5,0]
    array[64:66, ...] = [0,0.5,0]
    array[:, 192:194, ...] = [0,0.5,0]
    array[192:194, ...] = [0,0.5,0]
    array[:, 128:130, ...] = [0,1,0]
    array[128:130, ...] = [0, 1, 0]
    return array

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    #env = Environment()
    env = Environment(
        scene="/home/julian/PycharmProjects/VisionMA/src/resources/vrep_scenes/stereo_vision_robot_collection.ttt",
        texture="/home/julian/PycharmProjects/VisionMA/src/resources/textures/mcgillManMade_600x600_bmp_selection/",
        headless=True,
        num_screens=2
    )
    env.robot.set_position([0, 0, 0], joint_limit_type="none")
    env.step()
    for i in range(10):
        env.episode_reset()
        for i in range(20):
            env.step()
            if i == 10:
                left_clear, _ = env.robot.get_vision()
                left_point, x, y = select_random_point(left_clear.copy())
                left_point = lines(left_point)
                x, y = distance_center(x, y, 256, 256)
                delta_pan, delta_tilt = pan_tilt_deg(x, y)
                pan, tilt, vergence = env.robot.position
                env.robot.set_position([pan+delta_pan, tilt+delta_tilt, vergence], None)
                left_shifted, _ = env.robot.get_vision()
                left_shifted_lines = lines(left_shifted.copy())
                combined_clean = np.hstack([left_clear, left_shifted,])
                combined_lines = np.hstack([left_point, left_shifted_lines,])
                combined = np.vstack([combined_clean, combined_lines, ])
                plt.imshow(combined)
                plt.title("Move cam to target \n(original/shifted agent; \noriginal w. random target point/shifted screen)")
                plt.show()
            #time.sleep(0.2)
        # left, right = env.robot.get_vision()
        # plt.imshow(left)
        # plt.show()
    env.close()
