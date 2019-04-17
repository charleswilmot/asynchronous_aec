import os
import getpass
import numpy as np

# Gazebo Axes
# X: Depth
# Y: Horizontal
# Z: Vertical

# eyes position offset in x dimension [m]
# robot's head relaxes downwards at the start of an experiment (vrep??)
# X_EYES_POSITION is the final position
X_EYES_POSITION = 0 # Vrep Scene: Eyes on Depth 0
# interocular distance [m]
Y_EYES_DISTANCE = 0.034000 + 0.034000
Z_EYES_DISTANCE = 0 # Vrep Scene: Eyes on Vertical 0

X_DEFAULT_SCREEN_POS = 2
Y_DEFAULT_SCREEN_POS = 0
# Z_DEFAULT_SCREEN_POS ^= center of screen
Z_DEFAULT_SCREEN_POS = 0 # Vrep Scene: Eyes on Vertical 0
ROLL_DEFAULT_SCREEN_POS = np.pi / 2
PITCH_DEFAULT_SCREEN_POS = 0
YAW_DEFAULT_SCREEN_POS = -np.pi / 2

SCREEN_WIDTH = 1.5

MAX_TILT = 30  # (70 in the icub_head .ini conf file)
MAX_PAN = 30  # 60 --> 30 in simulator
MAX_VERGENCE = 17
MOTION_DONE_THRESHOLD = 0.01

ITERATION_CAMERA_FRAMERATE = 13

YARP_STD_PORT = 10000
GAZEBO_STD_PORT = 11311

GROUP_PATH = "/home/aecgroup/aecdata"
YARP_PATH = GROUP_PATH + "/Software/yarp/install/bin/"
YARPSERVER_CMD = YARP_PATH + "yarpserver"
YARPVIEWER_CMD = YARP_PATH + "yarpview"

IMAGE_SHAPE = (240, 320, 3)
# IMAGE_SHAPE = (480, 640, 3)

SCREEN_PATH = "/tmp/" + getpass.getuser() + "/screen/"
SCREEN_SDF_PATH = SCREEN_PATH + "screen.sdf"
SCREEN_CONFIG_PATH = SCREEN_PATH + "vtcfg.ini"

SCREEN_SDF = """<?xml version='1.0'?>
<sdf version="1.4">
  <model name="{model_name}">
    <pose>"""+ str(X_DEFAULT_SCREEN_POS) +""" """+ str(Y_DEFAULT_SCREEN_POS) +""" """+ str(Z_DEFAULT_SCREEN_POS) +""" """+ str(np.pi/2) +""" 0 """+ str(-np.pi/2) +"""</pose>
    <!--static>true</static-->
    <link name="{model_name}_link">
      <pose>0 0 0 0 0 0</pose>
      <visual name="{model_name}_visual">
        <geometry>
          <plane>
            <size>{height} {width}</size>
            <normal>0 0 1</normal>
          </plane>
        </geometry>
        <material>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
        </material>
        <plugin name="VideoTexture" filename="libgazebo_yarp_videotexture.so">
          <yarpConfigurationFile>file://""" + SCREEN_CONFIG_PATH + """</yarpConfigurationFile>
        </plugin>
      </visual>
      <gravity>0</gravity>
    </link>
  </model>
</sdf>
"""

SCREEN_CONFIG = """name /{name}
defaultSourcePortName /{source_port_name}
heightRes {height}
widthRes  {width}
heightLen {height_len}
"""

current_path = os.path.dirname(os.path.realpath(__file__))
REPOSITORY_PATH = os.path.realpath(current_path[:-len(current_path.split("/")[-1]) - 1])
DEFAULT_MODEL_CONFIG_NAME = "default_model.conf"
DEFAULT_MODEL_CONFIG = os.path.realpath(REPOSITORY_PATH + "/conf/cacla_var/" + DEFAULT_MODEL_CONFIG_NAME)
DEFAULT_EPERIMENT_CONFIG_NAME = "default_experiment.conf"
DEFAULT_EPERIMENT_CONFIG = os.path.realpath(REPOSITORY_PATH + "/conf/cacla_var/" + DEFAULT_EPERIMENT_CONFIG_NAME)
DEFAULT_PLOT_CONFIG_NAME = "default_plot.conf"
DEFAULT_PLOT_CONFIG = os.path.realpath(REPOSITORY_PATH + "/conf/cacla_var/" + DEFAULT_PLOT_CONFIG_NAME)
DEFAULT_HYPER_CONFIG_NAME = "default_hyper.conf"
DEFAULT_HYPER_CONFIG = os.path.realpath(REPOSITORY_PATH + "/conf/cacla_var/" + DEFAULT_HYPER_CONFIG_NAME)

# to be removed
# TRANSITING_CONFIG_PATH = GROUP_PATH + '/.transiting_config_file/'
# HYPER_RESULTS_PATH = GROUP_PATH + '/Results_python/hyper_results/'
