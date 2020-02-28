from environment.environment import *
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    env = Environment(headless=True)
    env.robot.reset_speed()
    env.robot.set_position([0, 0, 0], joint_limit_type="none")
    env.screen.set_texture(0)
    # env.screen.set_trajectory(1, 90 / 320 * 20, 0, preinit=True)
    env.screen.set_trajectory(1, 0, 0, preinit=True)
    for i in range(10):
        env.robot.set_delta_tilt_speed(90 / 320)
        env.step()  # moves the screen... for the next iteration!
        left, right = env.robot.get_vision()
        plt.imshow(left)
        plt.show()
        print(env.robot.speed, env.screen.tilt_pan_speed)
