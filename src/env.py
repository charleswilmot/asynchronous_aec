from enviroment.environment import *

if __name__ == "__main__":
    import time

    env = Environment(texture="/home/julian/PycharmProjects/VisionMA/res/textures/mcgillManMade_600x600_bmp_selection/",
                      scene="/home/julian/PycharmProjects/VisionMA/res/vrep_scenes/stereo_vision_robot.ttt",
                      headless=False)
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