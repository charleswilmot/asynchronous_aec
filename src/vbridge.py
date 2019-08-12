import socket
import os, signal
import subprocess as sub
import vrep
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import RESSOURCES

# Module containing classes for controling robots in vRep (warning:chaos)

danceModelPaths = ['./models/furniture/tables/conference table.ttm',
'./models/furniture/tables/diningTable.ttm',
'./models/furniture/tables/high table.ttm',
'./models/furniture/tables/customizable table.ttm',
'./models/furniture/chairs/conference chair.ttm',
'./models/furniture/chairs/sofa.ttm',
'./models/furniture/chairs/swivel chair.ttm',
'./models/furniture/chairs/dining chair.ttm',
'./models/furniture/plants/indoorPlant.ttm',
'./models/furniture/shelves-cupboards-racks/rack.ttm',
'./models/furniture/shelves-cupboards-racks/deep cupboard.ttm',
'./models/office items/projector screen.ttm',
'./models/office items/laptop.ttm',
'./models/office items/projector.ttm',
'./models/people/Standing Bill.ttm',
'./models/people/Working Bill.ttm',
'./models/people/Walking Bill.ttm',
'./models/people/Sitting Bill.ttm',
'./models/people/path planning Bill.ttm',
'./people/mannequin.ttm',
'./people/Bill on path.ttm']

class objectHandles:
    def __init__(self):
        self.tiltJointLeft = "vs_eye_tilt_left#"
        self.tiltJointRight = "vs_eye_tilt_right#"
        self.panJointLeft = "vs_eye_pan_left#"
        self.panJointRight = "vs_eye_pan_right#"
        self.worldCenter = "vs_center#"
        self.camLeft = "vs_cam_left#"
        self.camRight = "vs_cam_right#"


class quickConf:
    def __init__(self):
        self.scene_path = "/home/aecgroup/aecdata/Software/vrep_scenes/aec_pursuit_proxi_2_no_screen_mars_fast_physics_90deg_2.ttt"
        self.vrep_location = "/home/aecgroup/aecdata/Software/vrep/vrep.sh"
        self.sim_mode_sync = True
        self.sim_mode_headless = True
        self.sim_debug = False


class Simulator:
    def __init__(self, names, params, port=None):
        self.names = names
        self.vrep_location = params.vrep_location
        self.startSimulator(params.sim_mode_headless, params.sim_debug, port)
        connectionFailed = self.connect(10)
        if connectionFailed:
            print("Connecting to vRep failed :(")
            exit()
        self.loadScene(params.scene_path)
        self.startSimulation(params.sim_mode_sync)

    def __del__(self):
        self.stopSimulator()

    def startSimulator(self, headless = False, debug = False, port=None):
        cmd = self.vrep_location
        if headless:
            cmd += ' -h'
        if debug:
            cmdDebug = 'TRUE'
            self.debug = True
        else:
            cmdDebug = 'FALSE'
            self.debug = False
        self.vrep_port = self._get_open_port() if port is None else port
        print('Selected {0} as simulator port.'.format(self.vrep_port))

        cmd += ' -gREMOTEAPISERVERSERVICE_{0}_{1}_FALSE'.format(self.vrep_port, cmdDebug)
        if debug:
            self.sim_sub_process = sub.Popen(cmd, shell=True, preexec_fn=os.setsid)
        else:
            self.sim_sub_process = sub.Popen(cmd, stdout=sub.PIPE,shell=True, preexec_fn=os.setsid)

    ## Executes a portscan within given range and stepsize on local host
    #  @param start_port start #port of portscan range
    #  @param end_port end #port of portscan range
    #  @param step stepsize of portscan
    #  @return 1st open #port in given range
    #          -1 if no open port was found in given range
    #  @todo implement error handling
    def _get_open_port(self, start_port=19000, end_port=65535, step=1):  # todo: remove end_port
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        port = start_port
        while is_port_in_use(port):
            port += 1
        return port

    def stopSimulator(self):
        self.stopSimulation()
        self.disconnect()
        os.killpg(os.getpgid(self.sim_sub_process.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.sim_sub_process.pid), signal.SIGKILL)

    def connect(self, retries=3):
        self.clientID = -1
        while self.clientID == -1 and retries > 0:
            vrep.simxFinish(-1) # just in case, close all opened connections
            self.clientID = vrep.simxStart('127.0.0.1', self.vrep_port, True, True, 5000, 5) # Connect to V-REP
            sleep(1)
            retries += -1
        if self.clientID == -1:
            return 1
        else:
            print('Connected to Simulator succesfully after {0} tries.'.format(retries))
            return 0

    def disconnect(self):
        #ping = vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)
        print("Disconnected.")

    def getHandleByName(self, name):
        ack, handle = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)
        if self.debug:
            print("getHandle Ack:", ack)
        return handle

    def loadScene(self, scenePath):
        ack = vrep.simxLoadScene(self.clientID, scenePath, 0, vrep.simx_opmode_blocking)
        return ack

    def startSimulation(self, syncedMode=False):
        self.syncedMode = syncedMode
        ackSync = vrep.simxSynchronous(self.clientID, syncedMode)
        ackStart = vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        return ackStart or ackSync

    def stopSimulation(self):
        ackStop = vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        ackClose = vrep.simxCloseScene(self.clientID, vrep.simx_opmode_blocking)
        return ackStop or ackClose

    def buildHandleDict(self):
        self.handles = {}
        nameKeys = list(self.names.__dict__)
        for key in nameKeys:
            name = self.names.__dict__[key]
            self.handles[key] = self.getHandleByName(name)

    def stepSimulation(self):
        if self.syncedMode:
            ackTrigger = vrep.simxSynchronousTrigger(self.clientID)
            return ackTrigger
        else:
            return 0

    def waitForSimulation(self):
        if self.syncedMode:
            _, ping = vrep.simxGetPingTime(self.clientID)
            return ping
        else:
            return 0


class Universe:
    def __init__(self, gui=False, port=None):
        conf = quickConf()
        if gui:
            conf.sim_mode_headless = False
        self.sim = Simulator(objectHandles(), conf, port=port)
        self.sim.buildHandleDict()
        self.robot = Robot(self.sim, True)

    def __del__(self):
        self.sim.stopSimulator()


class Robot:
    def __init__(self, simulator, rgbEyes = False):
        self.sim = simulator
        self.rgbEyes = rgbEyes
        # Init Streaming for both eyes
        resCam1 = vrep.simxGetVisionSensorImage(self.sim.clientID,
                                                self.sim.handles["camLeft"],
                                                not rgbEyes,
                                                vrep.simx_opmode_streaming)

        resCam2 = vrep.simxGetVisionSensorImage(self.sim.clientID,
                                                self.sim.handles["camRight"],
                                                not rgbEyes,
                                                vrep.simx_opmode_streaming)

        if (resCam1[0] or resCam2[0]) == vrep.simx_return_ok:
            print("Cam Streaming Init: Ok")
        else:
            print("Cam Streaming Init returned: lcam {0}, rcam {1}".format(resCam1[0], resCam2[0]))

        # Init Joint Positions
        self.posPanLeft = 0
        self.posPanRight = 0

    def receiveImages(self):
        resA, resolution, imgCamLeft = vrep.simxGetVisionSensorImageFast(self.sim.clientID,
                                                                         self.sim.handles["camLeft"],
                                                                         not self.rgbEyes,
                                                                         vrep.simx_opmode_blocking)

        resB, _, imgCamRight = vrep.simxGetVisionSensorImageFast(self.sim.clientID,
                                                                 self.sim.handles["camRight"],
                                                                 not self.rgbEyes,
                                                                 vrep.simx_opmode_blocking)

        imgCamLeft = decodeImageF(imgCamLeft, resolution, self.rgbEyes)
        imgCamRight = decodeImageF(imgCamRight, resolution, self.rgbEyes)
        return imgCamLeft, imgCamRight

    def setJointPositionByHandle(self, handle, pos):
        vrep.simxSetJointPosition(self.sim.clientID, handle, pos, vrep.simx_opmode_oneshot)

    def setTiltJointPositions(self, pos):
        self.current_tilt = pos
        pos = -pos #Invert for convenience
        posConverted = pos*np.pi/180
        retLeft = vrep.simxSetJointPosition(self.sim.clientID,
                                            self.sim.handles["tiltJointLeft"],
                                            posConverted,
                                            vrep.simx_opmode_oneshot)

        retRight = vrep.simxSetJointPosition(self.sim.clientID,
                                             self.sim.handles["tiltJointRight"],
                                             posConverted,
                                             vrep.simx_opmode_oneshot)

        return retLeft + retRight

    def setPanJointPositions(self, vers, verg):
        self.current_version = float(vers)
        self.current_vergence = float(verg)

        angleLeft = (vers + verg / 2) * np.pi / 180 # VREP Calculates in Rad
        angleRight = (vers - verg / 2) * np.pi / 180

        retLeft = vrep.simxSetJointPosition(self.sim.clientID,
                                            self.sim.handles["panJointLeft"],
                                            angleLeft,
                                            vrep.simx_opmode_oneshot)

        retRight = vrep.simxSetJointPosition(self.sim.clientID,
                                             self.sim.handles["panJointRight"],
                                             angleRight,
                                             vrep.simx_opmode_oneshot)
        return retLeft + retRight

    # aec_py compliance methods
    def getEyePositions(self):
        return (self.current_tilt, self.current_version, -self.current_vergence)

    def setEyePositions(self, command): # command = [tilt, pan, vergence]
        self.setPanJointPositions(command[1], -command[2])
        self.setTiltJointPositions(command[0])


class SimObject:
    def __init__(self, simulator, objectPath, clientSideLoading=0):
        self.objectType = 'general'
        self.sim = simulator
        ack, self.baseHandle = self.__createObject(objectPath, clientSideLoading)
        if ack != 0:
            print('Creating Object failed ({})'.format(objectPath))

    def __del__(self):
        vrep.simxRemoveModel(self.sim.clientID,
                             self.baseHandle,
                             vrep.simx_opmode_blocking)

    def __createObject(self, objectPath, clientSideLoading):
        ack, objectHandle = vrep.simxLoadModel(self.sim.clientID,
                                               objectPath,
                                               clientSideLoading,
                                               vrep.simx_opmode_blocking)
        return ack, objectHandle

    def getChildHandle(self, childNum):
        ackChild, handleChild = vrep.simxGetObjectChild(self.sim.clientID,
                                                        self.baseHandle,
                                                        childNum,
                                                        vrep.simx_opmode_blocking)
        return handleChild

    def setPosition(self, pos):
        newPosition = (-pos[0], -pos[1], pos[2]) # horizontal, distance, vertical
        return vrep.simxSetObjectPosition(self.sim.clientID,
                                          self.baseHandle,
                                          self.sim.handles["worldCenter"],
                                          newPosition,
                                          vrep.simx_opmode_oneshot)

    def getPosition(self):
        ack, pos = vrep.simxGetObjectPosition(self.sim.clientID,
                                              self.baseHandle,
                                              self.sim.handles["worldCenter"],
                                              vrep.simx_opmode_blocking)
        posConverted = (-pos[0], -pos[1], pos[2])
        return posConverted

    def move(self, posDelta):
        currentPos = self.getPosition()
        newPosition = (currentPos[0] + posDelta[0],
                       currentPos[1] + posDelta[1],
                       currentPos[2] + posDelta[2])
        return self.setPosition(newPosition)

    def safePosDelta(self, posDelta):
        self.posDelta = posDelta

    def set_positions(self, gazebo_x, gazebo_y, gazebo_z, roll=np.pi / 2, pitch=0, yaw=-np.pi / 2): # x: Depth, y: Horizontal, z: Vertical
        self.setPosition((gazebo_y, gazebo_x, gazebo_z))

    ## Move this object in spherical coordinates
    #  @param dist distance to the point specified by {x,y,z}_offset
    #  @param pan pan angle
    #  @param tilt tilt angle
    #  @param x_offset x coordinate of the center of the spherical coordinate system
    #  @param y_offset y coordinate of the center of the spherical coordinate system
    #  @param z_offset z coordinate of the center of the spherical coordinate system
    def set_spherical_positions(self, dist, direction, angle):
        raddirection, radangle = np.deg2rad(direction), np.deg2rad(angle)
        sindirection, cosdirection = np.sin(raddirection), np.cos(raddirection)
        sinangle, cosangle = np.sin(radangle), np.cos(radangle)
        x = dist * cosangle
        y = dist * cosdirection * sinangle
        z = dist * sindirection * sinangle
        pos = np.array([x, y, z])
        self.set_positions(*pos)
        return x, y, z


class Screen(SimObject):
    def __init__(self, simulator, position=(0, 1, 0)):
        SimObject.__init__(self, simulator, RESSOURCES.REPOSITORY_PATH+'/scene/screen_res600_size1_5m.ttm', 1)
        self.objectType = 'screen'
        self.displayHandle = self.getChildHandle(0) # Display Connection is child 0
        self.setPosition(position)

    def setImage(self, image):
        shapeProduct = np.prod(image.shape)
        convertedImage = np.reshape(image, shapeProduct)
        return vrep.simxSetVisionSensorImage(self.sim.clientID,
                                             self.displayHandle,
                                             convertedImage,
                                             0,
                                             vrep.simx_opmode_blocking)

    def set_texture(self, texture):
        self.setImage(texture)


class ConstantSpeedScreen(Screen):
    def __init__(self, simulator, min_max_distance, max_distance_speed, max_angle_speed, textures):
        self.min_max_distance = min_max_distance
        self.max_distance_speed = max_distance_speed
        self.max_angle_speed = max_angle_speed
        self.textures = textures
        np.random.shuffle(self.textures)
        self.n_textures = self.textures.shape[0]
        super().__init__(simulator, (-2, 0, RESSOURCES.Z_DEFAULT_SCREEN_POS))
        self.init_constant_speed_objects()

    def update_constant_speed_objects(self):
        self.position += self.speeds
        self.set_spherical_positions(*self.position)

    def put_to_position(self):
        self.set_spherical_positions(*self.position)

    def init_constant_speed_objects(self):
        self.speeds = np.array([
            np.random.uniform(low=-self.max_distance_speed, high=self.max_distance_speed),
            0,
            np.random.uniform(low=-self.max_angle_speed, high=self.max_angle_speed)])
        self.position = np.array([
            np.random.uniform(low=self.min_max_distance[0], high=self.min_max_distance[1]),
            np.random.uniform(low=0, high=180),
            0])
        self.set_spherical_positions(*self.position)

    def update_textures(self):
        i = np.random.randint(low=0, high=self.n_textures)
        self.set_texture_by_index(i)

    def set_texture_by_index(self, index):
        self.set_texture(self.textures[index])

    def episode_init(self):
        self.update_textures()
        self.init_constant_speed_objects()

    def iteration_init(self):
        self.update_constant_speed_objects()

    def hide(self):
        self.set_positions(-2, 0, RESSOURCES.Z_DEFAULT_SCREEN_POS)

    def set_movement(self, distance, direction, angle, distance_speed, direction_speed, angle_speed):
        self.position[:] = (distance, direction, angle)
        self.speeds[:] = (distance_speed, direction_speed, angle_speed)

    def _get_distance(self):
        return self.position[0]
        # return self.getPosition()[0]

    distance = property(_get_distance)


class DanceInstructor:
    def __init__(self, simulator,robot, numDancers):
        self.sim = simulator
        self.robot = robot
        self.__initObjects(numDancers)

    def getRandomPosition(self):
        horizontal = (np.random.rand()-.5)*3
        vertical = (np.random.rand()-.5)*3
        distance = np.random.rand()*3 +6
        return (horizontal, distance, vertical)

    def getRandomObjectPath(self):
        num = np.random.randint(0, len(danceModelPaths))
        return danceModelPaths[num]

    def __initObjects(self, numDancers):
        self.dancers = []
        for _ in range(numDancers-2):
            dancer = SimObject(self.sim, self.getRandomObjectPath())
            dancer.setPosition(self.getRandomPosition())
            self.dancers.append(dancer)
        sc1 = Screen(self.sim)
        sc2 = Screen(self.sim)
        sc1.setPosition(self.getRandomPosition())
        sc2.setPosition(self.getRandomPosition())
        self.dancers.append(sc1)
        self.dancers.append(sc2)



    def dance(self, dances, stepsPerDance, images = []):
        for dance in range(dances):
            print('Dance:' ,dance)
            for dancer in self.dancers:
                posDelta = ((np.random.rand()-.5)/5,
                            (np.random.rand()-.5)/15,
                            (np.random.rand()-.5)/5)
                dancer.safePosDelta(posDelta)
                if dancer.objectType == 'screen' and len(images) > 0:
                    dancer.setImage(images[np.random.randint(0, len(images))])

            follow = self.dancers[np.random.randint(0, len(self.dancers))]

            for currentStep in range(stepsPerDance):
                print('Step:', currentStep)
                for dancer in self.dancers:
                    dancer.move(dancer.posDelta)
                followPos = follow.getPosition()
                vers, verg, tilt = stimPositionToAngles(followPos[0], followPos[1], followPos[2])
                self.robot.setPanJointPositions(vers, verg)
                self.robot.setTiltJointPositions(tilt)
                self.sim.stepSimulation()


def decodeImage(encodedImage, resolution, rgb):
    castedImage = np.array(encodedImage, dtype=np.uint8)
    if rgb:
        reshapedImage = castedImage.reshape((resolution[1],resolution[0],3))
    else:
        reshapedImage = castedImage.reshape((resolution[1],resolution[0]))
    flippedImage = np.flip(reshapedImage,axis=0)
    return flippedImage

def decodeImageF(encodedImage, resolution, rgb):
    if rgb:
        reshapedImage = encodedImage.reshape((resolution[1],resolution[0],3))
    else:
        reshapedImage = encodedImage.reshape((resolution[1],resolution[0]))
    flippedImage = np.flip(reshapedImage,axis=0)
    return flippedImage

def stimPositionToAngles(horizontal, distance, vertical):
    baseline = 0.068

    vers = np.degrees(np.arctan(horizontal / distance))
    verg = np.degrees(np.arctan(baseline/2 / distance))*2
    tilt = np.degrees(np.arctan(vertical / distance))

    return vers, verg, tilt

# Copy from gazebo.py
def spherical_to_cartesian(dist, pan, tilt,
                           x_offset=RESSOURCES.X_EYES_POSITION,
                           y_offset=0,
                           z_offset=RESSOURCES.Z_DEFAULT_SCREEN_POS):
    radpan, radtilt = np.deg2rad(pan), np.deg2rad(tilt)
    sinpan, cospan = np.sin(radpan), np.cos(radpan)
    sintilt, costilt = np.sin(radtilt), np.cos(radtilt)
    x = dist * cospan * costilt
    y = dist * sinpan * costilt
    z = dist * sintilt
    return np.array([x + x_offset, y + y_offset, z + z_offset])


def spherical_to_roll_pitch_yaw(dist, pan, tilt):
    return np.pi / 2 + np.deg2rad(tilt), 0, -np.pi / 2 + np.deg2rad(pan)


#def setPerfectPosition(horz,dist,vert):
#    vers, verg, tilt = stimPositionToAngles(horz,dist,vert)
#    robot.setPanJointPositions(vers, verg)
#    robot.setTiltJointPositions(tilt)
#    screen.setPosition(horz, dist, vert)

if __name__ == '__main__':
    from imageio import imread
    # Load images
    imageShape = [1024,1024,3]
    imageSize =  np.prod(imageShape)
    testImages = []
    testImages.append(np.reshape(imread("/home/aecgroup/aecdata/Software/vrep_scenes/test_img.png"), imageSize))
    testImages.append(np.reshape(imread("/home/aecgroup/aecdata/Software/vrep_scenes/test_img2.png"), imageSize))

    #plt.ion()
    #imgA, imgB = robot.receiveImages()
    #ax1 = plt.subplot(1,2,1)
    #ax2 = plt.subplot(1,2,2)
    #plot1 = ax1.imshow(imgA)
    #plot2 = ax2.imshow(imgB)

    uni = Universe()
    di = DanceInstructor(uni.sim,uni.robot, 2)
    print('Started dancing!')
    di.dance(8, 8, testImages)
    print('Stopped dancing.')

    sleep(3)
