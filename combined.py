import mujoco
import mujoco.viewer
import numpy as np
import sys
import os #for ubuntu only
import time
from collections import deque
from threading import Lock

import xsensdeviceapi as xda #for windows only
import keyboard
import math
from pyquaternion import Quaternion 

class MujocoController:
    def __init__(self, model_path, dt=0.02):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        # 初始化渲染器 
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = [0, 0, 0.4]  # 设置目标点
        self.viewer.cam.distance = 2.5         # 设置摄像头距离
        self.viewer.cam.elevation = -25        # 设置俯仰角
        self.viewer.cam.azimuth = 90           # 设置方位角
        
        # 获取关节信息
        self.num_joints = self.model.njnt
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                           for i in range(self.num_joints)]
        self.actuators = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
                          for i in range(self.num_joints)]
        
        # PD控制器参数
        self.kp = [30, 10, 5, 5, 5, 5, 3, 3]  # 位置比例增益
        self.kd = [2.5, 2, 2, 0.01, 0.01, 0.01, 0.01, 0.01]   # 速度微分增益
        
    def get_joint_info(self):
        """获取关节信息与运动范围"""
        print(f"模型包含 {self.num_joints} 个关节:")
        for i, name in enumerate(self.joint_names):
            joint_range=self.model.jnt_range[i]
            print(f"{i}: {name},range: {joint_range[0]:<.2f} to {joint_range[1]:<.2f}")
        return self.joint_names
    
    def set_target_joint_positions(self, target_positions):
        """设置目标关节位置"""
        if len(target_positions) != self.num_joints:
            raise ValueError(f"目标位置数量({len(target_positions)})与关节数量({self.num_joints})不匹配")
        self.target_joint_positions = np.array(target_positions)
    
    def _pd_controller(self):
        """PD控制器计算控制力矩"""
        # 获取当前关节位置和速度
        current_positions = self.data.qpos[:self.num_joints]
        current_velocities = self.data.qvel[:self.num_joints]
        
        # 计算位置误差
        position_error = self.target_joint_positions - current_positions
        
        # 计算控制力矩 (tau = kp * error + kd * (-velocity))
        ctrl_torques = self.kp * position_error - self.kd * current_velocities
        
        # 应用控制力矩
        def set_control_value(mjdata, ctrl_value, actuators):
            for c, actuator in zip(ctrl_value, actuators):
                mjdata.actuator(actuator).ctrl = c

        set_control_value(mjdata=self.data, ctrl_value=ctrl_torques, actuators=self.actuators)
    
    def update(self):
        """更新可视化"""
        # 应用控制
        self._pd_controller()
        
        # 模拟一步
        mujoco.mj_step(self.model, self.data)

        # 更新渲染
        self.viewer.sync()
    
    def close(self):
        self.viewer.close()
        mujoco.mj_delete_data(self.data)
        mujoco.mj_delete_model(self.model)


class XsPortInfoStr:
    def __str__(self, p):
        return f"Port: {p.portNumber():>2} ({p.portName()}) @ {p.baudrate():>7} Bd, ID: {p.deviceId().toString()}"

class XsDeviceStr:
    def __str__(self, d):
        return f"ID: {d.deviceId().toString()} ({d.productCode()})"

def find_closest_update_rate(supported_update_rates, desired_update_rate):
    if not supported_update_rates:
        return 0

    if len(supported_update_rates) == 1:
        return supported_update_rates[0]

    closest_update_rate = min(supported_update_rates, key=lambda x: abs(x - desired_update_rate))
    return closest_update_rate

def average_quaternions(q_list):
    """对多个四元数简单平均并归一化"""
    x = sum([q.x for q in q_list]) / len(q_list)
    y = sum([q.y for q in q_list]) / len(q_list)
    z = sum([q.z for q in q_list]) / len(q_list)
    w = sum([q.w for q in q_list]) / len(q_list)
    # norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    # return Quaternion(w / norm, x / norm, y / norm, z / norm)
    return Quaternion(w, x, y, z)  # pyquaternion uses (w, x, y, z) order


class WirelessMasterCallback(xda.XsCallback):
    def __init__(self):
        super().__init__()
        self.m_connectedMTWs = set()
        self.m_mutex = Lock()

    def getWirelessMTWs(self):
        with self.m_mutex:
            return self.m_connectedMTWs.copy()

    def onConnectivityChanged(self, dev, newState):
        with self.m_mutex:
            if newState == xda.XCS_Disconnected:
                print(f"\nEVENT: MTW Disconnected -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_Rejected:
                print(f"\nEVENT: MTW Rejected -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_PluggedIn:
                print(f"\nEVENT: MTW PluggedIn -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_Wireless:
                print(f"\nEVENT: MTW Connected -> {dev.deviceId()}")
                self.m_connectedMTWs.add(dev)
            elif newState == xda.XCS_File:
                print(f"\nEVENT: MTW File -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            elif newState == xda.XCS_Unknown:
                print(f"\nEVENT: MTW Unknown -> {dev.deviceId()}")
                self.m_connectedMTWs.discard(dev)
            else:
                print(f"\nEVENT: MTW Error -> {dev.deviceId()}")

                
                self.m_connectedMTWs.discard(dev)

class MtwCallback(xda.XsCallback):
    def __init__(self, mtwIndex, device):
        super().__init__()
        self.m_packetBuffer = deque(maxlen=300)
        self.m_mutex = Lock()
        self.m_mtwIndex = mtwIndex
        self.m_device = device

    def dataAvailable(self):
        with self.m_mutex:
            return bool(self.m_packetBuffer)

    def getOldestPacket(self):
        with self.m_mutex:
            packet = self.m_packetBuffer[0]
            return packet

    def deleteOldestPacket(self):
        with self.m_mutex:
            self.m_packetBuffer.popleft()

    def getMtwIndex(self):
        return self.m_mtwIndex

    def device(self):
        assert self.m_device is not None
        return self.m_device

    def onLiveDataAvailable(self, _, packet):
        with self.m_mutex:
            # NOTE: Processing of packets should not be done in this thread.
            self.m_packetBuffer.append(packet)
            if len(self.m_packetBuffer) > 300:
                self.deleteOldestPacket()


if __name__ == "__main__":

    desired_update_rate = 100  # Hz
    desired_radio_channel = 19

    wireless_master_callback = WirelessMasterCallback()
    mtw_callbacks = []

    print("Constructing XsControl...")
    control = xda.XsControl.construct()
    if control is None:
        print("Failed to construct XsControl instance.")
        sys.exit(1)

    try:
        print("Scanning ports...")

        detected_devices = xda.XsScanner_scanPorts()

        print("Finding wireless master...")
        wireless_master_port = next((port for port in detected_devices if port.deviceId().isWirelessMaster()), None)
        if wireless_master_port is None:
            raise RuntimeError("No wireless masters found")

        print(f"Wireless master found @ {wireless_master_port}")

        print("Opening port...")
        if not control.openPort(wireless_master_port.portName(), wireless_master_port.baudrate()):
            raise RuntimeError(f"Failed to open port {wireless_master_port}")

        print("Getting XsDevice instance for wireless master...")
        wireless_master_device = control.device(wireless_master_port.deviceId())
        if wireless_master_device is None:
            raise RuntimeError(f"Failed to construct XsDevice instance: {wireless_master_port}")

        print(f"XsDevice instance created @ {wireless_master_device}")

        print("Setting config mode...")
        if not wireless_master_device.gotoConfig():
            raise RuntimeError(f"Failed to goto config mode: {wireless_master_device}")

        print("Attaching callback handler...")
        wireless_master_device.addCallbackHandler(wireless_master_callback)

        print("Getting the list of the supported update rates...")
        supportUpdateRates = xda.XsDevice.supportedUpdateRates(wireless_master_device, xda.XDI_None)

        print("Supported update rates: ", end="")
        for rate in supportUpdateRates:
            print(rate, end=" ")
        print()

        new_update_rate = find_closest_update_rate(supportUpdateRates, desired_update_rate)

        print(f"Setting update rate to {new_update_rate} Hz...")

        if not wireless_master_device.setUpdateRate(new_update_rate):
            raise RuntimeError(f"Failed to set update rate: {wireless_master_device}")

        print("Disabling radio channel if previously enabled...")

        if wireless_master_device.isRadioEnabled():
            if not wireless_master_device.disableRadio():
                raise RuntimeError(f"Failed to disable radio channel: {wireless_master_device}")

        print(f"Setting radio channel to {desired_radio_channel} and enabling radio...")
        if not wireless_master_device.enableRadio(desired_radio_channel):
            raise RuntimeError(f"Failed to set radio channel: {wireless_master_device}")

        print("Waiting for MTW to wirelessly connect...\n")


        # This function checks for user input to break the loop
        def user_input_ready():
            return False  # Replace this with your method to detect user input


        wait_for_connections = True
        connected_mtw_count = len(wireless_master_callback.getWirelessMTWs())
        while wait_for_connections:
            time.sleep(0.1)
            next_count = len(wireless_master_callback.getWirelessMTWs())
            if next_count != connected_mtw_count:
                print(f"Number of connected MTWs: {next_count}. Press 'Y' to start measurement.")
                connected_mtw_count = next_count

            wait_for_connections = not keyboard.is_pressed('y')



        print("Starting measurement...")
        if not wireless_master_device.gotoMeasurement():
            raise RuntimeError(f"Failed to goto measurement mode: {wireless_master_device}")

        print("Getting XsDevice instances for all MTWs...")
        all_device_ids = control.deviceIds()
        mtw_device_ids = [device_id for device_id in all_device_ids if device_id.isMtw()]
        mtw_devices = []
        for device_id in mtw_device_ids:
            mtw_device = control.device(device_id)
            if mtw_device is not None:
                mtw_devices.append(mtw_device)
            else:
                raise RuntimeError("Failed to create an MTW XsDevice instance")

        print("Attaching callback handlers to MTWs...")
        mtw_callbacks = [MtwCallback(i, mtw_devices[i]) for i in range(len(mtw_devices))]
        for i in range(len(mtw_devices)):
            mtw_devices[i].addCallbackHandler(mtw_callbacks[i])

        print("Creating a log file...")
        logFileName = "logfile.mtb"
        if wireless_master_device.createLogFile(logFileName) != xda.XRV_OK:
            raise RuntimeError("Failed to create a log file. Aborting.")
        else:
            print("Created a log file: %s" % logFileName)

        print("Starting recording...")
        ready_to_record = False

        while not ready_to_record:
            ready_to_record = all([mtw_callbacks[i].dataAvailable() for i in range(len(mtw_callbacks))])
            if not ready_to_record:
                print("Waiting for data available...")
                time.sleep(0.5)
            #     optional, enable heading reset before recording data, make sure all sensors have aligned physically the same heading!!
            # else:
            #     print("Do heading reset before recording data, make sure all sensors have aligned physically the same heading!!")
            #     all([mtw_devices[i].resetOrientation(xda.XRM_Heading) for i in range(len(mtw_callbacks))])

        if not wireless_master_device.startRecording():
            raise RuntimeError("Failed to start recording. Aborting.")

        print("\nMain loop. Press any key to quit\n")
        print("Waiting for data available...")

        euler_data = [xda.XsEuler()] * len(mtw_callbacks)
        quarterly_data = [xda.XsQuaternion(None)] * len(mtw_callbacks)   #储存实时四元数

        
        # For reference data collection
        reference_quat = [None] * len(mtw_callbacks)     #储存相对四元数
        reference_set = False

        SAMPLES_FOR_REFERENCE = 50        # Number of samples to collect for reference data
        reference_buffer = [[] for _ in range(len(mtw_callbacks))]
        wait_for_reference = True
        wait_for_reference1 = True

        print_counter = 0

        # 启动Mujoco
        model_path = "assets/Piper/scene.xml" 
        # 创建控制器
        controller = MujocoController(model_path)
        # 获取关节信息

        joint_names = controller.get_joint_info()  
        '''
        0: joint1,range: -2.62 to 2.62 躯干Yaw
        1: joint2,range: -1.57 to 1.57 
        2: joint3,range: -1.64 to 1.33
        3: joint4,range: -1.83 to 1.83 固定1.83
        4: joint5,range: -1.22 to 1.22 手肘Yaw
        5: joint6,range: -3.14 to 3.14 手腕Roll
        6: joint7,range: -0.04 to 0.00 夹爪开合
        7: joint8,range: -0.04 to 0.00 夹爪开合
        '''
        target_positions = np.zeros(controller.num_joints)
        #设定初始位置
        target_positions[0] = math.radians(0)  #大臂Yaw角————大臂IMUYaw角
        target_positions[1] = math.radians(-45)  
        target_positions[2] = math.radians(25)  #肩关节Pitch角————大臂IMUPitch角
        target_positions[3] = 1.57  # 固定值 
        target_positions[4] = math.radians(0)  # 手肘Yaw角————小臂IMUYaw角
        target_positions[5] = 0  # 假设手腕Roll角————手部IMURoll角
        target_positions[6] = 0  # 假设夹爪开合角度
        target_positions[7] = 0  # 假设夹爪开合角度


        while not user_input_ready() and controller.viewer.is_running():
            time.sleep(0)

            while wait_for_reference:
                time.sleep(0.1)
                if wait_for_reference1:
                    print(f" Press 'Space' to start getting reference orientation.")
                    wait_for_reference1=False                         # 使得仅打印一次
                wait_for_reference = not keyboard.is_pressed('space')

            if  not reference_set:
                print("Collecting reference data...keep the sensors still.")   # 手臂伸直平举，掌心朝下

                for sample in range(SAMPLES_FOR_REFERENCE):
                    time.sleep(0.1)           # Adjust the sleep time as needed

                    for i in range(len(mtw_callbacks)):
                        if mtw_callbacks[i].dataAvailable():
                            packet = mtw_callbacks[i].getOldestPacket()
                            #euler_data[i] = packet.orientationEuler() 世界坐标系下的绝对欧拉角
                            Q = packet.orientationQuaternion() 
                            reference_buffer[i].append(Quaternion(Q[0], Q[1], Q[2], Q[3]))  #(w, x, y, z) order for pyquaternion
                            quarterly_data[i] = Quaternion(Q[0], Q[1], Q[2], Q[3])  # Update the real-time quaternion data
                            mtw_callbacks[i].deleteOldestPacket()

                # Collect reference quaternion data
                for i in range(len(mtw_callbacks)):
                    if mtw_callbacks[i].dataAvailable():
                        Q_list=reference_buffer[i]
                        ave_Q= average_quaternions(Q_list)
                        reference_quat[i] = ave_Q

                reference_set = True
                print("Reference data collected.")
                print(f"Reference Quaternion: {[f'{q.w:.10f}, {q.x:.10f}, {q.y:.10f}, {q.z:.10f}' for q in reference_quat]}")

            # real-time data processing
            new_data_available = False
            for i in range(len(mtw_callbacks)):
                if mtw_callbacks[i].dataAvailable():
                    new_data_available = True
                    packet = mtw_callbacks[i].getOldestPacket()
                    #euler_data[i] = packet.orientationEuler()
                    quarterly_data[i] = packet.orientationQuaternion()  # Update the real-time quaternion data
                    mtw_callbacks[i].deleteOldestPacket()


            if new_data_available and reference_set:
                # print only 1/x of the data in the screen.
                if print_counter % 1 == 0:
                    for i in range(len(mtw_callbacks)):
                        Q_current = quarterly_data[i]
                        Q_current = Quaternion(Q_current[0], Q_current[1], Q_current[2], Q_current[3]) # Convert to pyquaternion Quaternion
                        Q_reference = reference_quat[i]
                        
                        # Calculate relative quaternion
                        Q1 = Quaternion(Q_current.w, Q_current.x, Q_current.y, Q_current.z)           # (w, x, y, z) order for pyquaternion
                        Q2 = Quaternion(Q_reference.w, Q_reference.x, Q_reference.y, Q_reference.z)

                        relative_quat = Q2.inverse * Q1

                        # Convert relative quaternion to Euler angles
                        yaw, pitch, roll = relative_quat.yaw_pitch_roll  # return (yaw, pitch, roll)
                        roll = math.degrees(roll)
                        pitch = math.degrees(pitch)
                        yaw = math.degrees(yaw)
                       
                        # Print the results 
                        if controller.viewer.is_running():
                            # 设置目标位置
                            controller.set_target_joint_positions(target_positions)
                            # 运行仿真
                            controller.update()
                            # 获取IMU欧拉角  转为目标关节位置
                            if i == 0:  
                                target_positions[0] = math.radians(yaw)  # 大臂Yaw角
                                target_positions[2] = math.radians(25 + pitch)  # 大臂Pitch角
                            if i == 1:
                                target_positions[4] = math.radians(yaw) # 小臂Yaw角
                            if i == 2:
                                target_positions[5] = math.radians(roll)  # 手腕Roll角


                print_counter += 1

        controller.close()


        print("Setting config mode...")
        if not wireless_master_device.gotoConfig():
            raise RuntimeError(f"Failed to goto config mode: {wireless_master_device}")

        print("Disabling radio...")
        if not wireless_master_device.disableRadio():
            raise RuntimeError(f"Failed to disable radio: {wireless_master_device}")

    except Exception as ex:
        print(ex)
        print("****ABORT****")
    except:
        print("An unknown fatal error has occurred. Aborting.")
        print("****ABORT****")

    print("Closing XsControl...")
    control.close()

    print("Deleting mtw callbacks...")

    print("Successful exit.")
    print("Press [ENTER] to continue.")
    input()


    
    
