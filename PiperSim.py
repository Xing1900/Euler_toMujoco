import mujoco
import mujoco.viewer
import numpy as np
import math

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
    

if __name__ == "__main__":

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
    while (controller.viewer.is_running()):
        # 设置目标位置
        controller.set_target_joint_positions(target_positions)
        # 运行仿真
        controller.update()
        # 获取IMU欧拉角 添加获取IMU欧拉角的代码
        
        # 欧拉角转为目标关节位置
    

    controller.close() 
