#!/usr/bin/env python

import rospy
import tf.transformations as tf_trans
import math
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32, Bool, Float32MultiArray 
from natnet import DataFrame, NatNetClient

DEBUG = True

class OptiTrackBridge:
    def __init__(self):
        rospy.init_node('optitrack_bridge')

        # Publishers for Pole
        self.pole_pose_pub = rospy.Publisher('/optitrack/pole/pose', PoseStamped, queue_size=10)
        self.pole_euler_pub = rospy.Publisher('/optitrack/pole/euler', Vector3Stamped, queue_size=10)
        self.pole_status_pub = rospy.Publisher('/optitrack/pole/tracking_valid', Bool, queue_size=10)
        self.pole_markers_pub = rospy.Publisher('/optitrack/pole/markers', Float32MultiArray, queue_size=10)
        
        # Publishers for Base
        self.base_pose_pub = rospy.Publisher('/optitrack/base/pose', PoseStamped, queue_size=10)
        self.base_euler_pub = rospy.Publisher('/optitrack/base/euler', Vector3Stamped, queue_size=10)
        self.base_status_pub = rospy.Publisher('/optitrack/base/tracking_valid', Bool, queue_size=10)
        self.base_markers_pub = rospy.Publisher('/optitrack/base/markers', Float32MultiArray, queue_size=10)

        # Rigid Body IDs for Pose
        self.POLE_ID = 1 
        self.BASE_ID = 2

        # Model Names for Marker Sets
        self.POLE_MODEL_NAME = 'BraziliansPole'
        self.BASE_MODEL_NAME = 'BraziliansHandle'


        self.client = NatNetClient(
            server_ip_address="10.90.90.2", 
            local_ip_address="10.90.90.1", 
            use_multicast=True
        )
        self.client.on_data_frame_received_event.handlers.append(self.receive_new_frame)


    def receive_new_frame(self, data_frame: DataFrame):
        now = rospy.Time.now()

        # 1. Process Rigid Bodies for Orientation and Position
        print(" Data fram info: ", data_frame)
        for rb in data_frame.rigid_bodies:
            if rb.tracking_valid:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = now
                pose_msg.header.frame_id = "optitrack_world"
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = rb.pos
                
                qx, qy, qz, qw = rb.rot
                pose_msg.pose.orientation.x, pose_msg.pose.orientation.y = qx, qy
                pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = qz, qw

                euler = tf_trans.euler_from_quaternion([qx, qy, qz, qw])
                euler_msg = Vector3Stamped()
                euler_msg.header.stamp = now
                euler_msg.vector.x, euler_msg.vector.y, euler_msg.vector.z = [math.degrees(a) for a in euler]

                if rb.id_num == self.POLE_ID:
                    self.pole_pose_pub.publish(pose_msg)
                    self.pole_euler_pub.publish(euler_msg)
                    self.pole_status_pub.publish(True)
                elif rb.id_num == self.BASE_ID:
                    self.base_pose_pub.publish(pose_msg)
                    self.base_euler_pub.publish(euler_msg)
                    self.base_status_pub.publish(True)

        # 2. Process Marker Sets for individual point coordinates
        for ms in data_frame.marker_sets:
            marker_data = []
            for pos in ms.marker_pos_list:
                marker_data.extend([pos[0], pos[1], pos[2]]) 

            marker_msg = Float32MultiArray()
            marker_msg.data = marker_data

            if ms.model_name == self.POLE_MODEL_NAME:
                self.pole_markers_pub.publish(marker_msg)
            elif ms.model_name == self.BASE_MODEL_NAME:
                self.base_markers_pub.publish(marker_msg)

    def run(self):
        with self.client:
            self.client.request_modeldef()
            rospy.loginfo("Bridge active. Streaming MarkerSets and RigidBodies...")
            self.client.run_async()
            rospy.spin()

if __name__ == "__main__":
    bridge = OptiTrackBridge()
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass