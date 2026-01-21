#!/usr/bin/env python

"""
File: ros_optitrack.py
Description: ROS node to bridge OptiTrack NatNet streaming data to ROS topics.
             Publishes PoseStamped (Quaternions) and Vector3Stamped (Euler RPY).
Authors: [Bárbara Bueno, Victor Pepe and Thiago Velasquez (IAS Lab / TU Darmstadt)]
Date: December 2025
"""

import rospy
import tf.transformations as tf_trans
import math
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32, Bool 
from natnet import DataDescriptions, DataFrame, NatNetClient

# Debug mode will log all raw rigid body info received from motive
DEBUG = True

class OptiTrackBridge:
    def __init__(self):
        rospy.init_node('optitrack_bridge')

       
        self.pole_pose_pub = rospy.Publisher('/optitrack/pole/pose', PoseStamped, queue_size=10)
        self.pole_euler_pub = rospy.Publisher('/optitrack/pole/euler', Vector3Stamped, queue_size=10)
        self.pole_error_pub = rospy.Publisher('/optitrack/pole/marker_error', Float32, queue_size=10)
        self.pole_status_pub = rospy.Publisher('/optitrack/pole/tracking_valid', Bool, queue_size=10)
        
        self.base_pose_pub = rospy.Publisher('/optitrack/base/pose', PoseStamped, queue_size=10)
        self.base_euler_pub = rospy.Publisher('/optitrack/base/euler', Vector3Stamped, queue_size=10)
        self.base_error_pub = rospy.Publisher('/optitrack/base/marker_error', Float32, queue_size=10)
        self.base_status_pub = rospy.Publisher('/optitrack/base/tracking_valid', Bool, queue_size=10)


        # Look for Streming ID on Motive and change below
        self.POLE_ID = 9 
        self.BASE_ID = 10

        self.client = NatNetClient(
            server_ip_address="10.90.90.2", 
            local_ip_address="10.90.90.1", 
            use_multicast=True
        )
        self.client.on_data_frame_received_event.handlers.append(self.receive_new_frame)

    def receive_new_frame(self, data_frame: DataFrame):
        now = rospy.Time.now()

        rospy.loginfo(f"\nData frame info: {data_frame}\n\n")

        for rb in data_frame.rigid_bodies:

            if DEBUG:
                rospy.loginfo(f"\nRigid body info: {rb}\n\n")
            if rb.tracking_valid:


                # --- Get Position and Quaternion ---
                pose_msg = PoseStamped()
                pose_msg.header.stamp = now
                pose_msg.header.frame_id = "optitrack_world"
                
                pose_msg.pose.position.x = rb.pos[0]
                pose_msg.pose.position.y = rb.pos[1]
                pose_msg.pose.position.z = rb.pos[2]
                
                qx, qy, qz, qw = rb.rot[0], rb.rot[1], rb.rot[2], rb.rot[3]
                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw

                # -- Convert to roll pitch yaw in degrees --
                euler = tf_trans.euler_from_quaternion([qx, qy, qz, qw])
                
                euler_msg = Vector3Stamped()
                euler_msg.header.stamp = now
                euler_msg.header.frame_id = "optitrack_world"
                euler_msg.vector.x = math.degrees(euler[0]) 
                euler_msg.vector.y = math.degrees(euler[1]) # 
                euler_msg.vector.z = math.degrees(euler[2]) 

                # --- Publish  ---
                if rb.id_num == self.POLE_ID:
                    self.pole_pose_pub.publish(pose_msg)
                    self.pole_euler_pub.publish(euler_msg)
                    self.pole_error_pub.publish(rb.marker_error) 
                    self.pole_status_pub.publish(rb.tracking_valid) 
                    rospy.loginfo(f"Pole published (ID:{rb.id_num}) - Error: {rb.marker_error:.4f}")
                
                elif rb.id_num == self.BASE_ID:
                    self.base_pose_pub.publish(pose_msg)
                    self.base_euler_pub.publish(euler_msg)
                    self.base_error_pub.publish(rb.marker_error) 
                    self.base_status_pub.publish(rb.tracking_valid)
                    rospy.loginfo(f"Base published (ID:{rb.id_num}) - Error: {rb.marker_error:.4f}")

    def run(self):
        with self.client:
            self.client.request_modeldef()
            rospy.loginfo("OptiTrack Bridge Active. Streaming Pose and Euler data from Pole and Base...")
            self.client.run_async()
            rospy.spin()

if __name__ == "__main__":
    bridge = OptiTrackBridge()
    try:
        bridge.run()
    except rospy.ROSInterruptException:
        pass