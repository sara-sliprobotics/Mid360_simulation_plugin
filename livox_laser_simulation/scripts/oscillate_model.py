#!/usr/bin/env python3
"""Move the sensor model back and forth along X from -1 to 1, publishing odom TF."""

import rospy
import math
import tf2_ros
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


def main():
    rospy.init_node('oscillate_model')

    model_name = rospy.get_param('~model_name', 'iris_demo')
    x_min = rospy.get_param('~x_min', -1.0)
    x_max = rospy.get_param('~x_max', 1.0)
    period = rospy.get_param('~period', 6.0)  # seconds for one full cycle
    rate_hz = rospy.get_param('~rate', 50.0)
    # Sensor height offset: lidar_link Z (0.2) + sensor Z within link (0.05) = 0.25
    # The plugin outputs sensor-relative coords tagged as base_link,
    # so the TF must compensate for this offset.
    sensor_z_offset = rospy.get_param('~sensor_z_offset', 0.25)

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    tf_broadcaster = tf2_ros.TransformBroadcaster()
    odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)

    rate = rospy.Rate(rate_hz)
    start_time = rospy.Time.now()

    amplitude = (x_max - x_min) / 2.0
    offset = (x_max + x_min) / 2.0

    rospy.loginfo(f"Oscillating '{model_name}' on X: [{x_min}, {x_max}], period={period}s")

    while not rospy.is_shutdown():
        t = (rospy.Time.now() - start_time).to_sec()
        x = offset + amplitude * math.sin(2.0 * math.pi * t / period)
        vx = amplitude * (2.0 * math.pi / period) * math.cos(2.0 * math.pi * t / period)

        # Move model in Gazebo
        state = ModelState()
        state.model_name = model_name
        state.pose.position.x = x
        state.pose.position.y = 0.0
        state.pose.position.z = 0.0
        state.pose.orientation.w = 1.0
        state.reference_frame = 'world'
        try:
            set_state(state)
        except rospy.ServiceException:
            pass

        now = rospy.Time.now()

        # Publish odom -> base_link TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = sensor_z_offset
        tf_msg.transform.rotation.w = 1.0
        tf_broadcaster.sendTransform(tf_msg)

        # Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.twist.twist.linear.x = vx
        odom_pub.publish(odom_msg)

        rate.sleep()


if __name__ == '__main__':
    main()
