#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import math

def oscillate_person():
    rospy.init_node('person_oscillator', anonymous=True)
    pub = rospy.Publisher('/person_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    
    start_time = rospy.Time.now()
    period = 6.0  # Complete cycle every 6 seconds
    amplitude = 0.5  # Max velocity (m/s)
    
    rospy.loginfo("Starting person oscillation (left-right)")
    
    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - start_time).to_sec()
        
        # Sinusoidal velocity: moves left then right
        velocity = amplitude * math.sin(2 * math.pi * elapsed / period)
        
        cmd = Twist()
        cmd.linear.y = velocity
        
        pub.publish(cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        oscillate_person()
    except rospy.ROSInterruptException:
        pass
