import numpy as np
import open3d as o3d
from collections import deque

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class FrameAccumulator:
    """
    Rolling buffer that automatically subscribes to a PointCloud2 topic and
    keeps the last num_frames in a deque. When the buffer is full, the oldest
    frame is dropped automatically as new ones arrive.

    Any detector can call get_accumulated_cloud() at any time to get the
    merged cloud of all buffered frames.
    """

    def __init__(self, topic, num_frames=10):
        self._num_frames = num_frames
        self._buffer = deque(maxlen=num_frames)
        self._sub = rospy.Subscriber(topic, PointCloud2, self._callback, queue_size=1)
        rospy.loginfo(f"[FrameAccumulator] Subscribed to '{topic}', keeping {num_frames} frames")

    def _callback(self, msg):
        """Convert incoming PointCloud2 to Open3D and push into the rolling buffer."""
        pts = np.array([
            [p[0], p[1], p[2]]
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ])

        pcd = o3d.geometry.PointCloud()
        if len(pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(pts)

        self._buffer.append(pcd)

    def get_accumulated_cloud(self):
        """
        Merge all buffered frames into a single point cloud.
        Returns None if the buffer is empty.
        """
        if not self._buffer:
            return None

        all_pts = np.vstack([
            np.asarray(frame.points)
            for frame in self._buffer
            if len(frame.points) > 0
        ])

        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(all_pts)
        return merged

    def clear(self):
        """Empty the buffer."""
        self._buffer.clear()

    @property
    def is_full(self):
        """True once num_frames have been collected."""
        return len(self._buffer) == self._num_frames

    @property
    def size(self):
        """Number of frames currently in the buffer."""
        return len(self._buffer)

    @property
    def num_frames(self):
        """Target number of frames to accumulate."""
        return self._num_frames
