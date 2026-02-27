import threading
import numpy as np
import open3d as o3d
from collections import deque

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class FrameAccumulator:
    """
    Rolling buffer that automatically subscribes to a PointCloud2 topic and
    keeps the last num_frames in a deque. When the buffer is full, the oldest
    frame is dropped automatically as new ones arrive.

    Each frame is transformed into a fixed frame (default: 'odom') using TF
    so that accumulated clouds are spatially consistent even when the sensor
    is moving.

    Any detector can call get_accumulated_cloud() at any time to get the
    merged cloud of all buffered frames in the fixed frame.
    """

    def __init__(self, topic, num_frames=10, fixed_frame='odom'):
        self._num_frames = num_frames
        self._fixed_frame = fixed_frame
        self._buffer = deque(maxlen=num_frames)
        self._lock = threading.Lock()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._sub = rospy.Subscriber(topic, PointCloud2, self._callback, queue_size=1)
        rospy.loginfo(f"[FrameAccumulator] Subscribed to '{topic}', keeping {num_frames} frames, fixed_frame='{fixed_frame}'")

    def _callback(self, msg):
        """Convert incoming PointCloud2 to Open3D, transform to fixed frame, and push into the rolling buffer."""
        pts = np.array([
            [p[0], p[1], p[2]]
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ])

        if len(pts) == 0:
            return

        # Transform points into the fixed frame using TF
        source_frame = msg.header.frame_id
        try:
            transform = self._tf_buffer.lookup_transform(
                self._fixed_frame, source_frame, msg.header.stamp, rospy.Duration(0.1)
            )

            pts = self._apply_transform(pts, transform)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"[FrameAccumulator] TF lookup failed ({source_frame} -> {self._fixed_frame}): {e}. Dropping frame.")
            return  # Don't add untransformed points — they'd misalign with other frames

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        with self._lock:
            self._buffer.append(pcd)

    @staticmethod
    def _apply_transform(pts, transform):
        """Apply a geometry_msgs/TransformStamped to an Nx3 numpy array."""
        t = transform.transform.translation
        q = transform.transform.rotation

        # Quaternion to rotation matrix
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)]
        ])
        translation = np.array([t.x, t.y, t.z])

        return (pts @ rot.T) + translation

    def get_accumulated_cloud(self):
        """
        Merge all buffered frames into a single point cloud.
        All frames are already in the fixed frame.
        Returns None if the buffer is empty.
        """
        with self._lock:
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
        with self._lock:
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
