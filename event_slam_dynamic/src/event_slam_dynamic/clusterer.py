"""Dynamic track clustering placeholder.

TODO: upgrade to joint (x, y, u, v, t) clustering and contour completion.
"""

from typing import List

from .adapters import DynamicCluster, TrackState


class DynamicClusterer:
    """Minimal clusterer: each dynamic track is one cluster."""

    def cluster(self, dynamic_tracks: List[TrackState]) -> List[DynamicCluster]:
        clusters: List[DynamicCluster] = []
        for i, t in enumerate(dynamic_tracks):
            clusters.append(
                DynamicCluster(
                    cluster_id=i,
                    member_track_ids=[t.track_id],
                    center_x=t.last_position[0],
                    center_y=t.last_position[1],
                    mean_u=t.velocity_u,
                    mean_v=t.velocity_v,
                )
            )
        return clusters
