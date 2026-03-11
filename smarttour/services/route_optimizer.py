"""
OR-Tools-backed route optimization.
"""

from __future__ import annotations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from smarttour.models import Attraction, Route, TransportSegment
from smarttour.services.maps_client import RouteGraph


class RouteOptimizer:
    """
    Solve single-day attraction routing problems.
    """

    def optimize(
        self,
        attractions: list[Attraction],
        graph: RouteGraph,
        travel_mode: str,
    ) -> Route:
        """
        Optimize the order of attractions for one day.

        Args:
            attractions: Candidate attractions.
            graph: Duration matrix and transport segments.
            travel_mode: Selected transport mode.

        Returns:
            The optimized route.
        """

        if not attractions:
            return Route()
        if len(attractions) == 1:
            return Route(
                attraction_order=[attractions[0].id],
                segments=[],
                total_distance_m=0,
                total_duration_s=0,
            )
        order = self._solve_order(graph.duration_matrix)
        attraction_order = [attractions[index].id for index in order]
        segments: list[TransportSegment] = []
        total_distance = 0
        total_duration = 0
        for current_index, next_index in zip(order, order[1:], strict=False):
            segment = graph.segment_lookup[(current_index, next_index)]
            segments.append(
                TransportSegment(
                    origin_id=segment.origin_id,
                    destination_id=segment.destination_id,
                    travel_mode=travel_mode,
                    distance_m=segment.distance_m,
                    duration_s=segment.duration_s,
                    polyline=segment.polyline,
                )
            )
            total_distance += segment.distance_m
            total_duration += segment.duration_s
        return Route(
            attraction_order=attraction_order,
            segments=segments,
            total_distance_m=total_distance,
            total_duration_s=total_duration,
        )

    def _solve_order(self, duration_matrix: list[list[int]]) -> list[int]:
        """
        Solve the travel order using OR-Tools.

        Args:
            duration_matrix: Pairwise duration matrix in seconds.

        Returns:
            Ordered attraction indices.
        """

        size = len(duration_matrix)
        manager = pywrapcp.RoutingIndexManager(size, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def duration_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return duration_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(duration_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            return list(range(size))
        order: list[int] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            order.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return order
