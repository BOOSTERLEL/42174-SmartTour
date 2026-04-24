"""Itinerary job repository implementations."""

from smartour.domain.itinerary_job import ItineraryJob


class InMemoryItineraryJobRepository:
    """
    Process-local in-memory itinerary job repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.jobs: dict[str, ItineraryJob] = {}

    def save(self, job: ItineraryJob) -> None:
        """
        Save an itinerary job.

        Args:
            job: The itinerary job to save.
        """
        self.jobs[job.id] = job.model_copy(deep=True)

    def get(self, job_id: str) -> ItineraryJob | None:
        """
        Return an itinerary job by ID.

        Args:
            job_id: The itinerary job ID.

        Returns:
            The itinerary job when found.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return job.model_copy(deep=True)
