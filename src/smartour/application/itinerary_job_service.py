"""Application service for itinerary generation jobs."""

from smartour.application.planning_service import PlanningService
from smartour.core.errors import ExternalServiceError, PlanningInputError
from smartour.domain.conversation import ConversationState, MessageRole
from smartour.domain.itinerary_job import ItineraryJob
from smartour.infrastructure.repositories.conversations import (
    InMemoryConversationRepository,
)
from smartour.infrastructure.repositories.itinerary_jobs import (
    InMemoryItineraryJobRepository,
)
from smartour.integrations.google_maps.client import GoogleMapsClient


class ItineraryJobService:
    """
    Coordinates background itinerary generation jobs.
    """

    def __init__(
        self,
        conversation_repository: InMemoryConversationRepository,
        job_repository: InMemoryItineraryJobRepository,
        planning_service: PlanningService,
    ) -> None:
        """
        Initialize the itinerary job service.

        Args:
            conversation_repository: Repository used to update conversation state.
            job_repository: Repository used to persist job state.
            planning_service: Planning service used to generate itineraries.
        """
        self.conversation_repository = conversation_repository
        self.job_repository = job_repository
        self.planning_service = planning_service

    def create_job(self, conversation_id: str) -> ItineraryJob | None:
        """
        Create a queued itinerary generation job.

        Args:
            conversation_id: The source conversation ID.

        Returns:
            The queued job, or None when the conversation is missing.

        Raises:
            PlanningInputError: Raised when required slots are incomplete.
        """
        conversation = self.conversation_repository.get(conversation_id)
        if conversation is None:
            return None
        missing_slots = conversation.requirement.missing_required_slots()
        if missing_slots:
            raise PlanningInputError(
                "Cannot create an itinerary job until requirements are complete"
            )
        conversation.state = ConversationState.PLANNING
        conversation.add_message(
            MessageRole.ASSISTANT,
            "I am generating your itinerary now.",
        )
        self.conversation_repository.save(conversation)
        job = ItineraryJob(conversation_id=conversation_id)
        self.job_repository.save(job)
        return job

    def get_job(self, job_id: str) -> ItineraryJob | None:
        """
        Return an itinerary generation job by ID.

        Args:
            job_id: The itinerary job ID.

        Returns:
            The itinerary job when found.
        """
        return self.job_repository.get(job_id)

    async def run_job(
        self, job_id: str, google_maps_client: GoogleMapsClient
    ) -> ItineraryJob | None:
        """
        Run a queued itinerary generation job.

        Args:
            job_id: The itinerary job ID.
            google_maps_client: The Google Maps client group.

        Returns:
            The completed job when found.
        """
        job = self.job_repository.get(job_id)
        if job is None:
            return None
        job.mark_running()
        self.job_repository.save(job)
        try:
            itinerary = await self.planning_service.generate_for_conversation(
                job.conversation_id, google_maps_client
            )
            if itinerary is None:
                self._mark_failed(job, "Conversation not found")
            else:
                self._mark_succeeded(job, itinerary.id)
        except (PlanningInputError, ExternalServiceError) as error:
            self._mark_failed(job, str(error))
        except Exception as error:
            self._mark_failed(job, "Unexpected itinerary generation failure")
            raise error
        return self.job_repository.get(job.id)

    def _mark_succeeded(self, job: ItineraryJob, itinerary_id: str) -> None:
        """
        Persist successful job and conversation state.

        Args:
            job: The job to update.
            itinerary_id: The generated itinerary ID.
        """
        job.mark_succeeded(itinerary_id)
        self.job_repository.save(job)
        conversation = self.conversation_repository.get(job.conversation_id)
        if conversation is None:
            return
        conversation.state = ConversationState.READY_FOR_REVIEW
        conversation.add_message(
            MessageRole.ASSISTANT,
            "Your itinerary is ready for review.",
        )
        self.conversation_repository.save(conversation)

    def _mark_failed(self, job: ItineraryJob, error_message: str) -> None:
        """
        Persist failed job and conversation state.

        Args:
            job: The job to update.
            error_message: The sanitized failure reason.
        """
        job.mark_failed(error_message)
        self.job_repository.save(job)
        conversation = self.conversation_repository.get(job.conversation_id)
        if conversation is None:
            return
        conversation.state = ConversationState.FAILED
        conversation.add_message(
            MessageRole.ASSISTANT,
            "I could not generate the itinerary. Please adjust the requirements.",
        )
        self.conversation_repository.save(conversation)
