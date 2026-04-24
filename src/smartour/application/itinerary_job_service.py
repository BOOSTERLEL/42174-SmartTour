"""Application service for itinerary generation jobs."""

from typing import Any

from smartour.application.planning_service import PlanningService
from smartour.core.errors import ExternalServiceError, PlanningInputError
from smartour.domain.conversation import ConversationState, MessageRole
from smartour.domain.itinerary_job import ItineraryJob
from smartour.integrations.google_maps.client import GoogleMapsClient


class ItineraryJobService:
    """
    Coordinates background itinerary generation jobs.
    """

    def __init__(
        self,
        conversation_repository: Any,
        job_repository: Any,
        planning_service: PlanningService,
        rate_limiter: Any | None = None,
    ) -> None:
        """
        Initialize the itinerary job service.

        Args:
            conversation_repository: Repository used to update conversation state.
            job_repository: Repository used to persist job state.
            planning_service: Planning service used to generate itineraries.
            rate_limiter: Optional rate limiter for job creation.
        """
        self.conversation_repository = conversation_repository
        self.job_repository = job_repository
        self.planning_service = planning_service
        self.rate_limiter = rate_limiter

    async def create_job(
        self, conversation_id: str, client_host: str | None = None
    ) -> ItineraryJob | None:
        """
        Create a queued itinerary generation job.

        Args:
            conversation_id: The source conversation ID.
            client_host: The request client host when available.

        Returns:
            The queued job, or None when the conversation is missing.

        Raises:
            PlanningInputError: Raised when required slots are incomplete.
        """
        conversation = await self.conversation_repository.get(conversation_id)
        if conversation is None:
            return None
        missing_slots = conversation.requirement.missing_required_slots()
        if missing_slots:
            raise PlanningInputError(
                "Cannot create an itinerary job until requirements are complete"
            )
        await self._check_rate_limits(conversation_id, client_host)
        conversation.state = ConversationState.PLANNING
        conversation.add_message(
            MessageRole.ASSISTANT,
            "I am generating your itinerary now.",
        )
        await self.conversation_repository.save(conversation)
        job = ItineraryJob(conversation_id=conversation_id)
        await self.job_repository.save(job)
        return job

    async def _check_rate_limits(
        self, conversation_id: str, client_host: str | None
    ) -> None:
        """
        Enforce job creation rate limits when configured.

        Args:
            conversation_id: The source conversation ID.
            client_host: The request client host when available.
        """
        if self.rate_limiter is None:
            return
        await self.rate_limiter.check_and_record(
            "conversation", conversation_id, "itinerary_job"
        )
        if client_host:
            await self.rate_limiter.check_and_record("ip", client_host, "itinerary_job")

    async def get_job(self, job_id: str) -> ItineraryJob | None:
        """
        Return an itinerary generation job by ID.

        Args:
            job_id: The itinerary job ID.

        Returns:
            The itinerary job when found.
        """
        return await self.job_repository.get(job_id)

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
        job = await self.job_repository.get(job_id)
        if job is None:
            return None
        job.mark_running()
        await self.job_repository.save(job)
        try:
            itinerary = await self.planning_service.generate_for_conversation(
                job.conversation_id, google_maps_client
            )
            if itinerary is None:
                await self._mark_failed(job, "Conversation not found")
            else:
                await self._mark_succeeded(job, itinerary.id)
        except (PlanningInputError, ExternalServiceError) as error:
            await self._mark_failed(job, str(error))
        except Exception as error:
            await self._mark_failed(job, "Unexpected itinerary generation failure")
            raise error
        return await self.job_repository.get(job.id)

    async def _mark_succeeded(self, job: ItineraryJob, itinerary_id: str) -> None:
        """
        Persist successful job and conversation state.

        Args:
            job: The job to update.
            itinerary_id: The generated itinerary ID.
        """
        job.mark_succeeded(itinerary_id)
        await self.job_repository.save(job)
        conversation = await self.conversation_repository.get(job.conversation_id)
        if conversation is None:
            return
        conversation.state = ConversationState.READY_FOR_REVIEW
        conversation.add_message(
            MessageRole.ASSISTANT,
            "Your itinerary is ready for review.",
        )
        await self.conversation_repository.save(conversation)

    async def _mark_failed(self, job: ItineraryJob, error_message: str) -> None:
        """
        Persist failed job and conversation state.

        Args:
            job: The job to update.
            error_message: The sanitized failure reason.
        """
        job.mark_failed(error_message)
        await self.job_repository.save(job)
        conversation = await self.conversation_repository.get(job.conversation_id)
        if conversation is None:
            return
        conversation.state = ConversationState.FAILED
        conversation.add_message(
            MessageRole.ASSISTANT,
            "I could not generate the itinerary. Please adjust the requirements.",
        )
        await self.conversation_repository.save(conversation)
