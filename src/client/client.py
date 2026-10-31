import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)


class AgentClientError(Exception):
    pass


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            # Don't yield empty string tokens as they cause generator issues
                            if parsed != "":
                                yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str, agent: str | None = None) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
            agent (str, optional): The agent whose graph should interpret the thread.
        """
        agent = agent or self.agent
        request = ChatHistoryInput(thread_id=thread_id)
        url = f"{self.base_url}/{agent}/history" if agent else f"{self.base_url}/history"
        try:
            response = httpx.post(
                url,
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    def _user_threads_request(
        self, user_id: str, agent: str | None, limit: int
    ) -> tuple[str, dict[str, Any]]:
        agent_id = agent or self.agent
        url = f"{self.base_url}/{agent_id}/threads" if agent_id else f"{self.base_url}/threads"
        return url, UserThreadsInput(user_id=user_id, limit=limit).model_dump()

    def get_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """
        List a user's conversation threads.

        Args:
            user_id (str): User ID to list threads for.
            agent (str, optional): The agent whose threads should be listed.
            limit (int, optional): Maximum number of threads to return.
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        try:
            response = httpx.get(
                url,
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())

    async def aget_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """
        List a user's conversation threads asynchronously.

        Args:
            user_id (str): User ID to list threads for.
            agent (str, optional): The agent whose threads should be listed.
            limit (int, optional): Maximum number of threads to return.
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())


    def upload_construction_file(
        self,
        file_name: str,
        file_bytes: bytes,
        run_ocr: bool = True,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload a construction document or inspection photograph to the backend."""
        url = f"{self.base_url}/construction/upload"
        files = {"file": (file_name, file_bytes)}
        data: dict[str, Any] = {"run_ocr": "true" if run_ocr else "false"}
        if case_id:
            data["case_id"] = case_id
        try:
            response = httpx.post(
                url,
                files=files,
                data=data,
                headers=self._headers,
                timeout=self.timeout or 60.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error uploading file to construction service: {e}")

    def update_case_data(self, case_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update persistent case metadata, coordinates, or corrections."""
        url = f"{self.base_url}/construction/cases/{case_id}/update"
        try:
            response = httpx.post(
                url,
                json=data,
                headers=self._headers,
                timeout=self.timeout or 30.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error updating case data: {e}")

    def get_case_state(self, case_id: str) -> dict[str, Any]:
        """Fetch full case record and verification status."""
        url = f"{self.base_url}/construction/cases/{case_id}"
        try:
            response = httpx.get(
                url,
                headers=self._headers,
                timeout=self.timeout or 30.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error fetching case state: {e}")

    def run_case_verification(self, case_id: str) -> dict[str, Any]:
        """Trigger LangGraph verification pipeline for a validated case."""
        url = f"{self.base_url}/construction/cases/{case_id}/verify"
        try:
            response = httpx.post(
                url,
                headers=self._headers,
                timeout=self.timeout or 120.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error executing case verification: {e}")

    def record_officer_decision(
        self, case_id: str, decision: str, officer_name_or_id: str, notes: str | None = None
    ) -> dict[str, Any]:
        """Record an official human officer decision (APPROVE, REJECT, REQUEST_INSPECTION, HOLD)."""
        url = f"{self.base_url}/construction/decision"
        payload = {
            "case_id": case_id,
            "decision": decision,
            "officer_name_or_id": officer_name_or_id,
            "notes": notes,
        }
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error recording officer decision: {e}")

    def get_case_decision(self, case_id: str) -> dict[str, Any]:
        """Fetch audit record and decision for a construction verification case."""
        url = f"{self.base_url}/construction/case/{case_id}"
        try:
            response = httpx.get(
                url,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error retrieving case decision: {e}")

    def run_construction_discovery(
        self,
        latitude: float = 28.17,
        longitude: float = 77.61,
        radius_km: float = 5.0,
        before_date_start: str = "2021-01-01",
        before_date_end: str = "2021-03-31",
        after_date_start: str = "2025-01-01",
        after_date_end: str = "2025-03-31",
        min_change_area_m2: float = 5000.0,
        max_cloud_cover: float = 20.0,
        area_name: str = "Noida International Airport, Jewar",
    ) -> dict[str, Any]:
        """Run independent construction discovery across multi-temporal Sentinel-2 imagery."""
        url = f"{self.base_url}/construction/discover"
        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "before_date_start": before_date_start,
            "before_date_end": before_date_end,
            "after_date_start": after_date_start,
            "after_date_end": after_date_end,
            "min_change_area_m2": min_change_area_m2,
            "max_cloud_cover": max_cloud_cover,
            "area_name": area_name,
        }
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=180.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error executing construction discovery: {e}")

    def get_discovery_report(self, discovery_id: str) -> dict[str, Any]:
        """Retrieve previous construction discovery report."""
        url = f"{self.base_url}/construction/discover/{discovery_id}"
        try:
            response = httpx.get(url, headers=self._headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error fetching discovery report: {e}")

    def get_discovery_geojson(self, discovery_id: str) -> dict[str, Any]:
        """Retrieve GeoJSON candidate polygons for independent Google Earth Engine verification."""
        url = f"{self.base_url}/construction/discover/{discovery_id}/geojson"
        try:
            response = httpx.get(url, headers=self._headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error fetching discovery GeoJSON: {e}")
