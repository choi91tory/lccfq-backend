"""
gRPC service implementation for QPU Executor.
Wraps the executor instance to handle incoming gRPC requests.
"""

import grpc

from lccfq_backend.api.protobufs_compiled.qpu_service_pb2_grpc import QPUExecutorServicer
from lccfq_backend.api.protobufs_compiled.qpu_service_pb2 import (
    ExecutorRequest,
    ExecutorResponse,
    SubmitCircuitTaskRequest,
    SubmitCircuitTaskResponse,
    SubmitTestTaskRequest,
    SubmitTestTaskResponse,
    GetResultRequest,
    GetResultResponse,
    CircuitResultPayload,
    TestResultPayload,
    ControlAckPayload,
    GetTaskStatusRequest,
    GetTaskStatusResponse,
    TASK_STATUS_QUEUED,
    TASK_STATUS_RUNNING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_CANCELLED,
    CancelTaskRequest,
    CancelTaskResponse,
    GetTargetRequest,
    GetTargetResponse,
)
from lccfq_backend.backend.executor import QPUExecutor
from lccfq_backend.backend.result_store import ResultStore
from lccfq_backend.model.tasks import CircuitTask, TestTask, Gate, TaskStatus
from lccfq_backend.model.results import CircuitResult, TestResult, ControlAck
from lccfq_backend.backend.error import UnknownQPUTaskType
from lccfq_backend.utils.log import setup_logger

logger = setup_logger("ExecutorService")

_TASK_STATUS_TO_PROTO = {
    TaskStatus.QUEUED: TASK_STATUS_QUEUED,
    TaskStatus.RUNNING: TASK_STATUS_RUNNING,
    TaskStatus.COMPLETED: TASK_STATUS_COMPLETED,
    TaskStatus.FAILED: TASK_STATUS_FAILED,
    TaskStatus.CANCELLED: TASK_STATUS_CANCELLED,
}


class ExecutorService(QPUExecutorServicer):
    """gRPC service that wraps the QPU Executor."""

    def __init__(self, executor: QPUExecutor, result_store: ResultStore | None = None):
        """
        Initialize the ExecutorService with an executor instance.

        Args:
            executor: QPUExecutor instance to use for handling requests
            result_store: ResultStore for retrieving persisted task results
        """
        self.executor = executor
        self.result_store = result_store
        logger.info("ExecutorService initialized with executor instance")

    def Ping(self, request: ExecutorRequest, context: grpc.ServicerContext) -> ExecutorResponse:
        """
        Handle the Ping request to check service health.

        Args:
            request: ExecutorRequest message
            context: gRPC service context

        Returns:
            ExecutorResponse with success status and message
        """
        logger.info(f"Received Ping request from client {context.peer()}")
        response = ExecutorResponse(
            message="Pong from LCCFQ Backend",
            success=True
        )
        return response

    def SubmitCircuitTask(
        self,
        request: SubmitCircuitTaskRequest,
        context: grpc.ServicerContext
    ) -> SubmitCircuitTaskResponse:
        """
        Handle circuit task submission from gRPC client.

        Args:
            request: SubmitCircuitTaskRequest with gates and shots
            context: gRPC service context (contains client certificate metadata)

        Returns:
            SubmitCircuitTaskResponse with task_id and status
        """
        try:
            # Extract user ID from client certificate
            user_id = self._extract_user_from_context(context)
            logger.info(f"Received circuit task submission from user: {user_id}")

            # Convert proto Gates to Python Gate objects
            gates = [
                Gate(
                    symbol=g.symbol,
                    target_qubits=list(g.target_qubits),
                    control_qubits=list(g.control_qubits),
                    params=list(g.params),
                )
                for g in request.gates
            ]

            # Validate shots is positive
            if request.shots <= 0:
                logger.error(f"Invalid shots value: {request.shots}")
                return SubmitCircuitTaskResponse(
                    task_id="",
                    success=False,
                    message=f"Invalid shots value: {request.shots}. Must be positive."
                )

            # Create CircuitTask (auto-generates task_id via UUID)
            circuit_task = CircuitTask(
                gates=gates,
                shots=request.shots,
            )

            # Enqueue task directly in queue (async - don't execute immediately)
            # Using default priority=0 and no context_id for simplicity
            # NOTE: We call queue.enqueue() directly instead of executor.execute()
            # because execute() would immediately dispatch if QPU is idle,
            # but we want async behavior - just enqueue and let main loop handle execution
            self.executor.queue.enqueue(
                task=circuit_task,
                user=user_id,
                context_id=None,
                priority=0
            )

            # Mark the task as queued after it is successfully added to the queue
            self.executor.set_task_status(
                circuit_task.task_id,
                TaskStatus.QUEUED
            )

            logger.info(f"Circuit task enqueued: {circuit_task.task_id} for user {user_id}")

            return SubmitCircuitTaskResponse(
                task_id=circuit_task.task_id,
                success=True,
                message=f"Task {circuit_task.task_id} enqueued successfully"
            )

        except UnknownQPUTaskType as e:
            logger.error(f"Invalid task type: {e}")
            return SubmitCircuitTaskResponse(
                task_id="",
                success=False,
                message=f"Task submission failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error submitting circuit task: {e}", exc_info=True)
            return SubmitCircuitTaskResponse(
                task_id="",
                success=False,
                message=f"Internal error: {str(e)}"
            )

    def SubmitTestTask(
        self,
        request: SubmitTestTaskRequest,
        context: grpc.ServicerContext
    ) -> SubmitTestTaskResponse:
        """
        Handle test task submission from gRPC client.

        Args:
            request: SubmitTestTaskRequest with symbol, params, and shots
            context: gRPC service context (contains client certificate metadata)

        Returns:
            SubmitTestTaskResponse with task_id and status
        """
        try:
            # Extract user ID from client certificate
            user_id = self._extract_user_from_context(context)
            logger.info(f"Received test task submission from user: {user_id}")

            # Validate shots is positive
            if request.shots <= 0:
                logger.error(f"Invalid shots value: {request.shots}")
                return SubmitTestTaskResponse(
                    task_id="",
                    success=False,
                    message=f"Invalid shots value: {request.shots}. Must be positive."
                )

            # Validate symbol is not empty
            if not request.symbol or request.symbol.strip() == "":
                logger.error("Empty test symbol provided")
                return SubmitTestTaskResponse(
                    task_id="",
                    success=False,
                    message="Test symbol cannot be empty."
                )

            # Create TestTask (auto-generates task_id via UUID)
            test_task = TestTask(
                symbol=request.symbol,
                params=list(request.params),
                shots=request.shots,
            )

            # Enqueue task directly in queue (async - don't execute immediately)
            # Using default priority=0 and no context_id for simplicity
            # NOTE: We call queue.enqueue() directly instead of executor.execute()
            # because execute() would immediately dispatch if QPU is idle,
            # but we want async behavior - just enqueue and let main loop handle execution
            self.executor.queue.enqueue(
                task=test_task,
                user=user_id,
                context_id=None,
                priority=0
            )

            # Mark the task as queued after it is successfully added to the queue
            self.executor.set_task_status(
                test_task.task_id,
                TaskStatus.QUEUED
            )

            logger.info(f"Test task enqueued: {test_task.task_id} for user {user_id}")

            return SubmitTestTaskResponse(
                task_id=test_task.task_id,
                success=True,
                message=f"Task {test_task.task_id} enqueued successfully"
            )

        except UnknownQPUTaskType as e:
            logger.error(f"Invalid task type: {e}")
            return SubmitTestTaskResponse(
                task_id="",
                success=False,
                message=f"Task submission failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error submitting test task: {e}", exc_info=True)
            return SubmitTestTaskResponse(
                task_id="",
                success=False,
                message=f"Internal error: {str(e)}"
            )

    def GetResult(
        self,
        request: GetResultRequest,
        context: grpc.ServicerContext,
    ) -> GetResultResponse:
        """
        Retrieve a previously executed task result by task ID.

        Args:
            request: GetResultRequest with task_id
            context: gRPC service context

        Returns:
            GetResultResponse — found=True with the result payload, or found=False
        """
        task_id = request.task_id.strip()
        logger.info(f"GetResult request for task_id={task_id}")

        if not self.result_store:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Result store not configured on this server")
            return GetResultResponse(found=False, error_message="Result store not available")

        result = self.result_store.load(task_id)

        if result is None:
            return GetResultResponse(found=False, error_message=f"No result for task {task_id}")

        if isinstance(result, CircuitResult):
            return GetResultResponse(
                found=True,
                circuit_result=CircuitResultPayload(
                    task_id=result.task_id,
                    distribution=result.distribution,
                ),
            )
        if isinstance(result, TestResult):
            return GetResultResponse(
                found=True,
                test_result=TestResultPayload(
                    task_id=result.task_id,
                    parameters=result.parameters,
                ),
            )
        if isinstance(result, ControlAck):
            return GetResultResponse(
                found=True,
                control_ack=ControlAckPayload(
                    task_id=result.task_id,
                    status=result.status,
                    message=result.message or "",
                ),
            )

        logger.error(f"Unexpected result type for task {task_id}: {type(result)}")
        return GetResultResponse(found=False, error_message="Unexpected result type")

    def GetTaskStatus(
        self,
        request: GetTaskStatusRequest,
        context: grpc.ServicerContext,
    ) -> GetTaskStatusResponse:
        """
        Retrieve the current lifecycle status of a task.

        Args:
            request: GetTaskStatusRequest with task_id
            context: gRPC service context

        Returns:
            GetTaskStatusResponse with the current status, or found=False
        """
        task_id = request.task_id.strip()
        logger.info(f"GetTaskStatus request for task_id={task_id}")

        if not task_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Task ID cannot be empty")
            return GetTaskStatusResponse(
                found=False,
                error_message="Task ID cannot be empty",
            )

        status = self.executor.get_task_status(task_id)

        if status is None:
            return GetTaskStatusResponse(
                found=False,
                error_message=f"No status for task {task_id}",
            )

        return GetTaskStatusResponse(
            found=True,
            status=_TASK_STATUS_TO_PROTO[status],
        )

    def CancelTask(
        self,
        request: CancelTaskRequest,
        context: grpc.ServicerContext,
    ) -> CancelTaskResponse:
        """
        Cancel a queued task by task ID.
        """
        task_id = request.task_id.strip()
        logger.info(f"CancelTask request for task_id={task_id}")

        if not task_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Task ID cannot be empty")
            return CancelTaskResponse(
                found=False,
                cancelled=False,
                error_message="Task ID cannot be empty",
            )

        status = self.executor.get_task_status(task_id)

        if status is None:
            return CancelTaskResponse(
                found=False,
                cancelled=False,
                error_message=f"No status for task {task_id}",
            )

        if status != TaskStatus.QUEUED:
            return CancelTaskResponse(
                found=True,
                cancelled=False,
                error_message=(
                    f"Task {task_id} cannot be cancelled "
                    f"from status {status.value}"
                ),
            )

        cancelled = self.executor.cancel_task(task_id)

        if not cancelled:
            return CancelTaskResponse(
                found=True,
                cancelled=False,
                error_message=f"Task {task_id} could not be cancelled",
            )

        return CancelTaskResponse(
            found=True,
            cancelled=True,
        )

    def GetTarget(
        self,
        request: GetTargetRequest,
        context: grpc.ServicerContext,
    ) -> GetTargetResponse:
        """
        Retrieve the current QPU target information.
        """
        logger.info("GetTarget request received")

        try:
            observables = self.executor.hwman.get_observables()

            return GetTargetResponse(
                success=True,
                target_json=observables.model_dump_json(),
            )

        except Exception as e:
            logger.error(f"Error retrieving QPU target: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to retrieve QPU target")

            return GetTargetResponse(
                success=False,
                error_message=str(e),
            )

    def _extract_user_from_context(self, context: grpc.ServicerContext) -> str:
        """
        Extract user ID from the client's mTLS certificate.

        The client certificate's Common Name (CN) contains the user ID.

        Args:
            context: gRPC service context

        Returns:
            User ID string from certificate CN

        Raises:
            ValueError: If certificate or CN is missing
        """
        # Get authentication context
        auth_context = context.auth_context()

        # Extract the peer identity (certificate subject)
        # auth_context keys are strings, values are lists of bytes
        # e.g., {'x509_common_name': [b'user_id']}
        user_id = None
        for key, values in auth_context.items():
            if key == 'x509_common_name':
                # values is a list, get the first element and decode
                if isinstance(values, list) and len(values) > 0:
                    value = values[0]
                    if isinstance(value, bytes):
                        user_id = value.decode('utf-8')
                    else:
                        user_id = str(value)
                    break

        if not user_id:
            logger.error("No client certificate CN found in auth context")
            raise ValueError("Client certificate missing or invalid")

        logger.debug(f"Extracted user_id from certificate: {user_id}")

        return user_id