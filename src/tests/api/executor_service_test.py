from unittest.mock import Mock

import grpc

from lccfq_backend.api.protobufs_compiled.qpu_service_pb2 import (
    GetTaskStatusRequest,
    TASK_STATUS_RUNNING,
    TASK_STATUS_UNSPECIFIED,
    CancelTaskRequest,
)
from lccfq_backend.api.services.executor_service import ExecutorService
from lccfq_backend.backend.executor import QPUExecutor
from lccfq_backend.backend.users import UserManager
from lccfq_backend.config import BackendSettings
from lccfq_backend.model.tasks import TaskStatus, TestTask, TaskType


_config = BackendSettings(hwman_mock_mode=True)


def make_service() -> ExecutorService:
    executor = QPUExecutor(
        config=_config,
        users=UserManager(),
    )
    return ExecutorService(executor=executor)


def test_get_task_status_returns_current_status():
    service = make_service()
    service.executor.set_task_status("task-001", TaskStatus.RUNNING)

    response = service.GetTaskStatus(
        GetTaskStatusRequest(task_id="task-001"),
        Mock(),
    )

    assert response.found is True
    assert response.status == TASK_STATUS_RUNNING
    assert response.error_message == ""


def test_get_task_status_returns_not_found_for_unknown_task():
    service = make_service()

    response = service.GetTaskStatus(
        GetTaskStatusRequest(task_id="unknown-task"),
        Mock(),
    )

    assert response.found is False
    assert response.status == TASK_STATUS_UNSPECIFIED
    assert response.error_message == "No status for task unknown-task"


def test_get_task_status_rejects_empty_task_id():
    service = make_service()
    context = Mock()

    response = service.GetTaskStatus(
        GetTaskStatusRequest(task_id="  "),
        context,
    )

    assert response.found is False
    assert response.status == TASK_STATUS_UNSPECIFIED
    assert response.error_message == "Task ID cannot be empty"
    context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details.assert_called_once_with("Task ID cannot be empty")


def test_cancel_task_cancels_queued_task():
    service = make_service()

    task = TestTask(
        task_id="queued-task",
        type=TaskType.TEST,
        symbol="xeb",
        params=[1, 2, 3],
        shots=10,
    )
    service.executor.execute(task, user="alice")

    response = service.CancelTask(
        CancelTaskRequest(task_id=task.task_id),
        Mock(),
    )

    assert response.found is True
    assert response.cancelled is True
    assert response.error_message == ""
    assert service.executor.get_task_status(task.task_id) == TaskStatus.CANCELLED


def test_cancel_task_rejects_running_task():
    service = make_service()
    service.executor.set_task_status("running-task", TaskStatus.RUNNING)

    response = service.CancelTask(
        CancelTaskRequest(task_id="running-task"),
        Mock(),
    )

    assert response.found is True
    assert response.cancelled is False
    assert response.error_message == (
        "Task running-task cannot be cancelled from status running"
    )
    assert service.executor.get_task_status("running-task") == TaskStatus.RUNNING


def test_cancel_task_returns_not_found_for_unknown_task():
    service = make_service()

    response = service.CancelTask(
        CancelTaskRequest(task_id="unknown-task"),
        Mock(),
    )

    assert response.found is False
    assert response.cancelled is False
    assert response.error_message == "No status for task unknown-task"


def test_cancel_task_rejects_empty_task_id():
    service = make_service()
    context = Mock()

    response = service.CancelTask(
        CancelTaskRequest(task_id="  "),
        context,
    )

    assert response.found is False
    assert response.cancelled is False
    assert response.error_message == "Task ID cannot be empty"
    context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
    context.set_details.assert_called_once_with("Task ID cannot be empty")
