"""
Filename: executor.py
Author: Santiago Nunez-Corrales
Date: 2025-08-07
Version: 1.0
Description:
    This file implements the executor for task execution.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import time
from threading import Lock
from typing import Union, Optional, List, Dict

from .users import UserManager
from ..model.user import Permissions
from ..model.tasks import CircuitTask, TestTask, ControlTask, TaskType, TaskBase, TaskStatus
from ..model.results import CircuitResult, TestResult, ControlAck, TaskResult
from .queue import QPUTaskQueue, QueueEntry
from .fsm import QPUAbstraction, QPUEvent, QPUState
from .error import UnknownQPUTaskType, QPUQueueEmpty
from .hwman import make_hwman_client, HWManStatus
from .result_store import ResultStore
from ..config import BackendSettings
from ..utils.log import setup_logger


logger = setup_logger("lccfq.executor")

class QPUExecutor:
    """Representation of the QPU executor"""

    def __init__(self, config: BackendSettings, users: UserManager,result_store: Optional[ResultStore] = None):
        self.qpu = QPUAbstraction()
        self.hwman = make_hwman_client(config)
        self.queue = QPUTaskQueue()
        self.result_store = result_store
        self.users = users
        self._task_statuses: Dict[str, TaskStatus] = {}
        self._task_status_lock = Lock()

        logger.info("QPUExecutor initialized")

    def set_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Store the current lifecycle status of a task"""
        with self._task_status_lock:
            self._task_statuses[task_id] = status

        logger.info(f"Task {task_id} status updated to {status.value}")

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Return the current lifecycle status of a task"""
        with self._task_status_lock:
            return self._task_statuses.get(task_id)

    def is_qpu_online(self) -> bool:
        try:
            with open("/tmp/qpu_status.flag", "r") as f:
                status = f.read().strip()
                if status == "online":
                    if self.qpu.state != QPUState.BUSY:
                        self.qpu.state = QPUState.IDLE
                    return True
                else:
                    self.qpu.state = QPUState.UNRESPONSIVE
                    return False
        except FileNotFoundError:
            self.qpu.state = QPUState.UNRESPONSIVE
            return False

    def execute(self, task: TaskBase, user: str, context_id: Optional[str] = None, priority: int = 0) -> Optional[TaskResult]:
        if task.type not in TaskType:
            logger.error(f"Unsupported task type: {task.type}")
            raise UnknownQPUTaskType(f"Unsupported task type: {task.type}")

        entry = self.queue.enqueue(
            task=task,
            user=user,
            context_id=context_id,
            priority=priority
        )

        # Mark the task as queued after it is successfully added to the queue
        self.set_task_status(task.task_id, TaskStatus.QUEUED)

        logger.info(f"Task enqueued: {task.task_id} ({task.type}) by {user}")

        if self.qpu.state != QPUState.IDLE:
            logger.warning(f"QPU busy (state: {self.qpu.state}), deferring task {task.task_id}")
            return None

        return self._dispatch(entry.task)

    def _dispatch(self, task: Union[CircuitTask, TestTask, ControlTask]) -> Optional[Union[CircuitResult, TestResult, ControlAck]]:
        self.set_task_status(task.task_id, TaskStatus.RUNNING)

        try:
            match task.type:
                case TaskType.CIRCUIT:
                    result = self._execute_circuit(task)
                case TaskType.TEST:
                    result = self._execute_test(task)
                case TaskType.CONTROL:
                    result = self._execute_control(task)
                case _:
                    logger.error(f"Unknown task type dispatched: {task.type}")
                    raise UnknownQPUTaskType(task.type)

            if self.result_store and result is not None:
                self.result_store.save(result)

        except Exception:
            self.set_task_status(task.task_id, TaskStatus.FAILED)
            raise

        self.set_task_status(task.task_id, TaskStatus.COMPLETED)
        self._handle_deferred_tasks()
        return result

    def _execute_next(self) -> Optional[Union[CircuitResult, TestResult, ControlAck, List[Union[CircuitResult, TestResult, ControlAck]]]]:
        entry = self.queue.dequeue()
        if not entry:
            raise QPUQueueEmpty()

        context_id = entry.context_id

        if context_id:
            context_entries = self.queue.dequeue_all_for_context(context_id)
            logger.info(f"Executing context batch {context_id} ({len(context_entries)} tasks)")
            return self._execute_batched_context(context_entries)
        else:
            logger.info(f"Executing single task {entry.task.task_id} ({entry.task.type})")
            return self._dispatch(entry.task)

    def _execute_batched_context(self, context_entries: List[QueueEntry]) -> List[Union[CircuitResult, TestResult, ControlAck]]:
        users = {e.user for e in context_entries}
        logger.info(
            f"Beginning execution of context batch: {context_entries[0].context_id}, "
            f"tasks={len(context_entries)}, users={users}"
        )
        results = []
        batch_start = time.monotonic()

        for entry in context_entries:
            task: TaskBase = entry.task
            logger.debug(f"Executing task {task.task_id} in context {entry.context_id}")

            self.set_task_status(task.task_id, TaskStatus.RUNNING)

            try:
                match task.type:
                    case TaskType.CIRCUIT:
                        result = self._execute_circuit(task)
                    case TaskType.TEST:
                        result = self._execute_test(task)
                    case TaskType.CONTROL:
                        result = self._execute_control(task)
                    case _:
                        logger.error(f"Unknown task type in batch: {task.type}")
                        raise UnknownQPUTaskType(task.type)

                if self.result_store and result is not None:
                    self.result_store.save(result)

            except Exception:
                self.set_task_status(task.task_id, TaskStatus.FAILED)
                raise

            self.set_task_status(task.task_id, TaskStatus.COMPLETED)
            results.append(result)

        batch_elapsed = time.monotonic() - batch_start
        logger.info(
            f"Completed context batch {context_entries[0].context_id}, "
            f"tasks={len(results)}, elapsed={batch_elapsed:.3f}s"
        )
        self._handle_deferred_tasks()
        return results

    def _handle_deferred_tasks(self):
        while self.qpu.state == QPUState.IDLE:
            entry = self.queue.dequeue()

            if not entry:
                break

            logger.info(f"Executing deferred task {entry.task.task_id} ({entry.task.type})")
            result = self._dispatch(entry.task)
            logger.info(f"Deferred task {entry.task.task_id} completed with result: {result}")

    def _execute_circuit(self, task: CircuitTask) -> CircuitResult:

        if not self.users.check_permission(task.user, Permissions.SUBMIT_CIRCUITS):
            # TODO: Maybe add a status field to the result to indicate permission issues vs execution errors?
            return CircuitResult(task_id=task.task_id, distribution={}, raw_response={"error": "User does not have permission to submit circuits"})

        logger.debug(f"Executing CIRCUIT task {task.task_id}, gates={len(task.gates)}, shots={task.shots}")
        self.qpu.transition(QPUEvent.TASK_STARTED)
        t0 = time.monotonic()
        raw = self.hwman.run_circuit(task.gates, task.shots)
        elapsed = time.monotonic() - t0
        self.qpu.transition(QPUEvent.TASK_FINISHED)
        logger.info(f"CIRCUIT task {task.task_id} completed, elapsed={elapsed:.3f}s")
        return CircuitResult(task_id=task.task_id, distribution=raw["distribution"], raw_response=raw)

    def _execute_test(self, task: TestTask) -> TestResult:
        logger.debug(f"Executing TEST task {task.task_id}, symbol={task.symbol}, shots={task.shots}")
        self.qpu.transition(QPUEvent.TASK_STARTED)
        t0 = time.monotonic()
        raw = self.hwman.run_test(task.symbol, task.params, task.shots)
        elapsed = time.monotonic() - t0
        self.qpu.transition(QPUEvent.TASK_FINISHED)
        logger.info(f"TEST task {task.task_id} completed, elapsed={elapsed:.3f}s")
        return TestResult(task_id=task.task_id, parameters=raw, raw_response=raw)

    def _execute_control(self, task: ControlTask) -> ControlAck:
        logger.debug(f"Executing CONTROL task {task.task_id}: {task.command}")
        t0 = time.monotonic()

        match task.command:
            case "reset":

                if not self.users.check_permission(task.user, Permissions.RESET):
                    return ControlAck(task_id=task.task_id, status="error", message="User does not have permission to reset the QPU")
                self.qpu.transition(QPUEvent.RESET)
                return ControlAck(task_id=task.task_id, status="ok", message="QPU reset")

            case "retune":
                if not self.users.check_permission(task.user, Permissions.RETUNE):
                    return ControlAck(task_id=task.task_id, status="error", message="User does not have permission to retune the QPU")
                result = self.hwman.retune()
                if result.status == HWManStatus.OK:
                    self.qpu.update_observables(result.observables)
                    self.qpu.transition(QPUEvent.RETUNE)
                    return ControlAck(task_id=task.task_id, status="ok", message="QPU successfully re-tuned")
                else:
                    self.qpu.transition(QPUEvent.TUNE_FAIL)
                    return ControlAck(task_id=task.task_id, status="error", message=result.message)

            case "resetall":
                if not self.users.check_permission(task.user, Permissions.RESETALL):
                    return ControlAck(task_id=task.task_id, status="error", message="User does not have permission to perform full QPU reset")
                result = self.hwman.reset_all()
                if result.status == HWManStatus.OK:
                    self.qpu.transition(QPUEvent.RESET)
                    return ControlAck(task_id=task.task_id, status="ok", message="Full QPU reset complete")
                else:
                    self.qpu.transition(QPUEvent.TUNE_FAIL)
                    return ControlAck(task_id=task.task_id, status="error", message=result.message)

            case "qtol":
                if not self.users.check_permission(task.user, Permissions.QTOL):
                    return ControlAck(task_id=task.task_id, status="error", message="User does not have permission to perform QPU tolerance check")
                tolerance = float(task.params[0]) if task.params else 0.98
                max_retries = int(task.params[1]) if len(task.params) > 1 else 3
                logger.info(f"QTol task {task.task_id}: tolerance={tolerance:.3f}, max_retries={max_retries}")

                for attempt in range(max_retries):
                    fidelity = self.hwman.evaluate_fidelity()
                    logger.debug(f"QTol attempt {attempt + 1}/{max_retries}: fidelity={fidelity:.3f}, target={tolerance:.3f}")
                    if fidelity >= tolerance:
                        elapsed = time.monotonic() - t0
                        logger.info(f"QTol task {task.task_id} succeeded on attempt {attempt + 1}, elapsed={elapsed:.3f}s")
                        self.qpu.transition(QPUEvent.TUNE_SUCCESS)
                        return ControlAck(task_id=task.task_id, status="ok", message=f"Fidelity {fidelity:.3f} meets tolerance {tolerance:.3f}")
                    self.hwman.retune()

                fidelity = self.hwman.evaluate_fidelity()
                elapsed = time.monotonic() - t0
                if fidelity >= tolerance:
                    logger.warning(f"QTol task {task.task_id} met tolerance only after all {max_retries} retries, elapsed={elapsed:.3f}s")
                    self.qpu.transition(QPUEvent.TUNE_SUCCESS)
                    return ControlAck(task_id=task.task_id, status="warning", message=f"Fidelity {fidelity:.3f} meets tolerance after retries, but warning issued")
                else:
                    logger.error(f"QTol task {task.task_id} failed after {max_retries} retries, fidelity={fidelity:.3f} < {tolerance:.3f}, elapsed={elapsed:.3f}s")
                    self.qpu.transition(QPUEvent.TUNE_FAIL)
                    return ControlAck(task_id=task.task_id, status="error", message=f"Fidelity {fidelity:.3f} below tolerance {tolerance:.3f}")

            case _:
                logger.error(f"Unknown control command: {task.command}")
                return ControlAck(task_id=task.task_id, status="error", message=f"Unknown control: {task.command}")