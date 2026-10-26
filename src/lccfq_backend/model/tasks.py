"""
Filename: tasks.py
Author: Santiago Nunez-Corrales
Date: 2025-08-06
Version: 1.0
Description:
    This file defines a model for tasks addressed by the backend.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import uuid
import getpass

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from .context import QPUExecutionContext

class TaskType(str, Enum):
    """Task types LCCFQ backed admits
    """
    CIRCUIT = "circuit"
    TEST = "test"
    CONTROL = "control"


class TaskStatus(str, Enum):
    """Lifecycle states for individual tasks
    """
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def generate_task_id() -> str:
    """Generate a unique task id

    :return: UUID for a task
    """
    return str(uuid.uuid4())


def get_current_user() -> str:
    """Get current users. This may need to be a
    parameter sent by lccfq_lang.

    :return: username
    """
    return getpass.getuser()


def current_timestamp() -> str:
    """Get current timestamp

    :return: timestamp
    """
    return datetime.now().isoformat()


class TaskBase(BaseModel, ABC):
    """Abstract class for lccfq_backend tasks

    """
    task_id: str = Field(default_factory=generate_task_id)
    user: str = Field(default_factory=get_current_user)
    program_id: Optional[int] = None
    timestamp: str = Field(default_factory=current_timestamp)
    tags: Dict[str, str] = Field(default_factory=dict)
    type: TaskType
    execution_context: Optional[QPUExecutionContext] = None
    model_config = ConfigDict(use_enum_values=True)


class Gate(BaseModel):
    """Matching gate definition from lccfq_lang

    """
    symbol: str
    target_qubits: List[int]
    control_qubits: List[int]
    params: List[float]

    def __repr__(self):
        """Represent a gate

        :return: string representation of gate
        """
        return f"{self.symbol} @ {self.target_qubits} ctrl {self.control_qubits} params={self.params}"


class CircuitTask(TaskBase):
    """Model for a circuit-based task

    """
    type: Literal[TaskType.CIRCUIT] = Field(default=TaskType.CIRCUIT)
    gates: List[Gate]
    shots: int


class TestTask(TaskBase):
    """Model for a test task

    NOTE: The `params` field currently uses List[int] but this may not be adequate
    for all test types. This needs to be re-evaluated to support more flexible
    parameter types (e.g., floats, nested structures, or a more generic Dict/JSON
    format depending on the test symbol requirements).

    TODO: Revisit params type definition based on actual test requirements.
    """
    __test__ = False
    type: Literal[TaskType.TEST] = Field(default=TaskType.TEST)
    symbol: str
    params: List[int]  # TODO: Re-evaluate this type - may need List[float], Dict, or Union type
    shots: int


class ControlTask(TaskBase):
    """Model for a control task

    """
    type: Literal[TaskType.CONTROL] = Field(default=TaskType.CONTROL)
    command: str
    params: Optional[List[int]] = Field(default_factory=list)