import sys
import os
import json
import re
import ast
import time
import operator
import threading
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

import ollama
import faiss
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator


# =====================================================================
# SYSTEM CONFIGURATION & MULTI-AGENT THRESHOLDS
# =====================================================================

MAX_ROUNDS = 8
MAX_SCHEMA_REPAIR_ATTEMPTS = 2
MAX_FILE_WRITE_BYTES = 50_000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SANDBOX_DIR = os.path.join(BASE_DIR, "sandbox")
os.makedirs(SANDBOX_DIR, exist_ok=True)

# Knowledge base RAG paths
KNOWLEDGE_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
KNOWLEDGE_INDEX_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "index.faiss")
KNOWLEDGE_CHUNKS_FILE = os.path.join(KNOWLEDGE_STORE_DIR, "chunks.json")
TOP_K_KNOWLEDGE = 3

# Semantic Memory Store paths
MEMORY_STORE_DIR = os.path.join(BASE_DIR, "memory_store")
MEMORY_INDEX_FILE = os.path.join(MEMORY_STORE_DIR, "memory_lifecycle_index.faiss")
MEMORY_RECORDS_FILE = os.path.join(MEMORY_STORE_DIR, "memory_lifecycle_records.json")
MEMORY_EMBED_DIM = 768
TOP_K_MEMORY = 4
MEMORY_SIMILARITY_THRESHOLD = 0.40

# Console lock for clean synchronized printing
console_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with console_lock:
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            clean_args = [
                str(a).encode(encoding, errors="replace").decode(encoding)
                for a in args
            ]
            print(*clean_args, **kwargs)


def sanitize_agent_output(text: str) -> str:
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        cleaned.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        cleaned = cleaned.encode("ascii", errors="ignore").decode("ascii")
    return cleaned.strip()


# =====================================================================
# PYDANTIC DATA CONTRACTS FOR MULTI-AGENT ORCHESTRATION
# =====================================================================

class TaskRecord(BaseModel):
    round_num: int
    agent: str
    task_description: str
    output_key: str
    result_preview: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PlannedStep(BaseModel):
    id: int
    target_agent: Literal["researcher", "computation", "critic"]
    task: str
    output_key: str
    status: Literal["PENDING", "COMPLETED", "FAILED"] = "PENDING"


class DecompositionPlan(BaseModel):
    plan: list[PlannedStep] = Field(description="Step-by-step delegation plan addressing every aspect of the user goal.")


class BlackboardState(BaseModel):
    user_goal: str
    delegation_plan: list[PlannedStep] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    task_history: list[TaskRecord] = Field(default_factory=list)
    user_profile_facts: list[str] = Field(default_factory=list)
    critic_notes: list[str] = Field(default_factory=list)
    status: Literal["IN_PROGRESS", "NEEDS_REVISION", "COMPLETED", "FAILED"] = "IN_PROGRESS"


class SupervisorDecision(BaseModel):
    action: Literal["DELEGATE", "FINISH"] = Field(description="'DELEGATE' to assign a task to a specialist, or 'FINISH' if user goal is completely answered.")
    target_agent: Optional[Literal["researcher", "computation", "critic"]] = Field(default=None, description="The specialist agent to invoke (required for DELEGATE).")
    task_description: str = Field(default="", description="Specific, atomic mission instruction for the delegated specialist.")
    output_key: str = Field(default="artifact", description="Key name where this worker's output will be stored in the blackboard artifacts.")
    final_synthesis: Optional[str] = Field(default=None, description="Comprehensive final answer to user (required for FINISH).")

    @field_validator("target_agent")
    @classmethod
    def validate_target_agent(cls, v, info):
        action = info.data.get("action")
        if action == "DELEGATE" and not v:
            raise ValueError("target_agent is required when action is 'DELEGATE'.")
        return v


class SpecialistOutput(BaseModel):
    agent_name: str
    success: bool
    data: Any = Field(description="The primary deliverable produced by the specialist.")
    summary: str = Field(description="A concise summary of findings/results for the supervisor.")
