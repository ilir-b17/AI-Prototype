# tests/conftest.py
# Shared fixtures and pytest-asyncio configuration.
import asyncio
import types
import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock

# Ensure the project root is on sys.path so `src.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core import cloud_redaction
import src.core.orchestrator as orchestrator_module
from src.core.orchestrator import Orchestrator
from src.core.routing_assessor import RoutingAssessor
from src.core.synthesis_pipeline import SynthesisPipeline


async def _noop_async(*_args, **_kwargs):
	return None


def _ensure_energy_account(instance: Orchestrator) -> None:
	legacy_lock = getattr(instance, "_predictive_energy_budget_lock", None)
	if not hasattr(instance, "_energy_budget_lock"):
		instance._energy_budget_lock = legacy_lock if isinstance(legacy_lock, asyncio.Lock) else asyncio.Lock()

	if not hasattr(instance, "_energy_budget"):
		legacy_remaining = getattr(instance, "_predictive_energy_budget_remaining", None)
		if legacy_remaining is None:
			legacy_remaining = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
		instance._energy_budget = max(0, int(legacy_remaining))

	instance._predictive_energy_budget_lock = instance._energy_budget_lock
	instance._predictive_energy_budget_remaining = int(instance._energy_budget)


def _sync_legacy_energy_aliases(instance: Orchestrator) -> None:
	if hasattr(instance, "_energy_budget_lock"):
		instance._predictive_energy_budget_lock = instance._energy_budget_lock
	if hasattr(instance, "_energy_budget"):
		instance._predictive_energy_budget_remaining = int(instance._energy_budget)


def _ensure_pending_state_maps(instance: Orchestrator) -> None:
	if not hasattr(instance, "pending_tool_approval"):
		instance.pending_tool_approval = {}
	if not hasattr(instance, "pending_mfa"):
		instance.pending_mfa = {}
	if not hasattr(instance, "pending_hitl_state"):
		instance.pending_hitl_state = {}


def _ensure_routing_assessor(instance: Orchestrator) -> RoutingAssessor:
	assessor = getattr(instance, "routing_assessor", None)
	if assessor is not None:
		return assessor

	registry = getattr(getattr(instance, "cognitive_router", None), "registry", None)
	if registry is None:
		class _EmptyRegistry:
			def get_schemas(self):
				return []

		registry = _EmptyRegistry()

	assessor = RoutingAssessor(registry)
	instance.routing_assessor = assessor
	return assessor


def _ensure_synthesis_pipeline(instance: Orchestrator) -> SynthesisPipeline | object:
	pipeline = getattr(instance, "synthesis_pipeline", None)
	if pipeline is not None:
		return pipeline

	_ensure_pending_state_maps(instance)

	cognitive_router = getattr(instance, "cognitive_router", None)
	if cognitive_router is None:
		cognitive_router = MagicMock()
		cognitive_router.get_system_2_available.return_value = False
		instance.cognitive_router = cognitive_router

	ledger_memory = getattr(instance, "ledger_memory", None)
	if ledger_memory is None:
		ledger_memory = MagicMock()
		ledger_memory.create_synthesis_run = AsyncMock(return_value=None)
		ledger_memory.update_synthesis_run_status = AsyncMock()
		ledger_memory.append_synthesis_attempt = AsyncMock()
		ledger_memory.save_pending_approval = AsyncMock()
		ledger_memory.clear_pending_approval = AsyncMock()
		ledger_memory.register_tool = AsyncMock()
		ledger_memory.approve_tool = AsyncMock()
		ledger_memory.count_synthesis_runs_by_tool_status = AsyncMock(return_value=0)
		ledger_memory.count_synthesis_failures_fuzzy = AsyncMock(return_value=0)
		ledger_memory.count_synthesis_runs_for_user_window = AsyncMock(return_value=0)
		instance.ledger_memory = ledger_memory

	core_memory = getattr(instance, "core_memory", None)
	if core_memory is None:
		core_memory = MagicMock()
		core_memory.get_all = AsyncMock(return_value={})
		core_memory.update = AsyncMock()
		instance.core_memory = core_memory

	on_tools_changed = getattr(instance, "_invalidate_capabilities_cache", None)
	if not callable(on_tools_changed):
		on_tools_changed = lambda: None

	process_message_fn = getattr(instance, "process_message", None)
	if not callable(process_message_fn):
		process_message_fn = _noop_async

	route_to_system_2_redacted = getattr(instance, "_route_to_system_2_redacted", None)
	if not callable(route_to_system_2_redacted):
		async def _default_route_to_system_2_redacted(*_args, **_kwargs):
			return MagicMock(status="error", content="")
		route_to_system_2_redacted = _default_route_to_system_2_redacted

	fire_and_forget = getattr(instance, "_fire_and_forget", None)
	if not callable(fire_and_forget):
		def _default_fire_and_forget(coro):
			return asyncio.create_task(coro)
		fire_and_forget = _default_fire_and_forget

	notify_admin = getattr(instance, "_notify_admin", None)
	if not callable(notify_admin):
		notify_admin = _noop_async

	pipeline = SynthesisPipeline(
		cognitive_router=cognitive_router,
		ledger_memory=ledger_memory,
		core_memory=core_memory,
		pending_tool_approval=instance.pending_tool_approval,
		on_tools_changed=on_tools_changed,
		process_message_fn=process_message_fn,
		route_to_system_2_redacted=route_to_system_2_redacted,
		fire_and_forget=fire_and_forget,
		notify_admin=notify_admin,
		outbound_queue=getattr(instance, "outbound_queue", None),
	)

	if "_run_synthesis_self_test" in getattr(instance, "__dict__", {}):
		pipeline._run_synthesis_self_test = instance._run_synthesis_self_test
	if "_execute_synthesis_repair_loop" in getattr(instance, "__dict__", {}):
		pipeline._execute_synthesis_repair_loop = instance._execute_synthesis_repair_loop

	instance.synthesis_pipeline = pipeline
	return pipeline


# Compatibility: redaction helpers moved from Orchestrator to src.core.cloud_redaction
if not hasattr(Orchestrator, "_redact_text_for_cloud"):
	Orchestrator._redact_text_for_cloud = staticmethod(cloud_redaction.redact_text_for_cloud)

if not hasattr(Orchestrator, "_redact_messages_for_cloud"):
	def _compat_redact_messages_for_cloud(self, messages, *, allow_sensitive_context=False):
		return cloud_redaction.redact_messages_for_cloud(
			messages,
			allow_sensitive_context=allow_sensitive_context,
		)
	Orchestrator._redact_messages_for_cloud = _compat_redact_messages_for_cloud

if not hasattr(Orchestrator, "_cloud_payload_audit_sha256"):
	Orchestrator._cloud_payload_audit_sha256 = staticmethod(cloud_redaction.compute_payload_sha256)


# Compatibility: routing helpers moved to RoutingAssessor
if not hasattr(Orchestrator, "_extract_multiple_tickers"):
	Orchestrator._extract_multiple_tickers = staticmethod(RoutingAssessor._extract_multiple_tickers)

if not hasattr(Orchestrator, "_assess_request_route"):
	def _compat_assess_request_route(self, user_message, chat_history=None):
		assessor = _ensure_routing_assessor(self)
		return assessor.assess_request_route(user_message, chat_history)

	Orchestrator._assess_request_route = _compat_assess_request_route


# Compatibility: synthesis helpers moved to SynthesisPipeline
if not hasattr(Orchestrator, "_compute_synthesis_proof_sha256"):
	Orchestrator._compute_synthesis_proof_sha256 = staticmethod(SynthesisPipeline._compute_synthesis_proof_sha256)

if not hasattr(Orchestrator, "_run_synthesis_self_test"):
	async def _compat_run_synthesis_self_test(self, synthesis):
		pipeline = _ensure_synthesis_pipeline(self)
		return await pipeline._run_synthesis_self_test(synthesis)

	Orchestrator._run_synthesis_self_test = _compat_run_synthesis_self_test

if not hasattr(Orchestrator, "tool_synthesis_node"):
	async def _compat_tool_synthesis_node(self, state, router_result):
		pipeline = _ensure_synthesis_pipeline(self)
		if hasattr(self, "_run_synthesis_self_test"):
			pipeline._run_synthesis_self_test = self._run_synthesis_self_test
		if hasattr(self, "_execute_synthesis_repair_loop"):
			pipeline._execute_synthesis_repair_loop = self._execute_synthesis_repair_loop
		return await pipeline.tool_synthesis_node(state, router_result)

	Orchestrator.tool_synthesis_node = _compat_tool_synthesis_node

if not hasattr(Orchestrator, "_async_tool_synthesis"):
	async def _compat_async_tool_synthesis(self, user_id, result, state):
		override_node = getattr(self, "__dict__", {}).get("tool_synthesis_node")
		if callable(override_node):
			try:
				async def _run_synthesis():
					hitl_prompt = await override_node(state, result)
					await self._notify_admin(hitl_prompt)

				await asyncio.wait_for(
					_run_synthesis(),
					timeout=float(getattr(orchestrator_module, "_SYNTHESIS_LOCKOUT_TTL_SECONDS", 600)),
				)
			except asyncio.TimeoutError:
				try:
					await self._notify_admin(
						"CRITICAL: Tool synthesis timed out and was aborted. "
						f"User: {user_id}. Timeout: {getattr(orchestrator_module, '_SYNTHESIS_LOCKOUT_TTL_SECONDS', 600)}s."
					)
				except Exception:
					pass
			finally:
				in_progress = getattr(self, "_synthesis_in_progress", None)
				if isinstance(in_progress, dict):
					in_progress.pop(user_id, None)
			return

		pipeline = _ensure_synthesis_pipeline(self)
		await pipeline.async_tool_synthesis(user_id, result, state)

	Orchestrator._async_tool_synthesis = _compat_async_tool_synthesis


# Compatibility: energy account field migration for tests that still build __new__ objects.
_orig_async_init = Orchestrator.async_init


async def _compat_async_init(self):
	_ensure_pending_state_maps(self)
	_ensure_energy_account(self)
	_ensure_routing_assessor(self)
	if not hasattr(self, "_synthesis_in_progress"):
		self._synthesis_in_progress = {}
	if not hasattr(self, "outbound_queue"):
		self.outbound_queue = None
	return await _orig_async_init(self)


Orchestrator.async_init = _compat_async_init


_orig_process_message = Orchestrator.process_message


async def _compat_process_message(self, user_message, user_id):
	_ensure_pending_state_maps(self)
	_ensure_energy_account(self)
	_ensure_routing_assessor(self)
	_ensure_synthesis_pipeline(self)
	result = await _orig_process_message(self, user_message, user_id)
	_sync_legacy_energy_aliases(self)
	return result


Orchestrator.process_message = _compat_process_message


_orig_try_fast_path_response = Orchestrator._try_fast_path_response


async def _compat_try_fast_path_response(self, state_or_user_message, state=None):
	_ensure_routing_assessor(self)
	return await _orig_try_fast_path_response(self, state_or_user_message, state)


Orchestrator._try_fast_path_response = _compat_try_fast_path_response


_orig_should_invoke_goal_planner = Orchestrator._should_invoke_goal_planner


def _compat_should_invoke_goal_planner(self, user_message):
	_ensure_routing_assessor(self)
	return _orig_should_invoke_goal_planner(self, user_message)


Orchestrator._should_invoke_goal_planner = _compat_should_invoke_goal_planner


_orig_try_ad_hoc_dispatch_energy_gate = Orchestrator._try_ad_hoc_dispatch_energy_gate


async def _compat_try_ad_hoc_dispatch_energy_gate(self, state, *, dispatch_context):
	# Compatibility for lightweight __new__ orchestrator fixtures that do not
	# wire energy judge/ROI dependencies.
	if getattr(self, "energy_judge", None) is None or getattr(self, "energy_roi_engine", None) is None:
		return None
	_ensure_energy_account(self)
	return await _orig_try_ad_hoc_dispatch_energy_gate(
		self,
		state,
		dispatch_context=dispatch_context,
	)


Orchestrator._try_ad_hoc_dispatch_energy_gate = _compat_try_ad_hoc_dispatch_energy_gate


_orig_handle_blocked_result = Orchestrator._handle_blocked_result


async def _compat_handle_blocked_result(self, result, user_id, state):
	_ensure_synthesis_pipeline(self)
	return await _orig_handle_blocked_result(self, result, user_id, state)


Orchestrator._handle_blocked_result = _compat_handle_blocked_result


_orig_get_predictive_energy_budget_remaining = Orchestrator._get_predictive_energy_budget_remaining


async def _compat_get_predictive_energy_budget_remaining(self):
	_ensure_energy_account(self)
	value = await _orig_get_predictive_energy_budget_remaining(self)
	_sync_legacy_energy_aliases(self)
	return value


Orchestrator._get_predictive_energy_budget_remaining = _compat_get_predictive_energy_budget_remaining


_orig_refund_predictive_energy_budget = Orchestrator._refund_predictive_energy_budget


async def _compat_refund_predictive_energy_budget(self, amount, reason):
	_ensure_energy_account(self)
	await _orig_refund_predictive_energy_budget(self, amount, reason)
	_sync_legacy_energy_aliases(self)


Orchestrator._refund_predictive_energy_budget = _compat_refund_predictive_energy_budget


_orig_try_reserve_predictive_energy_budget = Orchestrator._try_reserve_predictive_energy_budget


async def _compat_try_reserve_predictive_energy_budget(self, *, predicted_cost, min_reserve, reason):
	_ensure_energy_account(self)
	reserved = await _orig_try_reserve_predictive_energy_budget(
		self,
		predicted_cost=predicted_cost,
		min_reserve=min_reserve,
		reason=reason,
	)
	_sync_legacy_energy_aliases(self)
	return reserved


Orchestrator._try_reserve_predictive_energy_budget = _compat_try_reserve_predictive_energy_budget


_orig_tick_predictive_energy_budget = Orchestrator._tick_predictive_energy_budget


async def _compat_tick_predictive_energy_budget(self, amount, reason):
	_ensure_energy_account(self)
	applied = await _orig_tick_predictive_energy_budget(self, amount, reason)
	_sync_legacy_energy_aliases(self)
	return applied


Orchestrator._tick_predictive_energy_budget = _compat_tick_predictive_energy_budget


if not hasattr(Orchestrator, "_predictive_budget_tracking_available"):
	def _compat_predictive_budget_tracking_available(self):
		_ensure_energy_account(self)
		return True

	Orchestrator._predictive_budget_tracking_available = _compat_predictive_budget_tracking_available


if not hasattr(Orchestrator, "_energy_gate_available"):
	def _compat_energy_gate_available(self):
		_ensure_energy_account(self)
		return True

	Orchestrator._energy_gate_available = _compat_energy_gate_available


if not hasattr(Orchestrator, "_apply_predictive_energy_tick_locked"):
	def _compat_apply_predictive_energy_tick_locked(self, amount: int) -> int:
		_ensure_energy_account(self)
		if amount <= 0:
			return 0
		cap = max(0, int(os.getenv("INITIAL_ENERGY_BUDGET", "100")))
		before = int(self._energy_budget)
		self._energy_budget = min(cap, before + int(amount))
		_sync_legacy_energy_aliases(self)
		return int(self._energy_budget) - before

	Orchestrator._apply_predictive_energy_tick_locked = _compat_apply_predictive_energy_tick_locked


if not hasattr(Orchestrator, "_compute_predictive_energy_replenishment_points_wallclock_locked"):
	def _compat_compute_predictive_energy_replenishment_points_wallclock_locked(self, *, now=None) -> int:
		_ensure_energy_account(self)
		current_time = float(time.time() if now is None else now)
		last_replenished_at = float(
			getattr(self, "_predictive_energy_budget_last_replenished_at", current_time)
		)
		if current_time <= last_replenished_at:
			return 0

		replenish_per_hour = self._resolve_energy_replenish_per_hour()
		if replenish_per_hour <= 0:
			self._predictive_energy_budget_last_replenished_at = current_time
			return 0

		elapsed_seconds = current_time - last_replenished_at
		replenishment_points = int((elapsed_seconds / 3600.0) * replenish_per_hour)
		if replenishment_points <= 0:
			return 0

		consumed_seconds = (float(replenishment_points) / replenish_per_hour) * 3600.0
		self._predictive_energy_budget_last_replenished_at = last_replenished_at + consumed_seconds
		return replenishment_points

	Orchestrator._compute_predictive_energy_replenishment_points_wallclock_locked = (
		_compat_compute_predictive_energy_replenishment_points_wallclock_locked
	)


if not hasattr(Orchestrator, "_replenish_predictive_energy_budget_wallclock_locked"):
	def _compat_replenish_predictive_energy_budget_wallclock_locked(self, *, now=None) -> int:
		points = self._compute_predictive_energy_replenishment_points_wallclock_locked(now=now)
		return self._apply_predictive_energy_tick_locked(points)

	Orchestrator._replenish_predictive_energy_budget_wallclock_locked = _compat_replenish_predictive_energy_budget_wallclock_locked
