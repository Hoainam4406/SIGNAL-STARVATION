"""Task Filter Wrappers for NavSim Grid Environments.

Controls Axis A of Signal Starvation by filtering navigation tasks
according to difficulty level, random-walk reachability probability (sa_uniform),
or starvation status.

Classes:
    - BaseTaskFilterWrapper: Base wrapper handling NavSim set_task_filter interactions.
    - LevelFilter: Restricts tasks to specific difficulty levels (easy/medium/hard/starved).
    - SAFilter: Restricts tasks to a specified range of sa_uniform [sa_min, sa_max].
    - StarvedFilter: Restricts tasks to exclude or solely include starved tasks.
    - make_task_filtered_env: Helper factory to quickly wrap an environment.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import gymnasium as gym

VALID_LEVELS: Set[str] = {"easy", "medium", "hard", "starved"}


class BaseTaskFilterWrapper(gym.Wrapper):
    """Base wrapper for controlling task sampling via NavSim's set_task_filter API.

    Supports criteria merging so that multiple filters can be chained without
    clobbering each other's constraints.
    """

    def __init__(
        self,
        env: gym.Env,
        merge_criteria: bool = False,
        **initial_criteria: Any,
    ):
        """Initialize the task filter wrapper.

        Args:
            env: The Gymnasium environment to wrap (must expose set_task_filter on unwrapped).
            merge_criteria: If True, merges new criteria with existing criteria on the env.
                            Default is False (cleanly applies only this wrapper's criteria).
            **initial_criteria: Initial filter criteria to apply immediately.
        """
        super().__init__(env)
        self.merge_criteria = merge_criteria
        self.active_criteria: Dict[str, Any] = {}
        self.filter_stats: Dict[str, int] = {}
        self.total_resets: int = 0
        self.last_info: Optional[Dict[str, Any]] = None

        self._check_unwrapped_support()
        if initial_criteria:
            self.apply_filter(**initial_criteria)

    def _check_unwrapped_support(self) -> None:
        """Verify that the underlying environment supports task filtering."""
        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, "set_task_filter") or not callable(getattr(unwrapped, "set_task_filter")):
            raise AttributeError(
                f"The unwrapped environment {unwrapped} does not provide a callable 'set_task_filter' method. "
                "TaskFilterWrapper requires NavSim environments with episode task pools."
            )

    def apply_filter(self, **criteria: Any) -> Dict[str, int]:
        """Apply or update task filtering criteria.

        Args:
            **criteria: Keyword arguments to pass to env.unwrapped.set_task_filter.

        Returns:
            dict: Summary of matching maps and tasks, e.g. {'maps': 98, 'tasks': 673}.
        """
        unwrapped = self.env.unwrapped
        merged_criteria = {}

        if self.merge_criteria and hasattr(unwrapped, "_criteria") and isinstance(unwrapped._criteria, dict):
            merged_criteria.update(unwrapped._criteria)

        merged_criteria.update(criteria)

        stats = unwrapped.set_task_filter(**merged_criteria)
        self.active_criteria = dict(merged_criteria)
        self.filter_stats = dict(stats)
        return self.filter_stats

    def clear_filter(self) -> Dict[str, int]:
        """Clear all active task filters, restoring full dataset task availability."""
        unwrapped = self.env.unwrapped
        stats = unwrapped.set_task_filter()
        self.active_criteria = {}
        self.filter_stats = dict(stats)
        return self.filter_stats

    def get_filter_stats(self) -> Dict[str, int]:
        """Return the current count of eligible maps and tasks."""
        return dict(self.filter_stats)

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment, sampling only from the filtered task pool."""
        obs, info = self.env.reset(**kwargs)
        self.total_resets += 1
        self.last_info = info
        return obs, info


class LevelFilter(BaseTaskFilterWrapper):
    """Filters tasks by difficulty level ('easy', 'medium', 'hard', 'starved').

    Usage:
        env = LevelFilter(env, levels=['easy'])
        env = LevelFilter(env, levels='easy')
        env = LevelFilter(env, levels=['easy', 'medium'])
    """

    def __init__(
        self,
        env: gym.Env,
        levels: Union[str, Sequence[str]] = "easy",
        merge_criteria: bool = False,
    ):
        """Initialize LevelFilter.

        Args:
            env: The Gymnasium environment.
            levels: Single level string or sequence of level strings.
            merge_criteria: Whether to merge with existing criteria.
        """
        if isinstance(levels, str):
            self.levels: List[str] = [levels]
        else:
            self.levels = list(levels)

        if not self.levels:
            raise ValueError("levels must contain at least one level string.")

        invalid = set(self.levels) - VALID_LEVELS
        if invalid:
            raise ValueError(
                f"Invalid level(s): {sorted(invalid)}. Allowed levels are: {sorted(VALID_LEVELS)}"
            )

        super().__init__(env, merge_criteria=merge_criteria, level=self.levels)

    def set_levels(self, levels: Union[str, Sequence[str]]) -> Dict[str, int]:
        """Update active levels dynamically."""
        if isinstance(levels, str):
            self.levels = [levels]
        else:
            self.levels = list(levels)

        invalid = set(self.levels) - VALID_LEVELS
        if invalid:
            raise ValueError(f"Invalid level(s): {sorted(invalid)}.")

        return self.apply_filter(level=self.levels)


class SAFilter(BaseTaskFilterWrapper):
    """Filters tasks by random-walk reachability probability range: [sa_min, sa_max].

    Usage:
        env = SAFilter(env, sa_min=0.1, sa_max=1.0)
        env = SAFilter(env, sa_max=0.001)  # hard & starved tasks
    """

    def __init__(
        self,
        env: gym.Env,
        sa_min: Optional[float] = 0.1,
        sa_max: Optional[float] = 1.0,
        merge_criteria: bool = False,
    ):
        """Initialize SAFilter.

        Args:
            env: The Gymnasium environment.
            sa_min: Minimum sa_uniform bound (None = unbounded below).
            sa_max: Maximum sa_uniform bound (None = unbounded above).
            merge_criteria: Whether to merge with existing criteria.
        """
        self.sa_min = float(sa_min) if sa_min is not None else None
        self.sa_max = float(sa_max) if sa_max is not None else None

        if self.sa_min is not None and self.sa_max is not None and self.sa_min > self.sa_max:
            raise ValueError(f"sa_min ({self.sa_min}) cannot be greater than sa_max ({self.sa_max}).")

        super().__init__(
            env,
            merge_criteria=merge_criteria,
            sa_uniform=(self.sa_min, self.sa_max),
        )

    def set_range(
        self,
        sa_min: Optional[float] = None,
        sa_max: Optional[float] = None,
    ) -> Dict[str, int]:
        """Update active sa_uniform range dynamically."""
        self.sa_min = float(sa_min) if sa_min is not None else None
        self.sa_max = float(sa_max) if sa_max is not None else None
        if self.sa_min is not None and self.sa_max is not None and self.sa_min > self.sa_max:
            raise ValueError(f"sa_min ({self.sa_min}) > sa_max ({self.sa_max}).")
        return self.apply_filter(sa_uniform=(self.sa_min, self.sa_max))


class StarvedFilter(BaseTaskFilterWrapper):
    """Filters tasks to exclude or solely select starved tasks.

    Usage:
        env = StarvedFilter(env, include_starved=False)  # Exclude starved tasks (starved=False)
        env = StarvedFilter(env, include_starved=True)   # Solely select starved tasks (starved=True)
    """

    def __init__(
        self,
        env: gym.Env,
        include_starved: bool = False,
        only_starved: Optional[bool] = None,
        merge_criteria: bool = False,
    ):
        """Initialize StarvedFilter.

        Args:
            env: The Gymnasium environment.
            include_starved: If False, excludes starved tasks (starved=False).
                             If True, selects only starved tasks (starved=True).
            only_starved: Optional explicit alias. If set, takes precedence over include_starved.
            merge_criteria: Whether to merge with existing criteria.
        """
        if only_starved is not None:
            self.target_starved = bool(only_starved)
        else:
            self.target_starved = bool(include_starved)

        self.include_starved = bool(include_starved)
        super().__init__(
            env,
            merge_criteria=merge_criteria,
            starved=self.target_starved,
        )

    def set_starved_mode(self, only_starved: bool) -> Dict[str, int]:
        """Update starved selection mode dynamically."""
        self.target_starved = bool(only_starved)
        return self.apply_filter(starved=self.target_starved)


def make_task_filtered_env(
    env: gym.Env,
    levels: Optional[Union[str, Sequence[str]]] = None,
    sa_min: Optional[float] = None,
    sa_max: Optional[float] = None,
    include_starved: Optional[bool] = None,
) -> gym.Env:
    """Convenience helper to apply task filtering wrappers sequentially.

    Args:
        env: Base Gymnasium environment.
        levels: Optional level or levels to filter by.
        sa_min: Optional minimum sa_uniform bound.
        sa_max: Optional maximum sa_uniform bound.
        include_starved: Optional boolean for starved filter.

    Returns:
        gym.Env: Wrapped environment with requested task filters applied.
    """
    wrapped_env = env
    if levels is not None:
        wrapped_env = LevelFilter(wrapped_env, levels=levels, merge_criteria=True)
    if sa_min is not None or sa_max is not None:
        wrapped_env = SAFilter(wrapped_env, sa_min=sa_min, sa_max=sa_max, merge_criteria=True)
    if include_starved is not None:
        wrapped_env = StarvedFilter(wrapped_env, include_starved=include_starved, merge_criteria=True)
    return wrapped_env
