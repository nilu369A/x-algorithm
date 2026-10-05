# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Callable
from typing import Any

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def manual_axes() -> frozenset[str]:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return frozenset()
    return frozenset(
        axis
        for axis, kind in zip(mesh.axis_names, mesh.axis_types)
        if kind == jax.sharding.AxisType.Manual
    )


def _spec_axes(spec: P) -> set[str]:
    return {
        axis
        for entry in spec
        if entry is not None
        for axis in (entry if isinstance(entry, tuple) else (entry,))
    }


def _strip_axes(spec: P, drop: frozenset[str]) -> P | None:
    def strip(entry):
        if isinstance(entry, tuple):
            kept = tuple(axis for axis in entry if axis not in drop)
            return kept or None
        return None if entry in drop else entry

    stripped = P(*(strip(entry) for entry in spec))
    return stripped if _spec_axes(stripped) else None


def with_sharding_constraint(x, shardings):
    drop = manual_axes()
    if not drop:
        return jax.lax.with_sharding_constraint(x, shardings)

    def is_leaf(s) -> bool:
        return isinstance(s, (NamedSharding, P))

    specs = jax.tree.leaves(shardings, is_leaf=is_leaf)
    stripped = [_strip_axes(s.spec if isinstance(s, NamedSharding) else s, drop) for s in specs]
    if all(s is None for s in stripped):
        return x
    assert all(s is not None for s in stripped), "constraint pytree mixing dropped and kept specs"
    return jax.lax.with_sharding_constraint(
        x, jax.tree.unflatten(jax.tree.structure(shardings, is_leaf=is_leaf), stripped)
    )


def maybe_shard_map(
    f: Callable[..., Any], mesh: jax.sharding.Mesh, in_specs, out_specs, **kwargs
) -> Callable[..., Any]:
    manual = manual_axes()
    if not manual:
        return jax.shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **kwargs)
    named: set[str] = set()
    for spec in jax.tree.leaves((in_specs, out_specs), is_leaf=lambda s: isinstance(s, P)):
        if isinstance(spec, P):
            named |= _spec_axes(spec)
    assert named <= manual, f"shard_map over {sorted(named - manual)} inside a manual region"
    return f
