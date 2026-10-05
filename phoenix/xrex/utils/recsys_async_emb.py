# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P

if TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )
    from xrex.cuda.row_emb import row_emb

    ContextHandle = async_emb.AsyncEmbContextHandle | row_emb.RowEmbContextHandle
else:
    ContextHandle = object


def is_row_sharded(context_handle: ContextHandle) -> bool:
    return hasattr(context_handle, "rows_per_rank")


def table_spec(context_handle: ContextHandle) -> P:
    if is_row_sharded(context_handle):
        return P(context_handle.table_axis, None)
    return P(None, context_handle.table_axis)


def _data_shards(context_handle: ContextHandle) -> int:
    return math.prod(context_handle.mesh.shape[axis] for axis in context_handle.data_axis)


def _rows_spec(context_handle: ContextHandle, ndim: int, rows_axis: int) -> P:
    rows_axis = max(ndim + rows_axis, 0)
    return P(*(context_handle.data_axis if i == rows_axis else None for i in range(ndim)))


def kernel_bindings(context_handle: ContextHandle):
    if is_row_sharded(context_handle):
        from xrex.cuda.row_emb import row_emb

        return row_emb
    from xrex.cuda.async_emb import async_emb

    return async_emb


def kernel_api(context_handle: ContextHandle):
    bindings = kernel_bindings(context_handle)
    if is_row_sharded(context_handle):
        return bindings.row_emb_api
    return bindings.async_emb_api


def lookup_start(
    context_handle: ContextHandle,
    token_ids: jax.Array,
    table: jax.Array,
    gate: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    bindings = kernel_bindings(context_handle)
    spec = table_spec(context_handle)

    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(_rows_spec(context_handle, token_ids.ndim, -2), spec, P()),
        out_specs=(spec, P(context_handle.data_axis, None)),
        check_vma=False,
    )
    def start(
        token_ids: jax.Array, table: jax.Array, gate: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        token_ids = token_ids.reshape(-1, token_ids.shape[-1])
        table_out, lookup_pin = bindings.lookup_start(token_ids, table, gate, context_handle)
        return table_out, lookup_pin[None, :]

    return start(token_ids, table, gate)


def lookup_done(
    context_handle: ContextHandle,
    lookup_pin: jax.Array,
    token_ids_shape: tuple[int, ...] | None = None,
) -> jax.Array:
    bindings = kernel_bindings(context_handle)
    local_shape = (-1, context_handle.emb_width)
    if token_ids_shape is not None:
        *leading, users, tokens = token_ids_shape
        users //= _data_shards(context_handle)
        local_shape = (*leading, users, tokens, context_handle.emb_width)

    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(P(context_handle.data_axis, None),),
        out_specs=_rows_spec(context_handle, len(local_shape), -3),
        check_vma=False,
    )
    def done(lookup_pin: jax.Array) -> jax.Array:
        return bindings.lookup_done(lookup_pin, context_handle).reshape(local_shape)

    return done(lookup_pin)


def depend(context_handle: ContextHandle, x: jax.Array, pin: jax.Array) -> jax.Array:
    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(P(context_handle.data_axis), P()),
        out_specs=P(context_handle.data_axis),
        check_vma=False,
    )
    def add_pin(x: jax.Array, pin: jax.Array) -> jax.Array:
        return x.at[(0,) * x.ndim].add(pin.astype(x.dtype)[0])

    return add_pin(x, pin)


def zero_pin(context_handle: ContextHandle, x: jax.Array) -> jax.Array:
    @shard_map(
        mesh=context_handle.mesh,
        in_specs=P(context_handle.data_axis),
        out_specs=P(),
        check_vma=False,
    )
    def pin(x: jax.Array) -> jax.Array:
        first = x.reshape(-1)[:1].astype(jnp.float32)
        return jnp.where(jnp.isfinite(first), first, 0.0) * 0.0

    return pin(x)


def stage_update(
    context_handle: ContextHandle,
    grads: jax.Array,
    segment_ids: jax.Array,
    unique_tokens: jax.Array,
    pending: jax.Array,
    gate: jax.Array,
) -> jax.Array:
    bindings = kernel_bindings(context_handle)

    if is_row_sharded(context_handle):

        @shard_map(
            mesh=context_handle.mesh,
            in_specs=(_rows_spec(context_handle, grads.ndim, -3), P(), P()),
            out_specs=P(context_handle.data_axis, None),
            check_vma=False,
        )
        def stage_rows(grads: jax.Array, pending: jax.Array, gate: jax.Array) -> jax.Array:
            grads = grads.reshape(-1, grads.shape[-1])
            pin = bindings.stage_update(grads, pending, gate, context_handle)
            return pin[None, :]

        return stage_rows(grads, pending, gate)

    @shard_map(
        mesh=context_handle.mesh,
        in_specs=(
            _rows_spec(context_handle, grads.ndim, -3),
            _rows_spec(context_handle, segment_ids.ndim, -2),
            P(),
            P(),
            P(),
        ),
        out_specs=P(context_handle.data_axis, None),
        check_vma=False,
    )
    def stage(
        grads: jax.Array,
        segment_ids: jax.Array,
        unique_tokens: jax.Array,
        pending: jax.Array,
        gate: jax.Array,
    ) -> jax.Array:
        pin = bindings.stage_update(
            grads.reshape(-1, grads.shape[-1]),
            segment_ids.reshape(-1),
            unique_tokens,
            pending,
            gate,
            context_handle,
        )
        return pin[None, :]

    return stage(grads, segment_ids, unique_tokens, pending, gate)
