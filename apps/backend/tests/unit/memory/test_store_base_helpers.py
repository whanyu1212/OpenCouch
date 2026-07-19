from __future__ import annotations

import json

import pytest

from agent.memory.store.base import (
    build_store_record,
    parse_store_record_value,
    prepare_memory_record_fields,
    unpack_memory_namespace,
)


def test_unpack_memory_namespace_accepts_owner_kind_tuple() -> None:
    assert unpack_memory_namespace(("user-1", "semantic")) == ("user-1", "semantic")


def test_unpack_memory_namespace_rejects_wrong_tuple_shape() -> None:
    with pytest.raises(ValueError, match="MemoryStore namespace"):
        unpack_memory_namespace(("user-1", "semantic", "extra"))


def test_prepare_memory_record_fields_derives_defaults() -> None:
    fields = prepare_memory_record_fields(
        {"category": "relationship", "created_at": "2026-01-01T00:00:00Z"},
        embedding=[0.1, 0.2],
    )

    assert fields.category == "relationship"
    assert fields.created_at == "2026-01-01T00:00:00Z"
    assert fields.last_referenced_at == "2026-01-01T00:00:00Z"
    assert fields.dormant_at is None
    assert fields.user_visible is True
    assert fields.embedding_dim == 2
    assert json.loads(fields.serialized_value) == {
        "category": "relationship",
        "created_at": "2026-01-01T00:00:00Z",
    }


def test_prepare_memory_record_fields_preserves_false_user_visible() -> None:
    fields = prepare_memory_record_fields(
        {"created_at": "2026-01-01T00:00:00Z", "user_visible": False},
        embedding=None,
    )

    assert fields.user_visible is False
    assert fields.embedding_dim is None


def test_parse_store_record_value_accepts_dict() -> None:
    value = {"evidence_quote": "I worry about work"}

    assert parse_store_record_value(value) == value


def test_parse_store_record_value_parses_json_string() -> None:
    assert parse_store_record_value('{"evidence_quote": "I worry about work"}') == {
        "evidence_quote": "I worry about work"
    }


def test_build_store_record_parses_value_and_coerces_key() -> None:
    record = build_store_record(
        namespace=("user-1", "semantic"),
        key=123,
        value='{"evidence_quote": "I worry about work"}',
        embedding=[0.1, 0.2],
        embedding_model="embedding-test",
    )

    assert record.namespace == ("user-1", "semantic")
    assert record.key == "123"
    assert record.value == {"evidence_quote": "I worry about work"}
    assert record.embedding == [0.1, 0.2]
    assert record.embedding_model == "embedding-test"
