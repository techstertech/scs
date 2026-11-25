"""End-to-end Java ingestion through the native parser and durable graph."""

from __future__ import annotations

from pathlib import Path

from scs.graph.models import NodeType, RelationshipType
from scs.graph.native import NativeGraph
from scs.indexing.parser.native import NativeParser
from scs.indexing.pipeline import IngestionPipeline
from scs.providers.base import ProviderMetadata
from scs.services.impact import is_test_path


class _PreJavaParser(NativeParser):
    """Simulate an earlier parser release before .java support existed."""

    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".py", ".ts", ".tsx", ".js", ".jsx", ".go"})


def _graph(tmp_path: Path) -> NativeGraph:
    provider = ProviderMetadata("test", "disabled", 2, False, "offline")
    return NativeGraph(
        database_path=tmp_path / "graph.db",
        vector_path=tmp_path / "vectors.usearch",
        provider_metadata_path=tmp_path / "provider.json",
        provider=provider,
    )


def _node_qname(graph: NativeGraph, node_id: str) -> object:
    node = graph.get_node_sync(node_id)
    assert node is not None
    return node.metadata.get("qualified_name")


def _node_file_path(graph: NativeGraph, node_id: str) -> object:
    node = graph.get_node_sync(node_id)
    assert node is not None
    return node.metadata.get("file_path")


def test_java_repository_ingests_symbols_and_cross_file_edges(
    repository: Path, tmp_path: Path
) -> None:
    files = {
        "src/main/java/com/acme/model/BaseEntity.java": """package com.acme.model;

public class BaseEntity {
    private String id;
}
""",
        "src/main/java/com/acme/model/Identifiable.java": """package com.acme.model;

public interface Identifiable {
    String getId();
}
""",
        "src/main/java/com/acme/model/User.java": """package com.acme.model;

/** Application user entity. */
public class User extends BaseEntity implements Identifiable {
    private final String name;

    public User(String name) {
        this.name = name;
    }

    @Override
    public String getId() {
        return name;
    }
}
""",
        "src/main/java/com/acme/util/Checks.java": """package com.acme.util;

public class Checks {
    public static void verify(String value) {
    }
}
""",
        "src/main/java/com/acme/service/UserService.java": """package com.acme.service;

import com.acme.model.User;
import static com.acme.util.Checks.verify;

public class UserService {
    public User find(String id) {
        verify(id);
        return build(id);
    }

    private User build(String id) {
        return new User(id);
    }
}
""",
        "src/test/java/com/acme/service/UserServiceTest.java": """package com.acme.service;

import com.acme.model.User;

public class UserServiceTest {
    void testFind() {
        UserService svc = new UserService();
        User u = svc.find("42");
    }
}
""",
    }

    for rel_path, content in files.items():
        target = repository / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    graph = _graph(tmp_path)
    result = IngestionPipeline(graph=graph, parser=NativeParser()).ingest(repository)

    assert result.files_discovered == len(files)
    assert result.files_failed == 0
    assert graph.get_all_ingested_file_languages_sync(str(repository.resolve())) == {
        rel_path: "java" for rel_path in files
    }

    user_matches = graph.search_by_name_sync("User", node_type=NodeType.CLASS)
    user_node = next(
        node
        for node in user_matches
        if node.metadata.get("qualified_name") == "com.acme.model.User"
    )
    find_matches = graph.search_by_name_sync("find", node_type=NodeType.METHOD)
    find_node = next(
        node
        for node in find_matches
        if node.metadata.get("qualified_name") == "com.acme.service.UserService.find"
    )
    user_outgoing = {
        (edge.relationship, _node_qname(graph, edge.target_id))
        for edge in graph.get_edges_sync(user_node.id, direction="outgoing")
    }
    assert (RelationshipType.INHERITS.value, "com.acme.model.BaseEntity") in user_outgoing
    assert (RelationshipType.IMPLEMENTS.value, "com.acme.model.Identifiable") in user_outgoing

    find_outgoing = {
        (edge.relationship, _node_qname(graph, edge.target_id))
        for edge in graph.get_edges_sync(find_node.id, direction="outgoing")
    }
    assert (RelationshipType.CALLS.value, "com.acme.util.Checks.verify") in find_outgoing
    assert (RelationshipType.CALLS.value, "com.acme.service.UserService.build") in find_outgoing

    file_nodes = {
        node.metadata["file_path"]: node
        for node in graph.list_nodes_sync(node_type=NodeType.FILE)
    }
    service_file = file_nodes["src/main/java/com/acme/service/UserService.java"]
    test_file = file_nodes["src/test/java/com/acme/service/UserServiceTest.java"]

    service_outgoing = {
        (edge.relationship, _node_qname(graph, edge.target_id))
        for edge in graph.get_edges_sync(service_file.id, direction="outgoing")
    }
    assert (RelationshipType.IMPORTS.value, "com.acme.model.User") in service_outgoing
    assert (RelationshipType.IMPORTS.value, "com.acme.util.Checks.verify") in service_outgoing

    test_outgoing = {
        (edge.relationship, _node_qname(graph, edge.target_id))
        for edge in graph.get_edges_sync(test_file.id, direction="outgoing")
    }
    assert (RelationshipType.IMPORTS.value, "com.acme.model.User") in test_outgoing

    # Files sharing `package com.acme.service` must keep separate ownership of their types.
    service_contained_files = {
        _node_file_path(graph, edge.target_id)
        for edge in graph.get_edges_sync(service_file.id, direction="outgoing")
        if edge.relationship == RelationshipType.CONTAINS.value
    }
    assert service_contained_files == {"src/main/java/com/acme/service/UserService.java"}

    test_contained_files = {
        _node_file_path(graph, edge.target_id)
        for edge in graph.get_edges_sync(test_file.id, direction="outgoing")
        if edge.relationship == RelationshipType.CONTAINS.value
    }
    assert test_contained_files == {"src/test/java/com/acme/service/UserServiceTest.java"}

    assert is_test_path("UserServiceTest.java")
    assert is_test_path("UserServiceTests.java")
    assert not is_test_path("UserService.java")


def test_previously_text_indexed_java_file_upgrades_incrementally(
    repository: Path, tmp_path: Path
) -> None:
    java_file = repository / "src" / "main" / "java" / "com" / "acme" / "App.java"
    java_file.parent.mkdir(parents=True)
    java_file.write_text(
        "package com.acme;\npublic class App {\n    public void start() {}\n}\n",
        encoding="utf-8",
    )

    graph = _graph(tmp_path)
    repo_root = str(repository.resolve())

    initial = IngestionPipeline(graph=graph, parser=_PreJavaParser()).ingest(repository)
    assert initial.files_failed == 0
    assert graph.get_all_ingested_file_languages_sync(repo_root) == {
        "src/main/java/com/acme/App.java": "text"
    }
    assert graph.search_by_name_sync("start", node_type=NodeType.METHOD) == []

    upgraded = IngestionPipeline(graph=graph, parser=NativeParser()).ingest(
        repository, force=False
    )
    assert upgraded.files_changed == 1
    assert upgraded.files_failed == 0
    assert graph.get_all_ingested_file_languages_sync(repo_root) == {
        "src/main/java/com/acme/App.java": "java"
    }
    methods = graph.search_by_name_sync("start", node_type=NodeType.METHOD)
    assert [node.metadata["qualified_name"] for node in methods] == [
        "com.acme.App.start"
    ]
