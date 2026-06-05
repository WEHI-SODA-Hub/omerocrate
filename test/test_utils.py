import pytest
from pathlib import Path
from rdflib import Graph
from omerocrate.uploader import select_first, select_many, select_one
from omerocrate.utils import uri_to_path


@pytest.fixture
def demo_graph(abstract_crate: Path) -> Graph:
    return Graph().parse(
        source=abstract_crate / "ro-crate-metadata.json", format="json-ld"
    )


def test_select_first_returns_first_when_multiple(demo_graph: Graph):
    """select_first should return the first result without raising when multiple rows exist."""
    result = select_first(demo_graph, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    assert result is not None


def test_select_first_raises_on_no_results(demo_graph: Graph):
    """select_first should raise ValueError when the query returns no rows."""
    with pytest.raises(ValueError, match="Expected at least one result"):
        select_first(
            demo_graph,
            "SELECT ?x WHERE { ?x <http://nonexistent.example/predicate> ?o }",
        )


# --- select_many ---


def test_select_many_returns_rows(demo_graph: Graph):
    """select_many should yield at least one row for a broad pattern."""
    rows = list(select_many(demo_graph, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"))
    assert len(rows) > 0


def test_select_many_returns_empty_when_no_match(demo_graph: Graph):
    """select_many should yield no rows when nothing matches."""
    rows = list(
        select_many(
            demo_graph,
            "SELECT ?x WHERE { ?x <http://nonexistent.example/predicate> ?o }",
        )
    )
    assert rows == []


def test_select_many_raises_on_non_select_query(demo_graph: Graph):
    """select_many should raise ValueError for non-SELECT queries (e.g. ASK)."""
    with pytest.raises(ValueError, match="Only SELECT queries are supported"):
        list(select_many(demo_graph, "ASK { ?s ?p ?o }"))


# --- select_one ---


def test_select_one_returns_row_when_exactly_one(demo_graph: Graph):
    """select_one should return the single row when exactly one result exists."""
    # The Person entity has exactly one schema:name triple
    row = select_one(
        demo_graph,
        "SELECT ?name WHERE { <https://isni.org/isni/0000000121235624> <http://schema.org/name> ?name }",
    )
    assert str(row["name"]) == "Wassily Kandinsky"


def test_select_one_raises_on_zero_results(demo_graph: Graph):
    """select_one should raise ValueError when the query returns no rows."""
    with pytest.raises(ValueError, match="Expected exactly one result, but got 0"):
        select_one(
            demo_graph,
            "SELECT ?x WHERE { ?x <http://nonexistent.example/predicate> ?o }",
        )


def test_select_one_raises_on_multiple_results(demo_graph: Graph):
    """select_one should raise ValueError when the query returns more than one row."""
    with pytest.raises(ValueError, match="Expected exactly one result, but got"):
        select_one(demo_graph, "SELECT ?s ?p ?o WHERE { ?s ?p ?o }")


def test_uri_to_path_valid_file_uri():
    uri = "file:///home/user/data/image.tif"
    result = uri_to_path(uri)
    assert result == Path("/home/user/data/image.tif")


def test_uri_to_path_percent_encoded():
    uri = "file:///home/user/my%20data/image%2B1.tif"
    result = uri_to_path(uri)
    assert result == Path("/home/user/my data/image+1.tif")


def test_uri_to_path_invalid_scheme_raises():
    uri = "https://example.com/image.tif"
    with pytest.raises(ValueError, match="URI scheme must be 'file'"):
        uri_to_path(uri)
