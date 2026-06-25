from main import app
from rag.loader import CHINESE_SEPARATORS


def test_fastapi_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/upload" in paths
    assert "/chat" in paths


def test_chinese_separators_start_with_paragraph_and_line_breaks() -> None:
    assert CHINESE_SEPARATORS[:2] == ["\n\n", "\n"]
