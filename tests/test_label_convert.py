from pathlib import Path

from floor.label_convert import parse_XML


def test_parse_XML():
    p = "data/file_0.xml"
    parse_XML(p, Path("."), Path("."))
