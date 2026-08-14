"""Keep CodeMeta synchronized with the tracked release metadata."""

import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_codemeta_matches_release_metadata():
    codemeta = json.loads((ROOT / "codemeta.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    distribution = project["name"]
    repository = project["urls"]["Repository"]
    version = project["version"]

    assert codemeta["@context"] == "https://w3id.org/codemeta/3.1"
    assert codemeta["@type"] == "SoftwareSourceCode"
    assert codemeta["name"] == "fidelis"
    assert distribution == "fidelis-memory"
    assert codemeta["version"] == version
    assert codemeta["codeRepository"] == repository
    assert codemeta["url"] == repository
    assert codemeta["issueTracker"] == project["urls"]["Issues"]
    assert codemeta["downloadUrl"] == (
        f"https://pypi.org/project/{distribution}/{version}/"
    )
    assert codemeta["releaseNotes"] == f"{repository}/releases/tag/v{version}"
