import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def cfg():
    from floodsight.core import load_config
    return load_config()


@pytest.fixture(scope="session")
def scene():
    from floodsight.synthetic import make_scene
    return make_scene()


@pytest.fixture(scope="session")
def terrain(scene):
    from floodsight.preprocess.geometry import slope_deg, visibility_mask
    from floodsight.preprocess.terrain import hydro_layers
    g = scene["grid"]
    return {"hydro": hydro_layers(scene["dem"], g), "slope": slope_deg(scene["dem"], g.res),
            "vis": visibility_mask(scene["dem"], g.res, scene["incidence"], scene["heading"])}


@pytest.fixture(scope="session")
def detection(scene, terrain, cfg):
    from floodsight.detect.baseline import detect_change
    h = terrain["hydro"]
    return detect_change(scene["pre"], scene["post"], terrain["vis"]["blind"], h["hand"], terrain["slope"],
                         h["streams"], scene["grid"].res, cfg)
