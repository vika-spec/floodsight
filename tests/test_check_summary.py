"""Start-page example cards: headline check numbers come from EMS or Kuro Siwo comparisons, never invented."""
import json

from server.api import KS_VARIANT, _check_summary, _label_check


def test_no_check_gives_none(tmp_path):
    assert _check_summary(tmp_path) is None


def test_ems_check(tmp_path):
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "validation.json").write_text(json.dumps(
        {"reference": "Copernicus EMS (x)", "flood_extent": {"iou": 0.466, "precision": 0.958}}))
    assert _check_summary(tmp_path) == {"reference": "Copernicus EMS (x)", "iou": 0.466, "precision": 0.958}


def test_kurosiwo_check_reports_whole_map_and_training_role(tmp_path):
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "kurosiwo_1111009.json").write_text(json.dumps({
        "event": "1111009", "model_role_for_event": "train",
        "scores": {"final map: flood water": {"iou": 0.9, "precision": 0.9, "recall": 0.9},
                   KS_VARIANT: {"iou": 0.848, "precision": 0.93, "recall": 0.9}}}))
    k = _label_check(tmp_path)
    assert k["iou"] == 0.848 and k["model_role"] == "train" and k["variant"] == KS_VARIANT
    assert _check_summary(tmp_path)["iou"] == 0.848


def test_damaged_file_shows_nothing(tmp_path):
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "validation.json").write_text("{not json")
    assert _check_summary(tmp_path) is None
