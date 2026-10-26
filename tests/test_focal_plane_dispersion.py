"""The pixels' width in wavelength from the SW focal plane, with `focal_plane_dispersion`."""
import sys

import astropy.units as u
import pytest
import yaml

from eclipse.config import Detector_SWC
from eclipse.main import main
from eclipse.readout import FocalPlane_SWC

UNIFORM = {"instrument": "SWC", "n_iter": 1, "uniform_intensity": "5000 erg / (s cm2 sr)"}


def _run(tmp_path, monkeypatch, config):
    """Run `config` and return the payload it saves."""
    main_module = sys.modules["eclipse.main"]
    saved, save = {}, main_module.save_results

    def keep(path, payload, **kwargs):
        saved["payload"] = payload
        return save(path, payload, **kwargs)

    monkeypatch.setattr(main_module, "save_results", keep)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    monkeypatch.chdir(tmp_path)
    main()
    return saved["payload"]

FOCAL_PLANE = FocalPlane_SWC.from_detector(Detector_SWC())


@pytest.mark.parametrize("wavelength", [171.073, 195.119, 211.317])
def test_the_pixel_width_is_that_of_the_row_the_wavelength_lands_on(wavelength):
    """Centred on a row, the width is the distance between that row's edges."""
    ccd, row = FOCAL_PLANE.row_of_wavelength(wavelength * u.AA)
    centre = FOCAL_PLANE.wavelength(round(row), ccd)
    edges = FOCAL_PLANE.row_edges(ccd)
    expected = abs(edges[round(row) + 1] - edges[round(row)]) / u.pixel
    assert FOCAL_PLANE.pixel_width(centre).to_value(u.AA / u.pix) == pytest.approx(
        expected.to_value(u.AA / u.pix), rel=1e-12)


def test_the_width_changes_by_under_a_per_cent_over_the_band():
    widths = [FOCAL_PLANE.pixel_width(w * u.AA).to_value(u.mAA / u.pix) for w in (171, 211)]
    assert widths[0] == pytest.approx(16.98, abs=0.01)
    assert widths[1] == pytest.approx(16.86, abs=0.01)


def test_a_run_takes_the_width_where_its_line_lands(tmp_path, monkeypatch):
    config = {**UNIFORM, "rest_wavelength": "171.073 AA",
              "simulation": {"focal_plane_dispersion": True, "noise": False}}
    payload = _run(tmp_path, monkeypatch, config)
    (combination,) = payload["results"]["all_combinations"].values()
    det = combination["config_objects"]["detector"]
    expected = FOCAL_PLANE.pixel_width(171.073 * u.AA)
    assert det.wvl_res.to_value(u.AA / u.pix) == pytest.approx(
        expected.to_value(u.AA / u.pix), rel=1e-12)
    assert combination["config_objects"]["simulation"].focal_plane_dispersion is True


@pytest.mark.parametrize("extra, message", [
    ({"detector": {"wvl_res": "16.9 mAA / pix"}}, "give one or the other"),
    ({"rest_wavelength": "198.9 AA"}, "needs the observed line on the focal plane"),
    ({"instrument": "EIS", "simulation": {"focal_plane_dispersion": True,
                                          "slit_width": "1 arcsec"}}, "for SWC only"),
])
def test_a_run_refuses_what_it_cannot_take_the_width_for(tmp_path, monkeypatch, extra, message):
    config = {**UNIFORM, "simulation": {"focal_plane_dispersion": True}, **extra}
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, monkeypatch, config)
