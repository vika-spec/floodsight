"""Raw Level-1 GRD chain: thermal-noise LUT parsing (range x azimuth blocks) and removal before calibration."""
import numpy as np

from floodsight.fetch.s1_local import noise_lut, parse_noise

XML = """<?xml version="1.0"?><noise><noiseRangeVectorList count="2">
<noiseRangeVector><line>0</line><pixel>0 100</pixel><noiseRangeLut>10 20</noiseRangeLut></noiseRangeVector>
<noiseRangeVector><line>100</line><pixel>0 100</pixel><noiseRangeLut>30 40</noiseRangeLut></noiseRangeVector>
</noiseRangeVectorList><noiseAzimuthVectorList count="1"><noiseAzimuthVector>
<firstAzimuthLine>0</firstAzimuthLine><firstRangeSample>0</firstRangeSample>
<lastAzimuthLine>100</lastAzimuthLine><lastRangeSample>49</lastRangeSample>
<line>0 100</line><noiseAzimuthLut>1 3</noiseAzimuthLut></noiseAzimuthVector></noiseAzimuthVectorList></noise>"""


def test_range_times_azimuth(tmp_path):
    f = tmp_path / "noise-s1a-iw-grd-vv-x.xml"
    f.write_text(XML)
    nz = parse_noise(f)
    N = noise_lut(nz, 0, 101, 0, 101)
    assert np.isclose(N[0, 0], 10 * 1) and np.isclose(N[0, 100], 20)       # outside the azimuth block: x1
    assert np.isclose(N[100, 0], 30 * 3) and np.isclose(N[50, 0], 20 * 2)  # inside: range x azimuth, interpolated
    assert np.isclose(N[100, 100], 40)


def test_old_format_and_missing(tmp_path):
    f = tmp_path / "noise-old.xml"
    f.write_text("<noise><noiseVectorList><noiseVector><line>0</line><pixel>0 10</pixel>"
                 "<noiseLut>5 5</noiseLut></noiseVector></noiseVectorList></noise>")
    N = noise_lut(parse_noise(f), 0, 3, 0, 11)
    assert np.allclose(N, 5)
    assert noise_lut(None, 0, 3, 0, 3) is None
