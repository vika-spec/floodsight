"""Re-running a named area must reuse the saved inputs and give the same shapes as a fresh download
(regression: the cached DEM came back as (1, rows, cols) and the slope step crashed on the re-run)."""
import numpy as np


def test_cached_dem_is_2d(tmp_path, cfg):
    from floodsight.core import AOI, RunContext, parse_date, write_raster
    from floodsight.fetch.dem import fetch_dem
    from floodsight.preprocess.geometry import slope_deg

    ctx = RunContext(AOI.parse("85.30,27.95,85.33,27.98"), parse_date("2026-08-26"), cfg,
                     runs_dir=tmp_path, name="rerun")
    g = ctx.grid
    z = (np.add.outer(np.arange(g.height), np.arange(g.width)) * 2.0 + 800).astype(np.float32)
    write_raster(ctx.path("inputs", "dem.tif"), z, g, nodata=np.nan)   # as a first run leaves it
    dem = fetch_dem(ctx)                                                  # second run: from cache, no network
    assert dem.shape == g.shape
    assert np.allclose(dem[5:-5, 5:-5], z[5:-5, 5:-5], atol=1.0)
    assert slope_deg(dem, g.res).shape == g.shape


import pytest  # noqa: E402


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")   # the file is made without one
def test_raster_without_map_position_is_refused(tmp_path, cfg):
    """A file with no CRS/geotransform must stop the run instead of being placed at the wrong location."""
    import numpy as np
    import pytest
    import rasterio

    from floodsight.core import AOI, Grid, read_to_grid

    p = tmp_path / "nogeo.tif"
    with rasterio.open(p, "w", driver="GTiff", width=10, height=10, count=1, dtype="float32") as dst:
        dst.write(np.ones((1, 10, 10), np.float32))
    with pytest.raises(ValueError, match="no map position"):
        read_to_grid(p, Grid.for_aoi(AOI.parse("85.30,27.95,85.31,27.96"), 20))


def test_unprojectable_point_is_outside():
    import numpy as np

    from floodsight.core import AOI, Grid
    g = Grid.for_aoi(AOI.parse("85.30,27.95,85.31,27.96"), 20)
    r, c = g.xy_to_rowcol(np.array([np.nan, np.inf, 1.0]), np.array([0.0, 0.0, np.nan]))
    assert (r == -1).all() and (c == -1).all()
