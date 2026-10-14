// One vocabulary for the map, the legend and the panels: the same colour always means the same thing.
export const C = {
  water: "#1E6EE6",
  debris: "#965F28",
  permanent: "#78AADC",
  blind: "#8A8F94",
  cut: "#C62A1E",
  detour: "#D08C0A",
  ok: "#2E8B3A",
  none: "#9AA3A9",
  building: "#B3261E",
  buildingMaybe: "#E39B21",
  road: "#E8701A",
  bridge: "#17242E",
  flow: "#6D3FD1",
  health: "#C62A1E",
  aoi: "#17242E",
};

export const STATUS = {
  cut_off: { label: "Cut off", long: "No road to a hospital any more (blocked by confident detections)", color: C.cut },
  likely_cut_off: { label: "Likely cut off", long: "Only low-confidence detections block the last route: check on the ground", color: "#E4573F" },
  unknown: { label: "Unknown", long: "The only remaining route crosses roads the radar could not see", color: "#7A6FA8" },
  detour: { label: "Long detour", long: "Still connected, but the route is much longer", color: C.detour },
  reachable: { label: "Road OK", long: "Same hospital, similar distance", color: C.ok },
  isolated_before: { label: "No road before", long: "Had no mapped road link even before the flood", color: C.none },
  no_road_data: { label: "No mapped road", long: "No road mapped near the settlement", color: "#C3C9CD" },
};

export const STAGES = [
  ["Searching the Sentinel-1 catalogue", "Find radar images before and after"],
  ["Downloading satellite, elevation and map data", "Download radar, elevation and OpenStreetMap"],
  ["Terrain analysis (slope, drainage, HAND)", "Read the terrain: slope, drainage, height above river"],
  ["Detecting flood water and debris", "Detect new water and debris"],
  ["Segmentation model (Siamese U-Net)", "Check with the AI segmentation model"],
  ["Adding Sentinel-2 optical evidence", "Add clear-sky optical evidence"],
  ["Assessing buildings, roads and bridges", "Find damaged buildings, roads and bridges"],
  ["Finding cut-off settlements", "Find settlements cut off from hospitals"],
  ["Helicopter landing options", "Check helipads and open ground near cut-off places"],
  ["Writing maps and results", "Write maps and results"],
  ["Generating situation reports", "Write the situation report (English, Nepali)"],
];
