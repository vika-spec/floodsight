"""One-page situation report (PDF) in English and Nepali.

Every number is read from results.json, which the pipeline computed from the maps. Text comes from fixed
templates in both languages; there is no language model anywhere in this path, so no number can be
invented or altered.
"""
from __future__ import annotations

from pathlib import Path

FONTS = Path(__file__).resolve().parent / "fonts"
DEVA_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")

T = {
    "en": {
        "title": "Flood Situation Report",
        "proto": "Educational prototype generated automatically from satellite data. Not an official emergency map.",
        "area": "Area", "date": "Flood date", "generated": "Generated",
        "verdict": {"flood_detected": "Flood / debris impact detected",
                    "possible_flood": "Possible flooding, not confirmed (each change seen by one source only)",
                    "no_significant_change": "No significant flood change detected"},
        "possible": "possible", "confirmed_note": "Numbers are confirmed (two sources agree); '+ possible' = seen by one source only.",
        "tiles": ["Flood water", "Debris / mud", "Buildings likely hit", "Roads affected", "Bridges hit",
                  "Settlements cut off"],
        "km2": "km²", "km": "km",
        "table": "Settlements that lost or changed road access to a hospital",
        "cols": ["#", "Settlement", "Road access now", "Buildings", "Nearest hospital (before)", "Before", "Now"],
        "min": "min", "noroute": "no route",
        "status": {"cut_off": "CUT OFF", "likely_cut_off": "Likely cut off", "unknown": "Unknown (unseen road)",
                   "detour": "Long detour", "reachable": "Unchanged",
                   "isolated_before": "No road even before", "no_road_data": "No mapped road"},
        "none_cut": "No mapped settlement lost its road connection to a hospital.",
        "data": "Data used",
        "s1": "Sentinel-1 radar: before {pre}, after {post} (track {track}, {gap} days apart)",
        "s2_yes": "Sentinel-2 optical: second sensor ({clear} of area cloud-free); confirms radar and adds what only it sees (low confidence)",
        "s2_no": "Sentinel-2 optical: not used (cloud or unavailable)",
        "osm": "OpenStreetMap as of {snap} (before the event)",
        "model_yes": "AI segmentation model: used, combined with rule-based detection",
        "model_no": "AI segmentation model: not used; rule-based detection only",
        "terrain": "Debris class only in mountain terrain (UNEP-WCMC definition); {pct} of this area is mountain",
        "limits": "Confidence and limits",
        "fixed_limit": ("Sentinel satellites revisit an area only every few days: this system maps damage after the "
                        "fact and cannot warn of a sudden glacier or lake outburst. Areas marked grey were hidden "
                        "from the radar and are unknown, not safe. Road access uses mapped roads only. "
                        "'Likely cut off' rests on low-confidence detections (debris counts as confident only when a "
                        "second radar pass or Sentinel-2 confirms it); 'Unknown' means the last route crosses "
                        "roads the radar could not see. Travel times use nominal speeds per road class. "
                        "Rows are ranked by priority: certainty, settlement size (mapped buildings), extra time."),
        "high_conf": "Confirmed: {v} km² of flood water and debris (two sources agree). A further {p} km² is possible only (one source); check before acting on it.",
        "health_csv": "Hospitals and clinics in the area: {n}. The map shows only those that settlements with lost or changed access depend on; full list with locations: dashboard / health_facilities.csv.",
        "notes_hdr": "",
        "attr": "Attributions",
        "access_hdr": "Can we get there? Roads and bridges",
        "roads_hdr": "Blocked or damaged roads (names from the pre-event map)",
        "bridges_hdr": "Bridges likely hit",
        "km_aff": "km affected", "uncertain": "uncertain", "none": "none detected", "more": "more",
        "full_list": "full list: dashboard / priority_list.csv",
        "heli_hdr": "Helicopter access for settlements cut off by road",
        "heli_pad": "mapped helipad", "heli_zone": "flat open ground",
        "heli_none": "none within {km} km",
        "heli_nobody": "No settlement needs air access by this analysis. Mapped helipads in the area: {n}.",
        "heli_note": ("Flat open ground = slope at most {slope}°, at least {diam} m clear, away from detected flood or "
                      "debris and mapped buildings or power lines. Not checked: trees, unmapped wires or buildings, "
                      "soft ground, debris the map missed. Confirm from the air. Distances are straight-line."),
    },
    "ne": {
        "title": "बाढी स्थिति प्रतिवेदन",
        "proto": "भू-उपग्रह तथ्याङ्कबाट स्वचालित रूपमा तयार गरिएको शैक्षिक प्रोटोटाइप। यो आधिकारिक आपतकालीन नक्सा होइन।",
        "area": "क्षेत्र", "date": "बाढी मिति", "generated": "तयार गरिएको",
        "verdict": {"flood_detected": "बाढी / गेग्रानको असर पत्ता लाग्यो",
                    "possible_flood": "सम्भावित बाढी, पुष्टि भएको छैन (प्रत्येक परिवर्तन एउटा स्रोतले मात्र देखेको)",
                    "no_significant_change": "बाढीको उल्लेखनीय परिवर्तन पत्ता लागेन"},
        "possible": "सम्भावित", "confirmed_note": "अङ्कहरू पुष्टि भएका हुन् (दुई स्रोत मिल्छन्); '+ सम्भावित' = एउटा स्रोतले मात्र देखेको।",
        "tiles": ["डुबान क्षेत्र", "गेग्रान / हिलो", "प्रभावित हुनसक्ने भवन", "प्रभावित सडक", "प्रभावित पुल",
                  "सम्पर्कविहीन बस्ती"],
        "km2": "वर्ग कि.मी.", "km": "कि.मी.",
        "table": "अस्पतालसम्मको सडक सम्पर्क गुमाएका वा परिवर्तन भएका बस्तीहरू",
        "cols": ["#", "बस्ती", "अहिलेको सडक सम्पर्क", "भवन", "नजिकको अस्पताल (पहिले)", "पहिले", "अहिले"],
        "min": "मिनेट", "noroute": "बाटो छैन",
        "status": {"cut_off": "सम्पर्क टुटेको", "likely_cut_off": "सम्पर्क टुटेको हुनसक्ने",
                   "unknown": "अज्ञात (रडारले नदेखेको सडक)", "detour": "लामो घुमाउरो बाटो", "reachable": "यथावत्",
                   "isolated_before": "पहिले पनि सडक थिएन", "no_road_data": "सडक नक्सामा छैन"},
        "none_cut": "नक्सामा भएका कुनै पनि बस्तीको अस्पतालसम्मको सडक सम्पर्क टुटेको पाइएन।",
        "data": "प्रयोग गरिएको तथ्याङ्क",
        "s1": "Sentinel-1 रडार: पहिले {pre}, पछि {post} (ट्र्याक {track}, {gap} दिनको फरक)",
        "s2_yes": "Sentinel-2 अप्टिकल: दोस्रो सेन्सर (क्षेत्रको {clear} भाग बादलरहित); रडारलाई पुष्टि गर्छ र आफूले मात्र देखेको थप्छ (कम विश्वसनीयता)",
        "s2_no": "Sentinel-2 अप्टिकल: प्रयोग भएन (बादल वा तथ्याङ्क उपलब्ध नभएकाले)",
        "osm": "OpenStreetMap: {snap} को अवस्था (घटनाअघि)",
        "model_yes": "AI सेग्मेन्टेसन मोडेल: प्रयोग गरियो, नियम-आधारित पहिचानसँग मिलाइयो",
        "model_no": "AI सेग्मेन्टेसन मोडेल: प्रयोग भएन; नियम-आधारित पहिचान मात्र",
        "terrain": "गेग्रान वर्ग पहाडी भूभागमा मात्र (UNEP-WCMC परिभाषा); यस क्षेत्रको {pct} भाग पहाडी हो",
        "limits": "विश्वसनीयता र सीमाहरू",
        "fixed_limit": ("Sentinel भू-उपग्रहले एउटै क्षेत्र केही दिनको अन्तरमा मात्र हेर्छ: यो प्रणालीले क्षति भइसकेपछि "
                        "नक्सा बनाउँछ, हिमनदी वा हिमताल विस्फोटको पूर्वसूचना दिन सक्दैन। खैरो रङका क्षेत्र रडारले "
                        "देख्न सकेन; ती अज्ञात हुन्, सुरक्षित होइनन्। सडक सम्पर्क नक्सामा भएका सडकमा मात्र आधारित छ। "
                        "'सम्पर्क टुटेको हुनसक्ने' कम विश्वसनीय पहिचानमा आधारित छ (गेग्रानलाई दोस्रो रडार अवलोकन वा Sentinel-2 ले "
                        "पुष्टि गरेमा मात्र विश्वसनीय मानिन्छ); 'अज्ञात' भन्नाले अन्तिम बाटो रडारले "
                        "नदेखेको सडकबाट जान्छ। यात्रा समय सडक प्रकारअनुसारको सामान्य गतिमा आधारित छ।"),
        "high_conf": "पुष्टि भएको: {v} वर्ग कि.मी. डुबान र गेग्रान (दुई स्रोत मिल्छन्)। थप {p} वर्ग कि.मी. सम्भावित मात्र हो (एउटा स्रोत); कारबाहीअघि जाँच गर्नुहोस्।",
        "health_csv": "क्षेत्रमा अस्पताल र स्वास्थ्य संस्था: {n}। नक्सामा सम्पर्क गुमाएका वा परिवर्तन भएका बस्तीहरू निर्भर रहेका मात्र देखाइएका छन्; स्थानसहित पूरा सूची: ड्यासबोर्ड / health_facilities.csv।",
        "notes_hdr": "प्रणालीका टिप्पणी (अंग्रेजीमा): ",
        "attr": "स्रोत उल्लेख",
        "access_hdr": "के त्यहाँ पुग्न सकिन्छ? सडक र पुल",
        "roads_hdr": "अवरुद्ध वा क्षतिग्रस्त सडक (घटनाअघिको नक्साअनुसार नाम)",
        "bridges_hdr": "प्रभावित हुनसक्ने पुल",
        "km_aff": "कि.मी. प्रभावित", "uncertain": "अनिश्चित", "none": "पत्ता लागेन", "more": "थप",
        "full_list": "पूरा सूची: ड्यासबोर्ड / priority_list.csv",
        "heli_hdr": "सडक सम्पर्क टुटेका बस्तीका लागि हेलिकप्टर पहुँच",
        "heli_pad": "नक्सामा भएको हेलिप्याड", "heli_zone": "समतल खुला जमिन",
        "heli_none": "{km} कि.मी. भित्र छैन",
        "heli_nobody": "यस विश्लेषणअनुसार कुनै बस्तीलाई हवाई पहुँच आवश्यक छैन। क्षेत्रमा नक्सामा भएका हेलिप्याड: {n}।",
        "heli_note": ("समतल खुला जमिन = भिरालोपन {slope}° भन्दा कम, कम्तीमा {diam} मिटर खाली, पत्ता लागेको बाढी वा "
                      "गेग्रान र नक्सामा भएका भवन वा बिजुलीका तारबाट टाढा। जाँच नगरिएको: रूख, नक्सामा नभएका तार वा "
                      "भवन, नरम जमिन, नक्साले छुटाएको गेग्रान। हावाबाटै पुष्टि गर्नुहोस्। दूरी सिधा रेखामा।"),
    },
}


RAG = {"cut_off": (200, 30, 30), "likely_cut_off": (228, 87, 63), "unknown": (122, 111, 168),
       "detour": (208, 140, 10), "reachable": (46, 139, 58)}


def _num(v, lang, nd=1):
    if v is None:
        return "–"
    s = f"{v:,.{nd}f}" if isinstance(v, float) and nd else f"{int(round(v)):,}"
    return s.translate(DEVA_DIGITS) if lang == "ne" else s


def _pdf(lang):
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(False)
    # One merged font (Noto Sans + Noto Sans Devanagari, SIL OFL) covers both scripts, so mixed
    # Nepali/English lines shape correctly without font switching.
    pdf.add_font("sans", "", str(FONTS / "FloodSightSans-Regular.ttf"))
    pdf.add_font("sans", "B", str(FONTS / "FloodSightSans-Bold.ttf"))
    try:
        pdf.set_text_shaping(True)       # correct Devanagari conjuncts (needs uharfbuzz)
    except Exception:  # noqa: BLE001
        pass
    main = "sans"
    pdf.add_page()
    return pdf, main


# Tighter layouts tried in order until everything fits above the footer: map height (mm), settlement rows,
# items per road/bridge/helicopter list, system notes. The report is one page by design (the brief asks for a
# one-page situation report); longer lists are in the dashboard and vectors/priority_list.csv.
LAYOUTS = [dict(map_h=86, rows=7, items=5, notes=3), dict(map_h=72, rows=6, items=4, notes=2),
           dict(map_h=58, rows=5, items=3, notes=1), dict(map_h=46, rows=4, items=3, notes=0),
           dict(map_h=36, rows=3, items=2, notes=0)]
FOOTER_Y = 279.0


def build_pdf(results: dict, run_dir: Path, lang: str, map_png: Path) -> Path:
    for lay in LAYOUTS:
        pdf, bottom = _render(results, lang, map_png, lay)
        if bottom <= FOOTER_Y:
            break
    out = run_dir / "report" / f"situation_report_{lang}.pdf"
    pdf.output(str(out))
    return out


def _render(results: dict, lang: str, map_png: Path, lay: dict):
    """Draw the page with one layout; returns (pdf, lowest y used by the content above the footer)."""
    t = T[lang]
    pdf, F = _pdf(lang)
    W = 210
    M = 10
    ink, muted = (25, 30, 40), (100, 105, 115)

    # --- header band ---
    verdict = results["verdict"]
    band = {"flood_detected": (160, 30, 30), "possible_flood": (176, 110, 10)}.get(verdict, (40, 110, 60))
    pdf.set_fill_color(*band)
    pdf.rect(0, 0, W, 24, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(M, 4)
    pdf.set_font(F, "B", 17)
    pdf.cell(0, 8, t["title"])
    pdf.set_xy(M, 13)
    pdf.set_font(F, "", 9)
    lon, lat = results["aoi"]["center"]
    pdf.cell(0, 5, f"{t['area']}: {lat:.3f}°N {lon:.3f}°E ({_num(results['aoi']['area_km2'], lang)} {t['km2']})   "
                   f"{t['date']}: {results['flood_date']}   {t['generated']}: {results['generated_utc']}")
    pdf.set_xy(M, 18)
    pdf.set_font(F, "B", 9)
    pdf.cell(0, 5, t["verdict"][verdict])
    pdf.set_text_color(*muted)
    pdf.set_xy(M, 25.5)
    pdf.set_font(F, "", 7)
    pdf.cell(0, 4, t["proto"])

    # --- key numbers ---
    a, d, acc = results["areas"], results.get("damage", {}), results.get("access", {})
    fc, dc = a.get("flood_high_conf_km2", a.get("flood_km2", 0)), a.get("debris_high_conf_km2", a.get("debris_km2", 0))
    rc = d.get("road_km_confirmed", d.get("road_km_affected", 0))
    bc = d.get("bridges_confirmed", d.get("bridges_hit", 0))
    vals = [f"{_num(fc, lang, 2)} {t['km2']}", f"{_num(dc, lang, 2)} {t['km2']}",
            _num(d.get("buildings_likely_hit", 0), lang, 0), f"{_num(rc, lang)} {t['km']}",
            _num(bc, lang, 0), _num(acc.get("settlements_cut_off", 0), lang, 0)]
    extra = [max(a.get("flood_km2", 0) - fc, 0), max(a.get("debris_km2", 0) - dc, 0), d.get("buildings_possibly_hit", 0),
             max(d.get("road_km_affected", 0) - rc, 0), max(d.get("bridges_hit", 0) - bc, 0),
             acc.get("settlements_likely_cut_off", 0)]
    extra_txt = [f"+ {_num(extra[0], lang, 2)} {t['possible']}", f"+ {_num(extra[1], lang, 2)} {t['possible']}",
                 f"+ {_num(extra[2], lang, 0)} {t['possible']}", f"+ {_num(extra[3], lang)} {t['km']} {t['possible']}",
                 f"+ {_num(extra[4], lang, 0)} {t['possible']}", f"+ {_num(extra[5], lang, 0)} {t['possible']}"]
    tw = (W - 2 * M - 5 * 2) / 6
    y = 31
    for i, (lab, v) in enumerate(zip(t["tiles"], vals)):
        x = M + i * (tw + 2)
        crit = i == 5 and acc.get("settlements_cut_off", 0)
        pdf.set_fill_color(*((253, 236, 236) if crit else (242, 244, 247)))
        pdf.rect(x, y, tw, 16, "F")
        pdf.set_text_color(*((170, 30, 30) if crit else ink))
        pdf.set_xy(x, y + 2)
        pdf.set_font(F, "B", 12)
        pdf.cell(tw, 6, v, align="C")
        pdf.set_xy(x, y + 8.4)
        pdf.set_font(F, "", 7)
        pdf.set_text_color(*muted)
        pdf.cell(tw, 4, lab, align="C")
        if extra[i] and extra[i] >= 0.005:
            pdf.set_xy(x, y + 12)
            pdf.set_font(F, "", 6.2)
            pdf.cell(tw, 3.4, extra_txt[i], align="C")

    # --- map ---
    from PIL import Image
    im = Image.open(map_png)
    mw = W - 2 * M
    mh = min(mw * im.height / im.width, lay["map_h"])
    mw_eff = mh * im.width / im.height
    pdf.image(str(map_png), x=M + (mw - mw_eff) / 2, y=50, w=mw_eff, h=mh)
    y = 50 + mh + 3

    # --- table ---
    pdf.set_text_color(*ink)
    pdf.set_xy(M, y)
    pdf.set_font(F, "B", 10)
    pdf.cell(0, 5, t["table"])
    y += 6
    rows = [r for r in results.get("settlements", [])
            if r["status"] in ("cut_off", "likely_cut_off", "unknown", "detour")]
    rows.sort(key=lambda r: r.get("priority") or 10 ** 6)
    widths = [7, 40, 36, 16, 47, 22, 22]
    if not rows:
        pdf.set_font(F, "", 8.5)
        pdf.set_xy(M, y)
        pdf.cell(0, 5, t["none_cut"])
        y += 7
    else:
        pdf.set_fill_color(230, 233, 238)
        pdf.set_font(F, "B", 7.5)
        x = M
        for wdt, c in zip(widths, t["cols"]):
            pdf.set_xy(x, y)
            pdf.cell(wdt, 5, c, fill=True)
            x += wdt
        y += 5
        pdf.set_font(F, "", 7.5)
        max_rows = lay["rows"]
        for r in rows[:max_rows]:
            x = M
            cut = r["status"] in ("cut_off", "likely_cut_off")

            def trip(km, mn):
                if km is None:
                    return t["noroute"]
                return f"{_num(km, lang)} {t['km']}" + (f", {_num(mn, lang, 0)} {t['min']}" if mn is not None else "")
            cells = [_num(r.get("priority"), lang, 0) if r.get("priority") else "", str(r.get("name", ""))[:26],
                     t["status"].get(r["status"], r["status"]), _num(r.get("buildings"), lang, 0),
                     str(r.get("target_before") or "–")[:30], trip(r.get("km_before"), r.get("min_before")),
                     trip(r.get("km_after"), r.get("min_after"))]
            for j, (wdt, c) in enumerate(zip(widths, cells)):
                pdf.set_xy(x, y)
                if j == 2:                       # red / amber / purple / green marker before the status
                    pdf.set_fill_color(*RAG.get(r["status"], (150, 150, 150)))
                    pdf.rect(x, y + 1.2, 2.4, 2.4, "F")
                    pdf.set_xy(x + 3.4, y)
                    pdf.set_text_color(*((180, 25, 25) if cut else ink))
                    pdf.cell(wdt - 3.4, 4.6, c)
                else:
                    pdf.set_text_color(*ink)
                    pdf.cell(wdt, 4.6, c)
                x += wdt
            y += 4.6
        if len(rows) > max_rows:
            pdf.set_xy(M, y)
            pdf.set_text_color(*muted)
            pdf.cell(0, 4.6, f"+ {_num(len(rows) - max_rows, lang, 0)} {t['more']} ({t['full_list']})")
            y += 4.6
        y += 2

    # --- can we get there: named roads and bridges (two columns) ---
    y = _access_block(pdf, F, t, results, lang, M, W, y, ink, muted, lay["items"])
    # --- helicopter access for settlements cut off by road ---
    y = _heli_block(pdf, F, t, results, lang, M, W, y, ink, muted, lay["items"])

    # --- data used + limits (two columns) ---
    inp = results["inputs"]
    p = inp["sentinel1_pairs"][0]
    lines = [t["s1"].format(pre=p["pre_date"], post=p["post_date"], track=p["track"], gap=_num(p["gap_days"], lang, 0))]
    for extra in inp["sentinel1_pairs"][1:]:
        lines.append(t["s1"].format(pre=extra["pre_date"], post=extra["post_date"], track=extra["track"],
                                    gap=_num(extra["gap_days"], lang, 0)))
    s2 = inp.get("sentinel2", {})
    lines.append(t["s2_yes"].format(clear=f"{s2.get('clear_fraction', 0):.0%}") if s2.get("used") else t["s2_no"])
    lines.append(t["osm"].format(snap=inp.get("osm_snapshot")))
    lines.append(t["model_yes"] if inp.get("model", {}).get("used") else t["model_no"])
    dt = inp.get("debris_terrain") or {}
    if dt.get("applied"):
        lines.append(t["terrain"].format(pct=f"{dt.get('mountain_fraction', 0):.0%}"))

    colw = (W - 2 * M - 6) / 2
    top = y + 3
    pdf.set_xy(M, top)
    pdf.set_font(F, "B", 9)
    pdf.set_text_color(*ink)
    pdf.cell(colw, 5, t["data"])
    pdf.set_font(F, "", 7.2)
    pdf.set_xy(M, top + 5.5)
    pdf.multi_cell(colw, 3.6, "\n".join("• " + l for l in lines))
    bottom = pdf.get_y()

    hc = round(a.get("flood_high_conf_km2", 0) + a.get("debris_high_conf_km2", 0), 2)
    poss = round(max(a.get("flood_km2", 0) + a.get("debris_km2", 0) - hc, 0), 2)
    lim = [t["high_conf"].format(v=_num(hc, lang, 2), p=_num(poss, lang, 2)), t["fixed_limit"]]
    n_h = results.get("health_facilities_total")
    if n_h:
        lim.insert(1, t["health_csv"].format(n=_num(n_h, lang, 0)))
    notes = [n for n in results.get("notes", []) if not n.startswith("OpenStreetMap snapshot")][:lay["notes"]]
    if notes:
        lim.append(t["notes_hdr"] + " ".join(notes))
    x2 = M + colw + 6
    pdf.set_xy(x2, top)
    pdf.set_font(F, "B", 9)
    pdf.cell(colw, 5, t["limits"])
    pdf.set_font(F, "", 7.2)
    pdf.set_xy(x2, top + 5.5)
    pdf.multi_cell(colw, 3.6, "\n".join("• " + l for l in lim))
    bottom = max(bottom, pdf.get_y())

    # --- footer: attributions ---
    pdf.set_draw_color(210, 212, 216)
    pdf.line(M, 280, W - M, 280)
    pdf.set_xy(M, 281)
    pdf.set_font("sans", "", 5.6)
    pdf.set_text_color(*muted)
    pdf.multi_cell(W - 2 * M, 2.6, "  ".join(results.get("attributions", [])) + f"  Run {results['run_id']}.")

    return pdf, bottom


def _access_block(pdf, F, t, results, lang, M, W, y, ink, muted, n=5):
    bl = results.get("blockages") or {}
    colw = (W - 2 * M - 6) / 2
    pdf.set_text_color(*ink)
    pdf.set_xy(M, y)
    pdf.set_font(F, "B", 10)
    pdf.cell(0, 5, t["access_hdr"])
    y += 5.5

    def ll(v):
        return f"{abs(v[0]):.3f}°{'N' if v[0] >= 0 else 'S'} {abs(v[1]):.3f}°{'E' if v[1] >= 0 else 'W'}"
    roads = [f"{r['name'][:34]}: {_num(r['km_affected'], lang, 2)} {t['km_aff']}"
             + ("" if r.get("certain") else f" ({t['uncertain']})") + f", {ll(r['lat_lon'])}"
             for r in bl.get("roads", [])[:n]]
    if bl.get("roads_total", 0) > len(roads):
        roads.append(f"+ {_num(bl['roads_total'] - len(roads), lang, 0)} {t['more']}")
    bridges = [f"{b['name'][:38]}, {ll(b['lat_lon'])}" for b in bl.get("bridges", [])[:n]]
    if bl.get("bridges_total", 0) > len(bridges):
        bridges.append(f"+ {_num(bl['bridges_total'] - len(bridges), lang, 0)} {t['more']}")
    hh = 0
    for i, (hdr, items) in enumerate(((t["roads_hdr"], roads), (t["bridges_hdr"], bridges))):
        x = M + i * (colw + 6)
        pdf.set_xy(x, y)
        pdf.set_font(F, "B", 7.8)
        pdf.set_text_color(*ink)
        pdf.cell(colw, 4, hdr)
        pdf.set_font(F, "", 7.2)
        pdf.set_text_color(*(ink if items else muted))
        pdf.set_xy(x, y + 4.4)
        pdf.multi_cell(colw, 3.5, "\n".join("• " + l for l in items) if items else t["none"])
        hh = max(hh, pdf.get_y() - y)
    return y + hh + 2


def _heli_block(pdf, F, t, results, lang, M, W, y, ink, muted, n=5):
    h = results.get("helicopter_access")
    if not h:
        return y
    crit = h["criteria"]
    pdf.set_text_color(*ink)
    pdf.set_xy(M, y)
    pdf.set_font(F, "B", 10)
    pdf.cell(0, 5, t["heli_hdr"])
    y += 5.5

    def ll(v):
        return f"{abs(v[0]):.3f}°{'N' if v[0] >= 0 else 'S'} {abs(v[1]):.3f}°{'E' if v[1] >= 0 else 'W'}"
    none = t["heli_none"].format(km=_num(crit["max_distance_km"], lang, 0))
    lines = []
    per = h.get("per_settlement", [])
    for s in per[:n]:
        pad = (f"{t['heli_pad']} {s['helipad']['name'][:24]} {_num(s['helipad']['km'], lang)} {t['km']}"
               if s.get("helipad") else f"{t['heli_pad']}: {none}")
        zone = (f"{t['heli_zone']} {_num(s['zone']['km'], lang)} {t['km']} ({ll(s['zone']['lat_lon'])})"
                if s.get("zone") else f"{t['heli_zone']}: {none}")
        lines.append(f"{str(s.get('name') or '')[:24]} [{t['status'].get(s['status'], s['status'])}]: {pad}; {zone}")
    if len(per) > n:
        lines.append(f"+ {_num(len(per) - n, lang, 0)} {t['more']} ({t['full_list']})")
    if not lines:
        lines.append(t["heli_nobody"].format(n=_num(len(h.get("helipads", [])), lang, 0)))
    pdf.set_font(F, "", 7.2)
    pdf.set_xy(M, y)
    pdf.multi_cell(W - 2 * M, 3.6, "\n".join("• " + l for l in lines))
    y = pdf.get_y() + 0.6
    pdf.set_xy(M, y)
    pdf.set_font(F, "", 6.4)
    pdf.set_text_color(*muted)
    pdf.multi_cell(W - 2 * M, 3.0, t["heli_note"].format(slope=_num(crit["max_slope_deg"], lang, 0),
                                                         diam=_num(crit["clear_diameter_m"], lang, 0)))
    return pdf.get_y() + 1.5


# Nepal's border, coarse (Natural Earth 1:10m admin-0 v5.1.2, public domain, simplified to 0.03 degrees, about 3 km).
# Only used to decide whether the Nepali report is also made; widened by 0.03 degrees so areas on the border
# get it too (an extra Nepali report is harmless, a missing one is not).
_NEPAL_LONLAT = (
    (87.83, 27.91), (88.17, 27.83), (88.02, 27.49), (88.04, 27.32), (87.97, 27.11), (88.1, 26.96), (88.17,
    26.76), (88.04, 26.41), (87.98, 26.37), (87.87, 26.46), (87.59, 26.38), (87.42, 26.43), (87.3, 26.35),
    (87.22, 26.41), (87.11, 26.4), (87.04, 26.58), (86.71, 26.41), (86.28, 26.61), (86.2, 26.58), (86.01,
    26.65), (85.83, 26.57), (85.71, 26.65), (85.7, 26.8), (85.6, 26.85), (85.3, 26.74), (85.19, 26.76),
    (85.16, 26.85), (85.02, 26.85), (84.82, 27.01), (84.63, 27.04), (84.66, 27.2), (84.61, 27.31), (84.29,
    27.38), (84.12, 27.51), (84.01, 27.44), (83.83, 27.43), (83.85, 27.34), (83.59, 27.46), (83.39, 27.47),
    (83.36, 27.39), (83.28, 27.33), (83.13, 27.44), (82.75, 27.5), (82.68, 27.69), (82.4, 27.68), (82.03,
    27.91), (81.86, 27.85), (81.47, 28.07), (81.4, 28.16), (81.3, 28.13), (81.17, 28.36), (80.91, 28.44),
    (80.56, 28.68), (80.49, 28.66), (80.49, 28.56), (80.33, 28.63), (80.04, 28.84), (80.03, 28.9), (80.13,
    29.11), (80.22, 29.13), (80.22, 29.21), (80.26, 29.2), (80.28, 29.31), (80.21, 29.42), (80.39, 29.6),
    (80.37, 29.76), (80.48, 29.81), (80.57, 29.95), (80.72, 30.01), (80.88, 30.21), (81.01, 30.19), (81.1,
    30.02), (81.22, 30.01), (81.27, 30.13), (81.35, 30.16), (81.39, 30.37), (81.51, 30.33), (81.61, 30.42),
    (82.09, 30.33), (82.08, 30.23), (82.16, 30.18), (82.15, 30.06), (82.54, 29.92), (82.84, 29.66), (82.98,
    29.66), (83.06, 29.58), (83.17, 29.62), (83.26, 29.54), (83.26, 29.47), (83.36, 29.46), (83.52, 29.18),
    (83.64, 29.15), (83.7, 29.23), (83.91, 29.29), (84.09, 29.26), (84.16, 29.19), (84.13, 29.14), (84.18,
    29.06), (84.23, 29.03), (84.21, 28.94), (84.38, 28.85), (84.45, 28.73), (84.62, 28.69), (84.68, 28.65),
    (84.68, 28.6), (84.81, 28.54), (84.93, 28.54), (85.05, 28.64), (85.16, 28.59), (85.08, 28.44), (85.11,
    28.29), (85.36, 28.26), (85.43, 28.3), (85.61, 28.25), (85.69, 28.34), (85.73, 28.22), (85.86, 28.14),
    (85.96, 27.89), (86.11, 27.93), (86.07, 28.08), (86.16, 28.16), (86.2, 28.0), (86.43, 27.91), (86.51,
    27.95), (86.55, 28.1), (86.66, 28.11), (86.73, 28.08), (86.74, 28.02), (87.01, 27.95), (87.16, 27.83),
    (87.36, 27.86), (87.37, 27.8), (87.53, 27.84), (87.7, 27.81), (87.83, 27.91)
)
_NEPAL = None


def touches_nepal(bbox) -> bool:
    """True if the analysed box overlaps Nepal (or lies within about 3 km of its border)."""
    global _NEPAL
    from shapely.geometry import Polygon, box
    if _NEPAL is None:
        _NEPAL = Polygon(_NEPAL_LONLAT).buffer(0.03)
    w, s_, e, n = bbox
    return _NEPAL.intersects(box(w, s_, e, n))


def report_languages(results: dict) -> list[str]:
    """English always; Nepali only when the area is in (or touches) Nepal, where responders read it."""
    try:
        return ["en", "ne"] if touches_nepal(results["aoi"]["bbox"]) else ["en"]
    except Exception:  # noqa: BLE001 - never lose the English report over this check
        return ["en"]


def build_reports(results: dict, run_dir: Path) -> dict:
    from .mapfig import render_map

    run_dir = Path(run_dir)
    map_png = render_map(run_dir, results, run_dir / "report" / "map.png")
    out = {}
    for lang in report_languages(results):
        out[lang] = str(build_pdf(results, run_dir, lang, map_png).relative_to(run_dir))
    out["map"] = str(map_png.relative_to(run_dir))
    return out
