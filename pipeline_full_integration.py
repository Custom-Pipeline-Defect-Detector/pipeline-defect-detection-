import os
import csv
import time
import math
import glob
from typing import List, Dict, Any, Tuple

# ---------------------- CONFIG ----------------------
MODEL_PATH = r"D:/train126/weights/best.pt"
IMAGES_DIR = r"D:/yolov8/data/heat"
RESULTS_DIR = r"D:/pipeline_detection_system"
INTERACTIVE = True  # per-image enter/skip/rerun/quit

# Drools / JVM config
JVM_DLL = r"D:/pipeline_detection_system/jdk1.8.0_351/jre_1/bin/server/jvm.dll"
DROOLS_MAIN_JAR = r"D:/pipeline_detection_system/jar/drools_reasoning.jar"
DROOLS_DEPS_DIR = r"D:/pipeline_detection_system/jar/drools_reasoning_jar"

# Where to drop the temporary defect Excels per image & defect
DEFECT_XLS_DIR = os.path.join(RESULTS_DIR, "defect_excel_single")
os.makedirs(DEFECT_XLS_DIR, exist_ok=True)

# Final summary CSV
FINAL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "results_summary.csv")

# YOLOv8 settings
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# Map your YOLO class names -> pinyin codes used by rules
# Deformation -> BX(变形), Deposition -> CJ(沉积), Obstacle -> ZAW(杂物), Rupture -> PL(破裂)
CLASS_TO_CODE = {
    "Deformation": "BX",
    "Deposition": "CJ",
    "Obstacle":   "ZAW",
    "Rupture":    "PL",
}
# numeric class ordering fallback (if the model has no .names)
FALLBACK_NAMES = ["Deformation", "Deposition", "Obstacle", "Rupture"]

# Severity mapping by grade (from reasoning only)
GRADE_TO_SEVERITY = {1: "Minor", 2: "Moderate", 3: "Severe", 4: "Critical"}
DESC_EN = {
    1: "Grade I — localized defect with limited structural impact.",
    2: "Grade II — defect affects structure; monitor and plan maintenance.",
    3: "Grade III — defect is serious; structure likely to fail in short term.",
    4: "Grade IV — imminent failure risk; immediate intervention required.",
}
ADVICE_EN = {
    1: "Minor repair when convenient; continue routine inspection.",
    2: "Schedule repair; increase inspection frequency.",
    3: "Repair as soon as possible to prevent failure.",
    4: "Take out of service and repair immediately.",
}
CN_TO_EN = {
    "等级Ⅰ，管段缺陷轻微，结构状况基本不受影响": "Grade I: minor defect; structure basically unaffected.",
    "等级Ⅱ，管段缺陷一般，结构状况有所影响": "Grade II: moderate defect; structure affected.",
    "等级Ⅲ，管段缺陷严重，结构状况受到影响": "Grade III: severe defect; structure affected.",
    "等级Ⅳ，管段缺陷极严重，存在破坏风险": "Grade IV: critical defect; high risk of failure.",
    "结构在短期内可能会发生破坏，应尽快修复": "Likely short-term failure; repair as soon as possible.",
    "需立即停止使用并进行加固或更换": "Immediately stop operation; reinforce or replace.",
}
# ----------------------------------------------------


# ---------------- YOLOv8 (Ultralytics) ----------------
def load_yolov8_model(model_path: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from e
    print(f"Loaded YOLOv8 model: {model_path}")
    return YOLO(model_path)


def run_yolov8_on_image(model, image_path: str, conf=0.25, imgsz=640):
    res = model.predict(source=image_path, conf=conf, imgsz=imgsz, verbose=False)
    return res


def extract_detections(result, image_path: str) -> List[Dict[str, Any]]:
    """
    Convert YOLOv8 result to a list of dicts with absolute xywh, confidence, class name.
    """
    dets = []
    names = getattr(result, "names", None) or getattr(result, "names", FALLBACK_NAMES)
    # result.boxes.xywh is tensor [xc,yc,w,h] in absolute pixels
    # result.orig_shape is (h, w)
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xywh = boxes.xywh.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    h, w = result.orig_shape
    for i in range(xywh.shape[0]):
        xc, yc, bw, bh = [float(v) for v in xywh[i]]
        conf = float(confs[i])
        cls_id = int(clss[i])
        cls_name = names[cls_id] if isinstance(names, dict) else (names[cls_id] if cls_id < len(names) else f"class_{cls_id}")
        dets.append({
            "image": image_path,
            "class_id": cls_id,
            "class_name": cls_name,
            "code": CLASS_TO_CODE.get(cls_name, "UNK"),
            "xc": xc, "yc": yc, "w": bw, "h": bh,
            "img_w": float(w), "img_h": float(h),
            "conf": conf
        })
    return dets
# ------------------------------------------------------


# -------------- Fallback Excel writer -----------------
# We use xlwt to create simple sheets per defect code. The columns are derived
# from the data that was visible in your logs (the Drools rules used fields like
# ruptureArea, Arclength for PL; and proportions for BX/ZAW/CJ).
#
# If one of your helper modules (excel_export_job.py / excel_export_job_zhx.py / excel_export_job_qsp.py)
# is importable and exposes the expected write_*_excel functions, we use that first.
def try_import_helpers():
    helpers = {}
    try:
        import excel_export_job as ee0
        helpers["excel_export_job"] = ee0
    except Exception:
        pass
    try:
        import excel_export_job_zhx as ee1
        helpers["excel_export_job_zhx"] = ee1
    except Exception:
        pass
    try:
        import excel_export_job_qsp as ee2
        helpers["excel_export_job_qsp"] = ee2
    except Exception:
        pass
    return helpers


HELPERS = try_import_helpers()

def _sheet_for_code(code: str) -> str:
    # Your jar expects sheet names like "PL", "BX", "ZAW", "CJ" etc.
    return code.upper()

def _safe_xlwt():
    try:
        import xlwt  # .xls
        return xlwt
    except Exception as e:
        raise RuntimeError("xlwt is required for the fallback writer (pip install xlwt).") from e

def _bbox_area_ratio(det: Dict[str, Any]) -> float:
    return max(0.0, (det["w"] * det["h"]) / (det["img_w"] * det["img_h"] + 1e-9))

def _approx_arclength(det: Dict[str, Any]) -> float:
    """
    A proxy used earlier in your working logs:
    scale arclength ~ bbox width ratio * 2*pi*R. Without a physical radius,
    we approximate in pixels: 2*pi*(img_w/2) * (w/img_w) = pi * w.
    """
    return math.pi * det["w"]

def _rupture_area(det: Dict[str, Any]) -> float:
    # Keep it consistent with your earlier success:
    return (det["w"] * det["h"]) / 100.0

def _write_xls_rows(book, sheet_name: str, header: List[str], rows: List[List[Any]]):
    xlwt = _safe_xlwt()
    sh = book.add_sheet(sheet_name)
    for j, head in enumerate(header):
        sh.write(0, j, head)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            sh.write(i, j, val)

def write_defect_excel_fallback(code: str, img_stem: str, dets: List[Dict[str, Any]]) -> str:
    """
    Build a per-defect Excel file your Drools rule can read.
    For each defect code we write a minimal set of features + uri.
    """
    xlwt = _safe_xlwt()
    book = xlwt.Workbook()

    sheet = _sheet_for_code(code)
    rows = []
    if code == "PL":  # rupture: rules used ruptureArea and Arclength
        # Columns tuned to what Drools used in your logs
        header = ["uri", "ruptureArea", "Arclength"]
        for det in dets:
            rows.append([f"{img_stem}.img", _rupture_area(det), _approx_arclength(det)])
    elif code in ("BX", "ZAW", "CJ"):
        # deformation / obstacle / deposition: use ratios commonly used by your rules
        header = ["uri", "bboxAreaRatio", "imgWidth", "imgHeight", "conf"]
        for det in dets:
            rows.append([f"{img_stem}.img", _bbox_area_ratio(det), det["img_w"], det["img_h"], det["conf"]])
    else:
        # generic
        header = ["uri", "bboxAreaRatio", "imgWidth", "imgHeight", "conf"]
        for det in dets:
            rows.append([f"{img_stem}.img", _bbox_area_ratio(det), det["img_w"], det["img_h"], det["conf"]])

    _write_xls_rows(book, sheet, header, rows)
    out_path = os.path.join(DEFECT_XLS_DIR, f"{code}_{img_stem}.xls")
    book.save(out_path)
    return out_path

def write_defect_excel(code: str, img_stem: str, dets: List[Dict[str, Any]]) -> str:
    """
    Try helper modules first; otherwise fallback.
    """
    # Prefer your known helpers if they expose write_*_excel functions
    try:
        if code == "PL":
            for mod in ("excel_export_job", "excel_export_job_zhx", "excel_export_job_qsp"):
                m = HELPERS.get(mod)
                if m and hasattr(m, "write_PL_excel"):
                    return m.write_PL_excel(dets, DEFECT_XLS_DIR, img_stem)  # type: ignore
        if code == "BX":
            for mod in ("excel_export_job", "excel_export_job_zhx", "excel_export_job_qsp"):
                m = HELPERS.get(mod)
                if m and hasattr(m, "write_BX_excel"):
                    return m.write_BX_excel(dets, DEFECT_XLS_DIR, img_stem)  # type: ignore
        if code == "ZAW":
            for mod in ("excel_export_job", "excel_export_job_zhx", "excel_export_job_qsp"):
                m = HELPERS.get(mod)
                if m and hasattr(m, "write_ZAW_excel"):
                    return m.write_ZAW_excel(dets, DEFECT_XLS_DIR, img_stem)  # type: ignore
        if code == "CJ":
            for mod in ("excel_export_job", "excel_export_job_zhx", "excel_export_job_qsp"):
                m = HELPERS.get(mod)
                if m and hasattr(m, "write_CJ_excel"):
                    return m.write_CJ_excel(dets, DEFECT_XLS_DIR, img_stem)  # type: ignore
        if code == "TJ":
            for mod in ("excel_export_job", "excel_export_job_zhx", "excel_export_job_qsp"):
                m = HELPERS.get(mod)
                if m and hasattr(m, "write_TJ_excel"):
                    return m.write_TJ_excel(dets, DEFECT_XLS_DIR, img_stem)  # type: ignore
        if code == "CK":
            for mod in ("excel_export_job", "excel_export_job_zhx", "excel_export_job_qsp"):
                m = HELPERS.get(mod)
                if m and hasattr(m, "write_CK_excel"):
                    return m.write_CK_excel(dets, DEFECT_XLS_DIR, img_stem)  # type: ignore
    except Exception as e:
        print(f"[warn] helper writer for {code} failed: {e}")

    # fallback
    return write_defect_excel_fallback(code, img_stem, dets)
# ------------------------------------------------------


# ----------------- JPype + Drools ---------------------
import jpype
import jpype.imports
from jpype.types import JString

_JVM_READY = False

def _start_jvm_once():
    global _JVM_READY
    if _JVM_READY:
        return
    if not jpype.isJVMStarted():
        jars = [DROOLS_MAIN_JAR]
        if os.path.isdir(DROOLS_DEPS_DIR):
            for fn in os.listdir(DROOLS_DEPS_DIR):
                if fn.lower().endswith(".jar"):
                    jars.append(os.path.join(DROOLS_DEPS_DIR, fn))
        classpath = os.pathsep.join(jars)
        jpype.startJVM(JVM_DLL, "-Xmx2g", f"-Djava.class.path={classpath}")
    _JVM_READY = True

def _pipe_field(obj, getter_name, default=None):
    try:
        if obj is None:
            return default
        m = getattr(obj, getter_name, None)
        if m is None:
            return default
        val = m()
        return None if val is None else val
    except Exception:
        return default

def _translate_cn(text_cn: str, grade: int, by_grade_map: Dict[int, str]) -> str:
    if not text_cn:
        return by_grade_map.get(grade, "")
    return CN_TO_EN.get(text_cn.strip(), by_grade_map.get(grade, ""))

def run_drools_on_excel(excel_path: str) -> List[Dict[str, Any]]:
    _start_jvm_once()
    # import your Java class
    try:
        from runall import runall
        RunAll = runall
    except Exception:
        from jpype import JClass
        RunAll = JClass("runall")

    result = RunAll().excel_pipe_drools(JString(excel_path))

    # unify to list
    pipes = []
    try:
        size = getattr(result, "size", None)
        if callable(size):
            for i in range(result.size()):
                pipes.append(result.get(i))
        else:
            pipes.append(result)
    except Exception:
        pipes = [result]

    parsed = []
    for p in pipes:
        parsed.append({
            "uri": _pipe_field(p, "getUri"),
            "gradePL": _pipe_field(p, "getGradePL"),
            "gradeBX": _pipe_field(p, "getGradeBX"),
            "gradeFS": _pipe_field(p, "getGradeFS"),
            "gradeCK": _pipe_field(p, "getGradeCK"),
            "gradeQF": _pipe_field(p, "getGradeQF"),
            "gradeTJ": _pipe_field(p, "getGradeTJ"),
            "gradeTL": _pipe_field(p, "getGradeTL"),
            "gradeAJ": _pipe_field(p, "getGradeAJ"),
            "gradeCR": _pipe_field(p, "getGradeCR"),
            "gradeSL": _pipe_field(p, "getGradeSL"),
            "gradeZAW": _pipe_field(p, "getGradeZAW"),
            "gradeCJ": _pipe_field(p, "getGradeCJ"),
            "damage_status_description_cn": _pipe_field(p, "getDamage_status_description"),
            "repair_advice_cn": _pipe_field(p, "getRepair_advice"),
            "ruptureArea": _pipe_field(p, "getRuptureArea"),
            "Arclength": _pipe_field(p, "getArclength"),
            "depositionRatio": _pipe_field(p, "getDepositionRatio"),
            "crossSectionLoss": _pipe_field(p, "getCrossSectionLoss"),
        })
    return parsed

def severity_from_drools(items: List[Dict[str, Any]], defect_code: str) -> Tuple[str, Any, str, str, Dict[str, Any]]:
    if not items:
        return "Unknown", None, "", "", {}
    x = items[0]
    field = {
        "PL": "gradePL",
        "BX": "gradeBX",
        "FS": "gradeFS",
        "CK": "gradeCK",
        "QF": "gradeQF",
        "TJ": "gradeTJ",
        "TL": "gradeTL",
        "AJ": "gradeAJ",
        "CR": "gradeCR",
        "SL": "gradeSL",
        "ZAW": "gradeZAW",
        "CJ": "gradeCJ",
    }.get(defect_code.upper())
    grade = x.get(field)
    try:
        grade = int(grade) if grade is not None else None
    except Exception:
        grade = None
    severity = GRADE_TO_SEVERITY.get(grade, "Unknown")
    desc_en = _translate_cn(x.get("damage_status_description_cn"), grade, DESC_EN)
    advice_en = _translate_cn(x.get("repair_advice_cn"), grade, ADVICE_EN)
    metrics = {k: v for k, v in x.items() if k in ("ruptureArea", "Arclength", "depositionRatio", "crossSectionLoss") and v is not None}
    return severity, grade, desc_en, advice_en, metrics
# ------------------------------------------------------


# ---------------------- PIPELINE ----------------------
def group_by_code(dets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {}
    for d in dets:
        code = d["code"]
        if code == "UNK":
            # ignore unknown classes
            continue
        buckets.setdefault(code, []).append(d)
    return buckets


def ensure_csv_header(path: str):
    header = [
        "Image", "DefectCode", "ClassName", "xc", "yc", "w", "h", "img_w", "img_h", "Confidence",
        "Severity", "Grade", "DamageDescription", "RepairAdvice",
        "RuptureArea", "ArcLength", "DepositionRatio", "CrossSectionLoss"
    ]
    need_header = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if need_header:
        w.writerow(header)
    return f, w


def process_single_image(model, image_path: str) -> List[Dict[str, Any]]:
    """
    Run detection for one image, export per-defect Excel(s), run Drools,
    and return a list of enriched rows.
    """
    rows = []
    res_list = run_yolov8_on_image(model, image_path, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
    if not res_list:
        return rows
    # Ultralytics v8 returns a list; take the first Result for this image
    r0 = res_list[0]
    dets = extract_detections(r0, image_path)
    if not dets:
        return rows

    img_stem = os.path.splitext(os.path.basename(image_path))[0]
    grouped = group_by_code(dets)

    for code, code_dets in grouped.items():
        # Write Excel for this defect & image (try helpers; fallback to generic)
        try:
            excel_path = write_defect_excel(code, img_stem, code_dets)
        except Exception as e:
            print(f"[warn] Excel writer for {code} failed: {e}")
            continue

        # Call Drools on that Excel
        try:
            items = run_drools_on_excel(excel_path)
        except Exception as e:
            print(f"[warn] Drools failed for {code}/{img_stem}: {e}")
            continue

        severity, grade, desc_en, advice_en, metrics = severity_from_drools(items, code)

        # Build summary rows per detection of this code
        for d in code_dets:
            rows.append([
                os.path.basename(image_path), code, d["class_name"],
                f"{d['xc']:.3f}", f"{d['yc']:.3f}", f"{d['w']:.3f}", f"{d['h']:.3f}",
                int(d["img_w"]), int(d["img_h"]), f"{d['conf']:.4f}",
                severity, grade if grade is not None else "",
                desc_en, advice_en,
                metrics.get("ruptureArea", ""), metrics.get("Arclength", ""),
                metrics.get("depositionRatio", ""), metrics.get("crossSectionLoss", "")
            ])

    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = load_yolov8_model(MODEL_PATH)

    # Gather images
    patterns = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]
    images = []
    for p in patterns:
        images.extend(glob.glob(os.path.join(IMAGES_DIR, p)))
    images = sorted(images)
    print(f"Found {len(images)} images under {IMAGES_DIR}")

    f, w = ensure_csv_header(FINAL_SUMMARY_CSV)

    try:
        for idx, img in enumerate(images, start=1):
            print(f"\n[{idx}/{len(images)}] Processing {img}")
            # Do one run, allow interactive control
            while True:
                rows = process_single_image(model, img)
                if rows:
                    for row in rows:
                        w.writerow(row)
                    f.flush()
                    print(f"  → Wrote {len(rows)} row(s) for {os.path.basename(img)}")
                else:
                    print("  → No defect Excel produced / no detections.")

                if not INTERACTIVE:
                    break

                try:
                    choice = input("Press Enter for next image, [s]=skip reasoning next time, [r]=re-run reasoning, [q]=quit: ").strip().lower()
                except EOFError:
                    choice = ""
                if choice == "q":
                    print("Exiting.")
                    return
                elif choice == "r":
                    print("Re-running reasoning for this image...")
                    continue
                elif choice == "s":
                    print("Skipping reasoning for this image (move on).")
                    break
                else:
                    break  # default: proceed to next image
    finally:
        f.close()

    print(f"\nAll images processed. Results saved to: {FINAL_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
