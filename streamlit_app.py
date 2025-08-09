import streamlit as st
from yolo.detector import Detector
from utils.images_io import download_image, bytes_to_pil, pil_to_bytes
from streamlit_paste_button import paste_image_button as p_button
from PIL import Image
import yaml

st.set_page_config(page_title="YOLOv8 Web", layout="centered")

@st.cache_resource
def load_cfg():
  app_cfg = yaml.safe_load(open("configs/app.yaml"))
  mdl_cfg = yaml.safe_load(open("configs/model.yaml"))
  return app_cfg, mdl_cfg

@st.cache_resource(show_spinner=False)
def load_detector(app_cfg, mdl_cfg):
  return Detector(
    model_path=mdl_cfg.get("model_path", "yolov8n.pt"),
    device=mdl_cfg.get("device", "cpu"),
    half=mdl_cfg.get("half", True),
    imgsz=app_cfg.get("imgsz", 416),
    conf=app_cfg.get("conf", 0.35),
    iou=app_cfg.get("iou", 0.5),
    max_det=app_cfg.get("max_det", 50),
  )

app_cfg, mdl_cfg = load_cfg()
detector = load_detector(app_cfg, mdl_cfg)

if "show_panel" not in st.session_state:
  st.session_state.show_panel = False
if "img_bytes" not in st.session_state:
  st.session_state.img_bytes = None
if "file_name" not in st.session_state:
  st.session_state.file_name = "image.png"

st.title("Demo detecting things in YOLOv8")

if st.button("Upload image"):
  st.session_state.show_panel = not st.session_state.show_panel
  st.session_state.img_bytes = None
  st.session_state.file_name = "image.png"

if st.session_state.show_panel:
  src = st.radio(
    "Upload in one of three ways:",
    ["Paste (Clipboard)", "URL", "Upload file"],
    horizontal=True
  )

  if src == "Paste (Clipboard)":
    pasted = p_button("Paste image here")
    if pasted and hasattr(pasted, "image_data") and pasted.image_data:
      st.session_state.img_bytes = pasted.image_data
    elif pasted and (not hasattr(pasted, "image_data") or pasted.image_data) :
      st.error(f"Clipboard value does not valid")

  if src == "Upload file":
    up = st.file_uploader(
      "Upload your image…",
      type=["jpg", "jpeg", "png"]
    )
    if up:
      st.session_state.img_bytes = up.getvalue()
      st.session_state.file_name = up.name

  if src == "URL":
    url = st.text_input(
      "Paste the URL here (https://…)",
      placeholder="https://example.com/image.jpg"
    )
    if url:
      try:
        st.session_state.img_bytes = download_image(url)
        name = url.split("/")[-1].split("?")[0] or "url_image.png"
        st.session_state.file_name = (
          name if any(name.lower().endswith(e) for e in [".jpg", ".jpeg", ".png"])
          else "url_image.png"
        )
      except Exception as e:
        st.error(f"Cannot load image {e}")

if st.session_state.img_bytes:
  try:
    img = bytes_to_pil(st.session_state.img_bytes) if isinstance(st.session_state.img_bytes, bytes) else st.session_state.img_bytes
  except Exception as e:
    st.error(f"Image is not valid: {e}")
    st.stop()

  holder = st.empty()
  res = detector.predict(img)
  det_rgb = res.plot()[..., ::-1]
  img_show = Image.fromarray(det_rgb)

  with holder.container():
    tabs = st.tabs(["Detected", "Original"])
    with tabs[0]:
      st.image(det_rgb, caption="Detected image", use_container_width=True)
    with tabs[1]:
      st.image(img, caption="Original image", use_container_width=True)

    id2name = getattr(res, "names", {})
    present_ids = res.boxes.cls.int().tolist() if hasattr(res, "boxes") and len(res.boxes) else []
    present_labels = sorted({id2name.get(i, str(i)) for i in present_ids})
    tag = st.selectbox("Filter & crop by label (labels in this image)", options=["—"] + present_labels, index=0)
    if tag != "—" and present_ids:
      name2id = {v: k for k, v in id2name.items()}
      cls_id = name2id.get(tag, None)
      if cls_id is not None:
        boxes = res.boxes.xyxy.tolist()
        clss = res.boxes.cls.int().tolist()
        crops = []
        W, H = img.size
        for b, c in zip(boxes, clss):
          if c == cls_id:
            x1, y1, x2, y2 = [int(max(0, v)) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
              crops.append(img.crop((x1, y1, x2, y2)))
        if crops:
          st.subheader(f"Cropped: {tag} × {len(crops)}")
          ncol = 3
          rows = (len(crops) + ncol - 1) // ncol
          idx = 0
          for _ in range(rows):
            cols = st.columns(ncol)
            for j in range(ncol):
              if idx < len(crops):
                cols[j].image(crops[idx], use_container_width=True)
                idx += 1
        else:
          st.info("No boxes for selected label.")

    c1, c2, c3 = st.columns(3)
    with c1:
      st.download_button(
        "Download original",
        data=pil_to_bytes(img),
        file_name=f"original_{st.session_state.file_name.rsplit('.', 1)[0]}.png",
        mime="image/png"
      )
    with c2:
      st.download_button(
        "Download detected",
        data = pil_to_bytes(img_show),
        file_name=f"detected_{st.session_state.file_name.rsplit('.', 1)[0]}.png",
        mime="image/png"
      )
    with c3:
      if st.button("Clear image"):
        st.session_state.img_bytes = None
        holder.empty()