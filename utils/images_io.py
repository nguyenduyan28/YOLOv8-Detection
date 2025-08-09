import io, requests


from PIL import Image

import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def bytes_to_pil(bytes_data : bytes) -> Image.Image:
  return Image.open(io.BytesIO(bytes_data)).convert("RGB")


def pil_to_bytes(pil_image : Image.Image, fmt="PNG") -> bytes:
  buf = io.BytesIO()
  pil_image.save(buf, format=fmt)
  return buf.getvalue()


def array_to_bytes(arr : np.array) -> bytes:
  return arr.tobytes()



def download_image(url : str, timeout = 10) -> bytes:
  r = requests.get(url, timeout=timeout)
  r.raise_for_status()
  return r.content