import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ---- Head-turn-only mutual gaze (works with glasses) ----
L_EO,L_EI,R_EI,R_EO,NOSE = 33,133,362,263,1
mpfm = mp.solutions.face_mesh

def xy(l,w,h): return np.array([l.x*w, l.y*h], dtype=np.float32)

def eye_mid_and_span(lms,w,h):
    lo, li = xy(lms[L_EO],w,h), xy(lms[L_EI],w,h)
    ri, ro = xy(lms[R_EI],w,h), xy(lms[R_EO],w,h)
    lc = (lo+li)/2.0; rc = (ro+ri)/2.0
    mid = (lc+rc)/2.0
    span = np.linalg.norm(rc-lc) + 1e-6
    return mid, span

def head_yaw(lms,w,h):
    """ + right, - left (camera view) """
    mid, span = eye_mid_and_span(lms,w,h)
    nose = xy(lms[NOSE], w, h)
    return float((nose[0] - mid[0]) / span)

def face_center(lms,w,h):
    pts = np.array([[p.x*w, p.y*h] for p in lms], dtype=np.float32)
    mn = pts.min(0); mx = pts.max(0)
    return (mn+mx)/2.0

def decide_mutual(a, b, H, yaw_thr=0.10, vert_align=0.22):
    if abs(a["center"][1]-b["center"][1]) / H > vert_align:
        return False
    a_should_right = b["center"][0] > a["center"][0]
    b_should_right = a["center"][0] > b["center"][0]
    a_ok = (a["yaw"] >  yaw_thr) if a_should_right else (a["yaw"] < -yaw_thr)
    b_ok = (b["yaw"] >  yaw_thr) if b_should_right else (b["yaw"] < -yaw_thr)
    return a_ok and b_ok

class Processor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mpfm.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        faces = []
        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks[:2]:
                lms = lm.landmark
                faces.append({
                    "center": face_center(lms, w, h),
                    "yaw":    head_yaw(lms, w, h),
                })

        # Draw debug text
        for i, f in enumerate(faces):
            arrow = "→" if f["yaw"] > 0 else "←"
            cv2.putText(img, f"P{i+1} yaw {arrow} {f['yaw']:+.2f}", (20, 70 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        msg, color = "NOT LOOKING AT EACH OTHER", (0, 0, 255)
        if len(faces) == 2 and decide_mutual(faces[0], faces[1], h):
            msg, color = "LOOKING AT EACH OTHER", (0, 255, 0)

        cv2.rectangle(img, (0,0), (w, 45), (0,0,0), -1)
        cv2.putText(img, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Mutual Gaze (Python, Streamlit)", page_icon="👀", layout="centered")
st.markdown("## 👀 Mutual Gaze Detector (Head-only) \n\nAllow camera when asked. Works on phones too.")
webrtc_streamer(
    key="gaze",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Processor,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
)
st.caption("Tip: good lighting helps. If false positives, rotate a bit more or step back slightly.")
