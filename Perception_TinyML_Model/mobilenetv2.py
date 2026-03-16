import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque

try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
except ImportError:
    print("Install PyTorch: pip install torch torchvision")
    exit(1)

DEVICE = (torch.device("mps") if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"OpenCV {cv2.__version__} PyTorch {torch.__version__} Device: {DEVICE}")

MODEL_CONFIG = "deploy.prototxt"
MODEL_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
MODEL_CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

def download_models():
    for url, path in [(MODEL_CONFIG_URL, MODEL_CONFIG), (MODEL_WEIGHTS_URL, MODEL_WEIGHTS)]:
        if not os.path.exists(path):
            print(f"Downloading {path}…")
            urllib.request.urlretrieve(url, path)

CAMERA_INDEX = 0
FACE_CONF = 0.6
CALIB_INTERVAL = 10
CR_MAD_MULT = 2.5
CB_MAD_MULT = 3.0
HUE_PAD = 12
HUE_MAX = 30
MIN_AREA = 3500
MAX_AREA_FRAC = 0.30
MAX_HULL_FRAC = 0.32
MIN_SOL = 0.30
MAX_SOL = 0.97
MIN_WH = 0.40
MAX_WH = 1.85
MIN_DEFECT = 6000
FINGER_ZONE = 0.62
ISOLATION = 0.15
DISPLAY_FRAC = 0.62
SMOOTH_WIN = 3
SMOOTH_HITS = 2
FACE_PAD = 10
NECK_EXT = 0.22
SHOULDER_SIDE = 0.45
SHOULDER_VERT = 0.35
C_FACE = (0, 255, 0)
C_HULL = (0, 200, 255)
C_VALLEY = (0, 0, 255)
C_TIP = (255, 255, 255)
C_LABEL = (30, 30, 30)

def detect_faces(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104,177,123))
    net.setInput(blob)
    dets = net.forward()
    boxes, scores = [], []
    for i in range(dets.shape[2]):
        c = dets[0,0,i,2]
        if c > FACE_CONF:
            boxes.append([max(0.,dets[0,0,i,3]), max(0.,dets[0,0,i,4]),
                          min(1.,dets[0,0,i,5]), min(1.,dets[0,0,i,6])])
            scores.append(float(c))
    return np.array(boxes), np.array(scores)

def sample_skin(frame, face_boxes):
    if not len(face_boxes):
        return None
    h, w = frame.shape[:2]
    b = face_boxes[0]
    fx1, fy1 = int(b[0]*w), int(b[1]*h)
    fx2, fy2 = int(b[2]*w), int(b[3]*h)
    fh, fw = fy2-fy1, fx2-fx1
    py1 = fy1+int(fh*0.45); py2 = fy1+int(fh*0.70)
    px1 = fx1+int(fw*0.30); px2 = fx1+int(fw*0.70)
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[py1:py2,px1:px2].reshape(-1,3).astype(float)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[py1:py2,px1:px2].reshape(-1,3).astype(float)
    yl, yh = np.percentile(ycc[:,0],30), np.percentile(ycc[:,0],90)
    keep = (ycc[:,0]>=yl)&(ycc[:,0]<=yh)
    ycc, hsv = ycc[keep], hsv[keep]
    if len(ycc) < 20: return None
    med = np.median(ycc, axis=0)
    mad = np.median(np.abs(ycc-med), axis=0) * 1.4826
    return med, mad, hsv.mean(axis=0)

def skin_mask(frame, calib):
    b = cv2.GaussianBlur(frame, (5,5), 0)
    ycc = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    if calib is None:
        m1 = cv2.inRange(ycc, np.array([0,133,90],np.uint8), np.array([255,182,135],np.uint8))
        m2 = cv2.inRange(hsv, np.array([0,20,40],np.uint8), np.array([25,255,255],np.uint8))
    else:
        med, mad, mhsv = calib
        cr_lo = max(118, int(med[1]-CR_MAD_MULT*max(mad[1],6)))
        cr_hi = min(255, int(med[1]+CR_MAD_MULT*max(mad[1],6)))
        cb_lo = max(85, int(med[2]-CB_MAD_MULT*max(mad[2],5)))
        cb_hi = min(255, int(med[2]+CB_MAD_MULT*max(mad[2],5)))
        h_lo = max(0, int(mhsv[0]-HUE_PAD))
        h_hi = min(HUE_MAX, int(mhsv[0]+HUE_PAD))
        m1 = cv2.inRange(ycc, np.array([0,cr_lo,cb_lo],np.uint8), np.array([255,cr_hi,cb_hi],np.uint8))
        m2 = cv2.inRange(hsv, np.array([h_lo,15,30],np.uint8), np.array([h_hi,255,255],np.uint8))
    mask = cv2.bitwise_and(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask = cv2.dilate(mask, k, iterations=1)
    return mask

def exclude_body(mask, face_boxes, shape):
    fh, fw = shape[:2]
    cv2.rectangle(mask, (0,int(fh*0.88)), (fw,fh), 0, -1)
    for b in face_boxes:
        fx1, fy1 = int(b[0]*fw), int(b[1]*fh)
        fx2, fy2 = int(b[2]*fw), int(b[3]*fh)
        faceW = fx2-fx1
        p = FACE_PAD
        cv2.rectangle(mask, (fx1-p,fy1-p), (fx2+p,fy2+p), 0, -1)
        cv2.rectangle(mask, (fx1-p*2,0), (fx2+p*2,max(0,fy1-p)), 0, -1)
        cv2.rectangle(mask, (fx1,fy2), (fx2,min(fh,fy2+int(NECK_EXT*fh))), 0, -1)
        sw = int(faceW*SHOULDER_SIDE)
        sy2 = min(fh, fy2+int(SHOULDER_VERT*fh))
        cv2.rectangle(mask, (fx1-sw,fy1), (fx1+p, sy2), 0, -1)
        cv2.rectangle(mask, (fx2-p, fy1), (fx2+sw,sy2), 0, -1)
    return mask

def find_hands(mask, face_boxes, shape):
    exclude_body(mask, face_boxes, shape)
    fh, fw = shape[:2]; fa = fh*fw
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA_FRAC*fa): continue
        hull = cv2.convexHull(c); ha = cv2.contourArea(hull)
        if ha < 1 or ha > MAX_HULL_FRAC*fa: continue
        sol = area/ha
        if not (MIN_SOL <= sol <= MAX_SOL): continue
        _,_,bw,bh = cv2.boundingRect(c)
        wh = bw/bh if bh>0 else 0
        if not (MIN_WH <= wh <= MAX_WH): continue
        hi = cv2.convexHull(c, returnPoints=False)
        d = cv2.convexityDefects(c, hi) if hi is not None and len(hi)>=3 else None
        max_d = int(d[:,0,3].max()) if d is not None else 0
        if max_d < MIN_DEFECT and sol < 0.75: continue
        valid.append(c)
    valid.sort(key=cv2.contourArea, reverse=True)
    return valid[:2]

def contour_y_range(c):
    pts = c[:,0,1]
    return int(pts.min()), int(pts.max())

def detect_fingers(contour):
    hi = cv2.convexHull(contour, returnPoints=False)
    if hi is None or len(hi)<3: return [], []
    defects = cv2.convexityDefects(contour, hi)
    ymin, ymax = contour_y_range(contour)
    y_cut = ymin + int((ymax-ymin)*FINGER_ZONE)
    tips, valleys = set(), set()
    if defects is not None:
        for d in defects:
            s,e,f,depth = d[0]
            if depth < MIN_DEFECT: continue
            st = tuple(contour[s][0]); en = tuple(contour[e][0]); va = tuple(contour[f][0])
            a = np.linalg.norm(np.subtract(st,va))
            b = np.linalg.norm(np.subtract(en,va))
            cc = np.linalg.norm(np.subtract(st,en))
            d2 = 2*a*b
            if d2==0: continue
            angle = np.degrees(np.arccos(np.clip((a**2+b**2-cc**2)/d2,-1,1)))
            if angle < 90:
                if st[1]<=y_cut: tips.add(st)
                if en[1]<=y_cut: tips.add(en)
                if va[1]<=y_cut: valleys.add(va)
    hull_pts = cv2.convexHull(contour)
    top_pt = tuple(hull_pts[int(hull_pts[:,0,1].argmin())][0])
    if top_pt[1] <= y_cut:
        tips.add(top_pt)
    return list(tips), list(valleys)

def classify_finger(tip, contour):
    M = cv2.moments(contour)
    if M["m00"]==0: return None
    cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
    angle = np.degrees(np.arctan2(cy-tip[1], tip[0]-cx))
    if 60 <= angle <= 120: return "Index Finger"
    if -30 <= angle < 60 or 120 < angle <= 200: return "Thumb"
    return None

def label_fingers(tips, contour):
    if not tips: return {}
    ymin, ymax = contour_y_range(contour)
    hand_h = max(1, ymax-ymin)
    sorted_tips = sorted(tips, key=lambda t: t[1])
    labels = {}
    top = sorted_tips[0]
    name = classify_finger(top, contour)
    if name: labels[top] = name
    if len(sorted_tips) >= 2:
        n2 = classify_finger(sorted_tips[1], contour)
        if n2 and n2 != name: labels[sorted_tips[1]] = n2
    return labels

class HandSmoother:
    TOL = 60
    def __init__(self):
        self.history = deque(maxlen=SMOOTH_WIN)
    def update(self, hands):
        self.history.append([(h["bbox"][0]+h["bbox"][2]//2,
                               h["bbox"][1]+h["bbox"][3]//2) for h in hands])
    def confirmed(self, hands):
        out = []
        for h in hands:
            bx, by, bw, bh = h["bbox"]
            cx, cy = bx+bw//2, by+bh//2
            hits = sum(any((cx-px)**2+(cy-py)**2 < self.TOL**2 for px,py in past)
                       for past in self.history)
            if hits >= SMOOTH_HITS: out.append(h)
        return out

class MN2:
    _T = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __init__(self):
        print("Loading MobileNetV2…")
        m = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feat = m.features.eval().to(DEVICE)
        with torch.no_grad(): self.feat(torch.zeros(1,3,224,224,device=DEVICE))
        print("MobileNetV2 ready.")
    def ratio(self, frame, contour):
        fh, fw = frame.shape[:2]
        x, y, cw, ch = cv2.boundingRect(contour)
        x1, y1 = max(0,x), max(0,y); x2, y2 = min(fw,x+cw), min(fh,y+ch)
        if x2-x1<10 or y2-y1<10: return 0.0
        t = self._T(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): f = self.feat(t)[0].mean(0).cpu().numpy()
        m = f.mean(); return float(f[:3].mean()/m) if m>1e-6 else 0.0

def draw_faces(frame, boxes, scores):
    fh, fw = frame.shape[:2]
    for box, score in zip(boxes, scores):
        x1, y1 = int(box[0]*fw), int(box[1]*fh)
        x2, y2 = int(box[2]*fw), int(box[3]*fh)
        cv2.rectangle(frame,(x1,y1),(x2,y2),C_FACE,2)
        lbl = f"Face {score:.2f}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(frame,(x1,y1-th-10),(x1+tw,y1),C_FACE,-1)
        cv2.putText(frame,lbl,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def draw_hand(frame, contour, tips, valleys, labels, mn2_ratio):
    ymin, ymax = contour_y_range(contour)
    y_cut = ymin+int((ymax-ymin)*DISPLAY_FRAC)
    clipped = contour[contour[:,0,1]<=y_cut]
    if len(clipped)>=3:
        cv2.drawContours(frame,[cv2.convexHull(clipped)],-1,C_HULL,2)
    for v in valleys:
        if v[1]<=y_cut: cv2.circle(frame,v,5,C_VALLEY,-1)
    for tip in tips:
        cv2.circle(frame,tip,7,C_TIP,-1); cv2.circle(frame,tip,7,C_HULL,2)
        name = labels.get(tip)
        if name:
            tx, ty = tip[0]-10, tip[1]-15
            (tw,th),_ = cv2.getTextSize(name,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
            cv2.rectangle(frame,(tx-3,ty-th-4),(tx+tw+3,ty+4),C_LABEL,-1)
            cv2.putText(frame,name,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
    bx, by, bw, bh = cv2.boundingRect(clipped if len(clipped)>=1 else contour)
    cv2.putText(frame,f"MN2:{mn2_ratio:.2f}",(bx,by+bh+18),cv2.FONT_HERSHEY_SIMPLEX,0.45,C_HULL,1)

def draw_hud(frame, fps, nf, nh, cal):
    cv2.putText(frame,f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame,f"Faces:{nf}", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,C_FACE,2)
    cv2.putText(frame,f"Hands:{nh}", (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.7,C_HULL,2)
    cv2.putText(frame,"CAL" if cal else "INIT",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,
                (0,255,0) if cal else (0,165,255),2)
    cv2.putText(frame,"q=quit s=save d=mask",(10,frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

def main():
    download_models()
    face_net = cv2.dnn.readNetFromCaffe(MODEL_CONFIG, MODEL_WEIGHTS)
    mn2 = MN2()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    smoother = HandSmoother()
    calib = None
    frame_n = 0
    fps = 0.0
    t0 = time.time()
    saved = 0
    debug_mask = False
    print("\nRunning — raise your hand, extend one finger to label it.")
    print("d=mask s=save q=quit\n")
    while True:
        ret, frame = cap.read()
        if not ret: break
        face_boxes, face_scores = detect_faces(face_net, frame)
        if frame_n % CALIB_INTERVAL == 0:
            c = sample_skin(frame, face_boxes)
            if c is not None: calib = c
        raw_mask = skin_mask(frame, calib)
        contours = find_hands(raw_mask.copy(), face_boxes, frame.shape)
        raw_hands = []
        for c in contours:
            tips, valleys = detect_fingers(c)
            labels = label_fingers(tips, c)
            ratio = mn2.ratio(frame, c)
            raw_hands.append(dict(contour=c, bbox=cv2.boundingRect(c),
                                  tips=tips, valleys=valleys, labels=labels, ratio=ratio))
        smoother.update(raw_hands)
        hands = smoother.confirmed(raw_hands)
        if debug_mask:
            dbg = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(dbg,"Skin mask (d=hide)",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.imshow("Skin Mask", dbg)
        else:
            try: cv2.destroyWindow("Skin Mask")
            except: pass
        draw_faces(frame, face_boxes, face_scores)
        for h in hands:
            draw_hand(frame, h["contour"], h["tips"], h["valleys"], h["labels"], h["ratio"])
        draw_hud(frame, fps, len(face_boxes), len(hands), calib is not None)
        frame_n += 1
        if frame_n % 10 == 0:
            fps = frame_n / (time.time()-t0)
        cv2.imshow("Hand Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("s"):
            fn = f"saved_{saved}.jpg"; cv2.imwrite(fn,frame); print(f"Saved {fn}"); saved+=1
        elif key == ord("d"): debug_mask = not debug_mask
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done — {frame_n} frames @ {fps:.1f} FPS")

if __name__ == "__main__":
    main()