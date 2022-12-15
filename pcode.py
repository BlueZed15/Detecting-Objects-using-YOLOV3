import cv2, numpy as ny

def eva(bprev,bnext):
    return [(bprev[0]+bnext[0])/2,(bprev[1]+bnext[1])/2,(bprev[2]+bnext[2])/2,(bprev[3]+bnext[3])/2]


vd = cv2.VideoCapture(r"C:\Users\Rohith\Downloads\production ID_4644508 (2) .mp4")
ob=cv2.dnn.readNet('yolov3-tiny.cfg','yolov3-tiny.weights')
while True:
    _,ima=vd.read()
    ima=cv2.resize(ima,(600,600))
    cv2.medianBlur(ima,3)
    im=cv2.dnn.blobFromImage(ima,1/255.0,(416,416),swapRB=True,crop=False)
    
    ob.setInput(im)
    lay=ob.getLayerNames()
    lay2=[lay[i-1] for i in ob.getUnconnectedOutLayers()]
    out=ob.forward(lay2)
    bound=[]
    confis=[]
    h,w=ima.shape[:2]

    for a in out:
        for det in a:
            prob=det[5:]
            confid=prob[ny.argmax(prob)]
            if confid > 0.6:
                box = det[:4] * (ny.array([w,h,w,h]))
                (cx, cy, wi, hi) = box.astype("int")
                x,y =int(cx-(wi/2)),int(cy-(hi/2))
                confis.append(float(confid))
                bound.append([x,y,int(wi),int(hi)])
                
    indices = cv2.dnn.NMSBoxes(bound, confis, 0.25, 0.1)
    print(indices,'....',bound)
    if len(indices) > 0:
        for i in range(len(indices)):
            if bound[indices[i]]==[]:
                rect=eva(bound[indices[i-1]],bound[indices[i+1]])
            else:
                rect=bound[indices[i]]
            xe,ye=int(rect[0]+rect[2]),int(rect[1]+rect[3])
            cv2.rectangle(ima,(int(rect[0]),int(rect[1])),(xe,ye),(0,255,0),1)

    cv2.imshow('og',ima)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.VideoCapture.release(vd)
cv2.destroyAllWindows()
