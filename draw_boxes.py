import numpy as np
import cv2

def drawBox(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  (x,y), (w,h) = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, (x,y), (w,h), color, thickness=1, lineType=cv2.LINE_AA)
  if label:
    t_w, t_h = cv2.getTextSize(label, 0, 0.5, thickness=1)[0]  # text width, height
    outside = y - t_h >= 3

    (w,h) = x + t_w, y - t_h - 3 if outside else y + t_h + 3
    cv2.rectangle(image, (x,y), (w,h), color, -1, cv2.LINE_AA)  # filled
    
    cv2.putText(image,
                label, (x, y - 2 if outside else y + t_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                txt_color,
                thickness=1,
                lineType=cv2.LINE_AA)
                
                
def drawBoxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  #Define COCO Labels
  if labels == []:
    labels = {0: u'car', 1: u'bus', 2: u'minibus',3: u'truck', 4: u'van', 5: u'motorbike'}
  #Define colors
  if colors == []:
    colors = [(89, 161, 197),(57,76,139),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),	(139,71,93),(84,139,84)]
  
  #plot each boxes
  for box in boxes:
    #add score in label if score=True
    if score :
      label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
    else :
      label = labels[int(box[-1])]
      
    #filter every box under conf threshold if conf threshold setted
    if conf is None or box[-2] > conf :
        # print(box)
        idx = int(box[-1]) % len(colors)
        color = colors[idx]
        drawBox(image, box, label, color)
               
                
                