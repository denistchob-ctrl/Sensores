#Vídeo IDENTIFICADOR DE PESSOAS.mp4
from ultralytics import YOLO
import cv2
import time
import os
import csv
from collections import defaultdict

MODEL_PATH = "yolov8n.pt"  # Path to the YOLOv8 model
VIDEO_PATH = "people.mp4"  # Path to the input video
OUTPUT_VIDEO = "output-v2.mp4"  # Path to save the output video
CONFIDENCE = 0.5  # Confidence threshold for detections

model = YOLO(MODEL_PATH)

#Estatísticas
total_frames = 0
total_objects = 0
class_counter = defaultdict(int)

#Configurações de Vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

#CSV Log
csv_file = open("detection_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Class", "Confidence"])

#Processamento de vídeo
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    total_frames += 1

    # Realizar a detecção
    results = model.predict(
        frame, 
        conf=CONFIDENCE, 
        verbose=False, 
        device="cpu" #sugestão da IA
    )

    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        class_counter[class_name] += 1
        total_objects += 1

        csv_writer.writerow([total_frames, class_name, round(conf,2)])

    annotated = results[0].plot()

    current_time = time.time()
    fps_live = total_frames / (current_time - start_time)

    cv2.putText(
        annotated,
        f"FPS: {fps_live:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Total Objects: {total_objects}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    writer.write(annotated)
    cv2.imshow("YOLOv8 Advanced Detector", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

    #Sugestão da IA para processar cada resultado individualmente.
    # for result in results:
    #     boxes = result.boxes.xyxy.cpu().numpy()
    #     confidences = result.boxes.conf.cpu().numpy()
    #     classes = result.boxes.cls.cpu().numpy()

    #     for box, conf, cls in zip(boxes, confidences, classes):
    #         if conf >= CONFIDENCE:
    #             total_objects += 1
    #             class_counter[int(cls)] += 1

    #             # Desenhar a caixa delimitadora e a classe no quadro
    #             x1, y1, x2, y2 = map(int, box)
    #             label = f"{model.names[int(cls)]}: {conf:.2f}"
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #             # Log para CSV
    #             csv_writer.writerow([total_frames, model.names[int(cls)], conf])

    # writer.write(frame)

#Limpeza de recursos
cap.release()
writer.release()
csv_file.close()
cv2.destroyAllWindows()

#Relatório final
print("\n================ Relatório Final ================")
print(f"Frames Processed : {total_frames}")
print(f"Total Objects Detected : {total_objects}")
for cls, count in sorted(class_counter.items()):
    print(f"{model.names[cls]}: {count}") #sugestão da IA. ver o que mostra ou apagar se nao for necessário.
    print(f"{cls}: {count}") 

