#Vídeo CONTANDO PESSOAS.mp4
from ultralytics import YOLO #pip install ultralytics
import cv2
import numpy as np
import supervision as sv  #pip install supervision
import os
from datetime import datetime

#Load model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

#Video
VIDEO_PATH = "people.mp4"

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(
        f"Não encontrei o arquivo de vídeo em: {os.path.abspath(VIDEO_PATH)}\n"
        "Confira se o nome está certo e se o script está sendo executado na pasta correta, "
        "ou use um caminho absoluto (ex: r'C:\\Users\\denis\\Videos\\people.mp4')."
    )

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"O OpenCV não conseguiu abrir o vídeo: {os.path.abspath(VIDEO_PATH)}\n"
        "O arquivo pode estar corrompido ou em um codec não suportado."
    )

#Tracker
tracker = sv.ByteTrack()

#Line Position
LINE_Y = 180

up_count = 0
down_count = 0

#Store previous position (última posição Y conhecida de cada track_id)
track_history = {}

# NOVO: Dicionário para armazenar informações completas de cada pessoa
people_data = {}
# Estrutura: {
#   track_id: {
#       'start_frame': int,
#       'start_position': (x, y),
#       'start_time': timestamp,
#       'end_frame': int,
#       'end_position': (x, y),
#       'end_time': timestamp,
#       'trajectory': [(x1,y1), (x2,y2), ...],  # lista de todas as posições
#       'direction': 'IN' or 'OUT' or 'UNKNOWN',
#       'crossed_line': False
#   }
# }

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # vídeo acabou (ou erro de leitura de um frame específico)
    
    frame_count += 1

    results = model(frame, classes=[0], verbose=False)[0]  # only person class
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 0, 255),
        2
    )

    # Lista de IDs ativos neste frame para verificar quem saiu
    active_ids = set()

    for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
        if track_id is None:
            continue

        active_ids.add(track_id)
        
        x1, y1, x2, y2 = bbox.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # NOVO: Registrar início da trajetória se for a primeira vez que vemos este ID
        if track_id not in track_history:
            track_history[track_id] = cy
            people_data[track_id] = {
                'start_frame': frame_count,
                'start_position': (cx, cy),
                'start_time': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                'end_frame': None,
                'end_position': None,
                'end_time': None,
                'trajectory': [(cx, cy)],
                'direction': 'UNKNOWN',
                'crossed_line': False
            }
            print(f"[{frame_count}] Nova pessoa detectada - ID {track_id} começou em ({cx}, {cy})")
        else:
            # Adicionar posição à trajetória
            people_data[track_id]['trajectory'].append((cx, cy))

        prev_y = track_history[track_id]

        #Moving Down (IN)
        if prev_y < LINE_Y and cy >= LINE_Y:
            down_count += 1
            people_data[track_id]['direction'] = 'IN'
            people_data[track_id]['crossed_line'] = True
            print(f"[{frame_count}] ID {track_id} cruzou a linha para BAIXO (IN) - posição: ({cx}, {cy})")
            
        #Moving Up (OUT)
        elif prev_y > LINE_Y and cy <= LINE_Y:
            up_count += 1
            people_data[track_id]['direction'] = 'OUT'
            people_data[track_id]['crossed_line'] = True
            print(f"[{frame_count}] ID {track_id} cruzou a linha para CIMA (OUT) - posição: ({cx}, {cy})")

        track_history[track_id] = cy

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # NOVO: Mostrar a trajetória da pessoa (últimos 30 pontos)
        trajectory = people_data[track_id]['trajectory']
        if len(trajectory) > 1:
            # Desenhar os últimos 30 pontos da trajetória
            points_to_draw = trajectory[-30:]
            for i in range(1, len(points_to_draw)):
                cv2.line(frame, points_to_draw[i-1], points_to_draw[i], (255, 0, 0), 2)
        
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # NOVO: Verificar IDs que não estão mais ativos (pessoas que saíram do frame)
    for track_id in list(people_data.keys()):
        if track_id not in active_ids and people_data[track_id]['end_frame'] is None:
            # A pessoa não está mais no frame
            last_position = people_data[track_id]['trajectory'][-1] if people_data[track_id]['trajectory'] else None
            people_data[track_id]['end_frame'] = frame_count
            people_data[track_id]['end_position'] = last_position
            people_data[track_id]['end_time'] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            print(f"[{frame_count}] ID {track_id} sumiu do frame - última posição: {last_position}")
            print(f"  → Início: frame {people_data[track_id]['start_frame']}, posição {people_data[track_id]['start_position']}")
            print(f"  → Fim: frame {people_data[track_id]['end_frame']}, posição {people_data[track_id]['end_position']}")
            print(f"  → Direção: {people_data[track_id]['direction']}")
            print(f"  → Duração: {people_data[track_id]['end_frame'] - people_data[track_id]['start_frame']} frames")
            print("-" * 50)

    # ---- Dashboard (agora dentro do loop, atualiza a cada frame) ----
    dashboard = np.ones((800, 1000, 3), dtype=np.uint8) * 255

    #Header
    cv2.rectangle(dashboard, (0, 0), (1000, 80), (0, 120, 0), -1)
    cv2.putText(
        dashboard,
        "People Counter",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 255, 255),
        4
    )

    # NOVO: Mostrar total de pessoas detectadas
    cv2.putText(
        dashboard,
        f"Total Pessoas: {len(people_data)}",
        (800, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    #Resize frame to fit dashboard
    frame_resized = cv2.resize(frame, (900, 500))
    #put video inside dashboard
    dashboard[90:590, 50:950] = frame_resized

    #Out Panel
    cv2.rectangle(dashboard, (50, 620), (450, 780), (0, 0, 180), -1)
    cv2.putText(
        dashboard, "OUT", (70, 715),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4
    )
    cv2.putText(
        dashboard, str(up_count), (270, 730),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6
    )

    #IN Panel
    cv2.rectangle(dashboard, (550, 620), (950, 780), (0, 120, 0), -1)
    cv2.putText(
        dashboard, "IN", (600, 715),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4
    )
    cv2.putText(
        dashboard, str(down_count), (770, 730),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6
    )

    cv2.imshow("Dashboard", dashboard)

    # pressione 'q' para sair a qualquer momento
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("RELATÓRIO FINAL - TRAJETÓRIA DAS PESSOAS")
print("="*60)

# NOVO: Relatório final detalhado
for track_id, data in people_data.items():
    print(f"\nID {track_id}:")
    print(f"  Início: frame {data['start_frame']}, posição {data['start_position']}, hora {data['start_time']}")
    print(f"  Fim: frame {data['end_frame']}, posição {data['end_position']}, hora {data['end_time']}")
    print(f"  Direção: {data['direction']}")
    print(f"  Cruzou a linha: {'Sim' if data['crossed_line'] else 'Não'}")
    print(f"  Total de frames visíveis: {data['end_frame'] - data['start_frame'] if data['end_frame'] else 'Em andamento'}")
    print(f"  Tamanho da trajetória: {len(data['trajectory'])} pontos")
    print("-" * 40)

print(f"\nResumo:")
print(f"Total IN (down): {down_count}")
print(f"Total OUT (up): {up_count}")
print(f"Total de pessoas detectadas: {len(people_data)}")

# NOVO: Salvar os dados em um arquivo CSV para análise posterior
import csv
with open('people_trajectory_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['track_id', 'start_frame', 'start_x', 'start_y', 'start_time', 
                  'end_frame', 'end_x', 'end_y', 'end_time', 'direction', 
                  'crossed_line', 'total_frames']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for track_id, data in people_data.items():
        writer.writerow({
            'track_id': track_id,
            'start_frame': data['start_frame'],
            'start_x': data['start_position'][0],
            'start_y': data['start_position'][1],
            'start_time': data['start_time'],
            'end_frame': data['end_frame'],
            'end_x': data['end_position'][0] if data['end_position'] else None,
            'end_y': data['end_position'][1] if data['end_position'] else None,
            'end_time': data['end_time'],
            'direction': data['direction'],
            'crossed_line': data['crossed_line'],
            'total_frames': data['end_frame'] - data['start_frame'] if data['end_frame'] else None
        })

print("\nDados salvos em 'people_trajectory_data.csv'")