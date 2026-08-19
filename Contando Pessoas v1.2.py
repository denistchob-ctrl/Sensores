#Vídeo CONTANDO PESSOAS.mp4
from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
import os
from datetime import datetime
import csv

# ===================== FUNÇÕES AUXILIARES =====================

def criar_painel_informacao(dashboard, x, y, largura, altura, 
                           cor_fundo, texto, valor, 
                           cor_texto=(255, 255, 255), 
                           tamanho_texto=0.9, tamanho_valor=1.3):
    """Cria um painel com retângulo e texto + valor"""
    cv2.rectangle(dashboard, (x, y), (x + largura, y + altura), cor_fundo, -1)
    
    espacamento = 10
    largura_texto = cv2.getTextSize(texto + ":", cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, 2)[0][0]
    
    pos_x_texto = x + espacamento
    pos_y_texto = y + (altura // 2) + 5
    
    pos_x_valor = x + espacamento + largura_texto + 5
    pos_y_valor = y + (altura // 2) + 8
    
    cv2.putText(dashboard, f"{texto}:", (pos_x_texto, pos_y_texto),
                cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, cor_texto, 2)
    cv2.putText(dashboard, str(valor), (pos_x_valor, pos_y_valor),
                cv2.FONT_HERSHEY_SIMPLEX, tamanho_valor, cor_texto, 3)

def criar_barra_superior(dashboard, largura, altura_barra, 
                         titulo, total_pessoas, 
                         contador1_label, contador1_valor,
                         contador2_label, contador2_valor,
                         cor_fundo=(0, 120, 0)):
    """Cria a barra superior com todas as informações"""
    cv2.rectangle(dashboard, (0, 0), (largura, altura_barra), cor_fundo, -1)
    
    cv2.putText(dashboard, titulo, (20, altura_barra // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(dashboard, f"TOTAL: {total_pessoas}", (350, altura_barra // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    criar_painel_informacao(dashboard, 550, 10, 250, 60,
                           (0, 0, 180), contador1_label, contador1_valor)
    criar_painel_informacao(dashboard, 840, 10, 250, 60,
                           (110, 120, 0), contador2_label, contador2_valor)

def adicionar_rodape(dashboard, largura, altura, frame_count):
    """Adiciona informações no rodapé do dashboard"""
    cv2.putText(dashboard, f"Frame: {frame_count} | Pressione 'q' para sair",
                (10, altura - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100, 100, 100), 1)

def desenhar_trajetoria(frame, trajectory, cor=(255, 0, 0), ultimos_pontos=30):
    """Desenha a trajetória da pessoa no frame"""
    if len(trajectory) > 1:
        points_to_draw = trajectory[-ultimos_pontos:]
        for i in range(1, len(points_to_draw)):
            cv2.line(frame, points_to_draw[i-1], points_to_draw[i], cor, 2)

# ===================== CONFIGURAÇÕES INICIAIS =====================

model = YOLO("yolov8n.pt")

vPadrao = "VERTICAL"  # "HORIZONTAL" ou "VERTICAL"
#vPadrao = "HORIZONTAL"  # "HORIZONTAL" ou "VERTICAL"
if vPadrao == "HORIZONTAL":
    VIDEO_PATH = "people.mp4"
    LINE_Y = 300
    LINE_X = 0
else:
    VIDEO_PATH = "paulista.mp4"
    LINE_Y = 0
    LINE_X = 650

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

tracker = sv.ByteTrack()

up_count = 0
down_count = 0

track_history = {}
people_data = {}

frame_count = 0

# ===== CONFIGURAÇÕES DO DASHBOARD - DEFINIDAS ANTES DO LOOP =====
BARRA_ALTURA = 80
MARGEM = 10
DASHBOARD_WIDTH = 1200
DASHBOARD_HEIGHT = 700

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    results = model(frame, classes=[0], verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Desenha a linha de referência
    if vPadrao == "HORIZONTAL":
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)
    else:
        cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 0, 255), 2)

    active_ids = set()

    for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
        if track_id is None:
            continue

        active_ids.add(track_id)
        
        x1, y1, x2, y2 = bbox.astype(int)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

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
        else:
            people_data[track_id]['trajectory'].append((cx, cy))

        # Lógica de contagem
        if vPadrao == "HORIZONTAL":
            prev_y = track_history[track_id]
            if prev_y < LINE_Y and cy >= LINE_Y:
                down_count += 1
                people_data[track_id]['direction'] = 'IN'
                people_data[track_id]['crossed_line'] = True
            elif prev_y > LINE_Y and cy <= LINE_Y:
                up_count += 1
                people_data[track_id]['direction'] = 'OUT'
                people_data[track_id]['crossed_line'] = True
            track_history[track_id] = cy
        else:
            prev_x = track_history[track_id]
            if prev_x < LINE_X and cx >= LINE_X:
                down_count += 1
                people_data[track_id]['direction'] = 'IN'
                people_data[track_id]['crossed_line'] = True
            elif prev_x > LINE_X and cx <= LINE_X:
                up_count += 1
                people_data[track_id]['direction'] = 'OUT'
                people_data[track_id]['crossed_line'] = True
            track_history[track_id] = cx

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        desenhar_trajetoria(frame, people_data[track_id]['trajectory'])
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Verifica pessoas que saíram
    for track_id in list(people_data.keys()):
        if track_id not in active_ids and people_data[track_id]['end_frame'] is None:
            last_position = people_data[track_id]['trajectory'][-1] if people_data[track_id]['trajectory'] else None
            people_data[track_id]['end_frame'] = frame_count
            people_data[track_id]['end_position'] = last_position
            people_data[track_id]['end_time'] = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # ===== CRIAÇÃO DO DASHBOARD COM TAMANHO DINÂMICO =====
    
    # Pega as dimensões originais do vídeo
    frame_height_original = frame.shape[0]
    frame_width_original = frame.shape[1]
    
    # Calcula o espaço disponível para o vídeo
    espaco_disponivel_altura = DASHBOARD_HEIGHT - BARRA_ALTURA - (MARGEM * 2)
    espaco_disponivel_largura = DASHBOARD_WIDTH - (MARGEM * 2)
    
    # Calcula a proporção original
    proporcao_original = frame_width_original / frame_height_original
    
    # Calcula novas dimensões mantendo a proporção
    nova_largura = espaco_disponivel_largura
    nova_altura = int(nova_largura / proporcao_original)
    
    # Se a altura ultrapassar o espaço disponível, ajusta pela altura
    if nova_altura > espaco_disponivel_altura:
        nova_altura = espaco_disponivel_altura
        nova_largura = int(nova_altura * proporcao_original)
    
    # Redimensiona mantendo a proporção
    frame_resized = cv2.resize(frame, (nova_largura, nova_altura))
    
    # Centraliza o vídeo no dashboard
    y_offset = BARRA_ALTURA + MARGEM + (espaco_disponivel_altura - nova_altura) // 2
    x_offset = MARGEM + (espaco_disponivel_largura - nova_largura) // 2
    
    # Cria o dashboard
    dashboard = np.ones((DASHBOARD_HEIGHT, DASHBOARD_WIDTH, 3), dtype=np.uint8) * 255
    
    # Insere o vídeo no dashboard
    dashboard[y_offset:y_offset + nova_altura, x_offset:x_offset + nova_largura] = frame_resized
    
    # Define os textos conforme orientação
    if vPadrao == "HORIZONTAL":
        txtOUT = "SAÍDA"
        txtIN = "ENTRADA"
    else:
        txtOUT = "ESQUERDA"
        txtIN = "DIREITA"
    
    # Cria a barra superior
    criar_barra_superior(
        dashboard,
        DASHBOARD_WIDTH,
        BARRA_ALTURA,
        "CONTADOR",
        len(people_data),
        txtOUT, up_count,
        txtIN, down_count
    )
    
    # Adiciona rodapé
    adicionar_rodape(dashboard, DASHBOARD_WIDTH, DASHBOARD_HEIGHT, frame_count)
    
    cv2.imshow("Contador de Pessoas", dashboard)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===================== RELATÓRIO FINAL =====================
print("\n" + "="*60)
print("RELATÓRIO FINAL - TRAJETÓRIA DAS PESSOAS")
print("="*60)

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
if vPadrao == "HORIZONTAL":
    print(f"Total ENTRADAS: {down_count}")
    print(f"Total SAÍDAS: {up_count}")
else:
    print(f"Total DIREITA: {down_count}")
    print(f"Total ESQUERDA: {up_count}")
print(f"Total de pessoas detectadas: {len(people_data)}")

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