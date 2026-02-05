import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def parse_tde_yolox_log_complete(log_file_path):
    pattern = re.compile(
        r"^(?P<timestamp>[\d-]+ [\d:]+).*? - "
        r"epoch: (?P<epoch>\d+)/(?P<max_epoch>\d+), "
        r"iter: (?P<iter>\d+)/(?P<max_iter>\d+), "
        r"gpu mem: (?P<gpu_mem>\d+)Mb, "
        r"mem: (?P<sys_mem>[\d.]+)Gb, "
        r"iter_time: (?P<iter_time>[\d.]+)s, "
        r"data_time: (?P<data_time>[\d.]+)s, "
        r"total_loss: (?P<total_loss>[\d.]+), "
        r"iou_loss: (?P<iou_loss>[\d.]+), "
        r"l1_loss: (?P<l1_loss>[\d.]+), "
        r"conf_loss: (?P<conf_loss>[\d.]+), "
        r"cls_loss: (?P<cls_loss>[\d.]+), "
        r"lr: (?P<lr>[\de.-]+), "
        r"size: (?P<size>\d+), "
        r"ETA: (?P<eta>.*)$"
    )

    data = []
    if not os.path.exists(log_file_path): return pd.DataFrame()

    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                row = match.groupdict()
                for num_field in ['epoch', 'iter', 'max_iter', 'gpu_mem', 'sys_mem', 'iter_time', 
                                  'data_time', 'total_loss', 'iou_loss', 'l1_loss', 
                                  'conf_loss', 'cls_loss', 'lr', 'size']:
                    row[num_field] = float(row[num_field])
                row['global_step'] = (row['epoch'] - 1) * row['max_iter'] + row['iter']
                data.append(row)
    return pd.DataFrame(data)

def generate_final_dashboard(df, output_name="TDE_YOLOX_Final_Research_Dashboard.html"):
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Loss Components (Total/IOU/L1)", "Learning Rate (Scheduler)",
            "TTT Performance (Conf Loss)", "Engram Performance (CLS Loss)",
            "Efficiency: Iter Time vs Data Time", "Dynamic Multi-Scale Size",
            "System RAM (Stability Monitor)", "GPU VRAM (Capacity Monitor)"
        ),
        vertical_spacing=0.07,
        horizontal_spacing=0.1
    )

    x = df['global_step']

    # 1. Losses (Combined)
    fig.add_trace(go.Scatter(x=x, y=df['total_loss'], name="Total"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df['iou_loss'], name="IOU"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df['l1_loss'], name="L1"), row=1, col=1)

    # 2. LR
    fig.add_trace(go.Scatter(x=x, y=df['lr'], name="LR", line=dict(color='yellow')), row=1, col=2)

    # 3. TTT Conf (Crucial for OOD)
    fig.add_trace(go.Scatter(x=x, y=df['conf_loss'], name="Conf (TTT)", line=dict(color='red')), row=2, col=1)

    # 4. Engram CLS (Identity)
    fig.add_trace(go.Scatter(x=x, y=df['cls_loss'], name="CLS (Engram)", line=dict(color='green')), row=2, col=2)

    # 5. Timing (Comparison) - Essential for bottlenecking
    fig.add_trace(go.Scatter(x=x, y=df['iter_time'], name="Iter Time"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df['data_time'], name="Data Time"), row=3, col=1)

    # 6. Multi-Scale Size 
    fig.add_trace(go.Scatter(x=x, y=df['size'], name="Image Size", line=dict(color='cyan')), row=3, col=2)

    # 7. Sys Mem
    fig.add_trace(go.Scatter(x=x, y=df['sys_mem'], name="Sys RAM (GB)", fill='tozeroy', line=dict(color='magenta')), row=4, col=1)

    # 8. VRAM
    fig.add_trace(go.Scatter(x=x, y=df['gpu_mem'], name="VRAM (MB)", fill='tozeroy', line=dict(color='lime')), row=4, col=2)

    fig.update_layout(
        height=1600, width=1400, 
        title_text=f"TDE-YOLOX Holistic Research Analytics (Last Update: {df['timestamp'].iloc[-1]})",
        template="plotly_dark",
        showlegend=True #
    )
    
    fig.write_html(output_name)
    print(f"Dashboard generated: {output_name}")

if __name__ == "__main__":
    LOG_PATH = "D:\\YOLOX-3rd\\TDE_log.txt" 
    df = parse_tde_yolox_log_complete(LOG_PATH)
    if not df.empty:
        generate_final_dashboard(df)