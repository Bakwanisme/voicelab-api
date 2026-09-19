from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import librosa
import parselmouth
from parselmouth.praat import call
import numpy as np

# Inisialisasi API
app = FastAPI(title="Voice Lab API")

# Konfigurasi CORS agar bisa diakses dari Cloudflare Pages Anda
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti "*" dengan domain Cloudflare Anda jika sudah punya domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Voice Lab DSP Engine Active"}

# --- ENDPOINT 1: ANALISIS & KOMPARASI ---
@app.post("/api/compare")
async def compare_audio(source_file: UploadFile = File(...), target_file: UploadFile = File(...)):
    src_path = f"temp_src_{source_file.filename}"
    tgt_path = f"temp_tgt_{target_file.filename}"
    
    try:
        with open(src_path, "wb") as buffer: shutil.copyfileobj(source_file.file, buffer)
        with open(tgt_path, "wb") as buffer: shutil.copyfileobj(target_file.file, buffer)
        
        # Ekstraksi dasar (F0 & F1) menggunakan Parselmouth
        src_snd = parselmouth.Sound(src_path)
        tgt_snd = parselmouth.Sound(tgt_path)
        
        src_f0 = np.median(src_snd.to_pitch().selected_array['frequency'][src_snd.to_pitch().selected_array['frequency'] > 0])
        tgt_f0 = np.median(tgt_snd.to_pitch().selected_array['frequency'][tgt_snd.to_pitch().selected_array['frequency'] > 0])
        
        # Format respons yang dibutuhkan oleh tabel HTML di frontend
        delta_data = [
            {
                "name": "F0 Median (Pitch Center)", 
                "src": f"{round(src_f0, 2)} Hz", 
                "tgt": f"{round(tgt_f0, 2)} Hz", 
                "delta": f"{round(tgt_f0 - src_f0, 2)} Hz", 
                "isNeg": (tgt_f0 - src_f0) < 0
            },
            # Metrik prosodi dan formant bisa ditambahkan di sini dengan memanggil class analyzer utuh
        ]
        
        return {"status": "success", "data": delta_data}
    finally:
        if os.path.exists(src_path): os.remove(src_path)
        if os.path.exists(tgt_path): os.remove(tgt_path)

# --- ENDPOINT 2: SINTESIS DSP ---
@app.post("/api/process")
async def process_audio(
    source_file: UploadFile = File(...),
    pitch: float = Form(0),
    f1: float = Form(0),
    f2: float = Form(0),
    dynamics: float = Form(1.0),
    tempo: float = Form(1.0)
):
    src_path = f"temp_process_{source_file.filename}"
    out_path = "output_final.wav"
    
    try:
        with open(src_path, "wb") as buffer: shutil.copyfileobj(source_file.file, buffer)
        
        sound = parselmouth.Sound(src_path)
        
        # Konversi mono jika audio stereo
        if call(sound, "Get number of channels") > 1:
            sound = call(sound, "Convert to mono")
            
        # Ekstraksi baseline untuk DSP
        pitch_obj = sound.to_pitch()
        f0_min = pitch_obj.get_minimum() or 75
        f0_max = pitch_obj.get_maximum() or 500
        f0_median = np.median(pitch_obj.selected_array['frequency'][pitch_obj.selected_array['frequency'] > 0])
        
        # Kalkulasi Parameter Slider
        target_f0 = f0_median * (2 ** (pitch / 12.0))
        formant_ratio = 1.0 + (f1 / 100.0) # Menggunakan F1 slider sebagai rasio pergeseran utama
        
        # Eksekusi algoritma Praat
        manipulated = call(sound, "Change gender", f0_min, f0_max, formant_ratio, target_f0, dynamics, tempo)
        manipulated.save(out_path, "WAV")
        
        return FileResponse(out_path, media_type="audio/wav", filename="hasil_voice_lab.wav")
    finally:
        if os.path.exists(src_path): os.remove(src_path)
