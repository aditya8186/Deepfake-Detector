# Deepfake Video Detector (CNN+LSTM)

Train a CNN+LSTM model to classify videos in `data/real` and `data/fake` as Real or Fake. Pipeline: extract frames → detect/crop faces → compute CNN features over time → LSTM classification. Flask app provides an upload UI for inference.

## Quickstart
1. Create/activate your virtual environment (already done per your setup)
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Prepare dataset (extract frames, detect faces, save sequences):
   - `python utils/preprocess_dataset.py`
4. Train (choose one):
   - Baseline CNN+LSTM: `python train.py`
   - EfficientNet+BILSTM (recommended): `EPOCHS=30 BATCH_SIZE=4 python train_eff.py`
5. Evaluate and visualize:
   - `python evaluate.py` (writes `outputs/classification_report.txt` and `outputs/confusion_matrix.png`)
6. Run Flask app:
   - `python app/app.py`

## Project Structure
```
deepfake_detector/
  app/
    static/
    templates/
    app.py
  data/
    real/
    fake/
  models/
  notebooks/
  outputs/
  utils/
    frame_extractor.py
    face_detection.py
    preprocess_dataset.py
  requirements.txt
  train.py
  train_eff.py
  evaluate.py
  README.md
```

## Notes
- Default sequence length: 32 frames, face crops resized to 160x160.
- Outputs: cached sequences in `outputs/sequences/`, manifest in `outputs/manifest.csv`, trained models in `models/`.

