import gradio as gr
from predict import predict_video

def infer(video):
    result = predict_video(video, sequence_length=10, frame_size=128, segments=7, models_dir='models')
    return f"Prediction: {result['prediction']}\nConfidence: {result['confidence']:.4f}\nProbs: {result['probs']}\nModel: {result['model_path']}"

with gr.Blocks() as demo:
    gr.Markdown("# Deepfake Detector")
    gr.Markdown("Upload a short video clip to detect if it's Real or Fake.")
    with gr.Row():
        video_in = gr.Video(label="Input video")
    btn = gr.Button("Analyze")
    out = gr.Textbox(lines=5, label="Result")
    btn.click(fn=infer, inputs=video_in, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
