import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import tensorflow as tf

# Allow running this file directly: add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.frame_extractor import extract_frames
from utils.face_detection import crop_primary_face
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = str(ROOT_DIR / "uploads")
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{ROOT_DIR / 'app' / 'app.db'}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    db = SQLAlchemy(app)
    login_manager = LoginManager(app)
    login_manager.login_view = "login"

    class User(db.Model, UserMixin):
        id = db.Column(db.Integer, primary_key=True)
        email = db.Column(db.String(255), unique=True, nullable=False)
        password_hash = db.Column(db.String(255), nullable=False)
        is_admin = db.Column(db.Boolean, default=False, nullable=False)
        created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

        def set_password(self, password: str) -> None:
            self.password_hash = generate_password_hash(password)

        def check_password(self, password: str) -> bool:
            return check_password_hash(self.password_hash, password)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        db.create_all()
        
        # Create first admin user if none exists
        if not User.query.filter_by(is_admin=True).first():
            admin_email = "admin@deepfake-detector.com"
            admin_password = "admin123"  # Change this in production!
            
            if not User.query.filter_by(email=admin_email).first():
                admin_user = User(email=admin_email, is_admin=True)
                admin_user.set_password(admin_password)
                db.session.add(admin_user)
                db.session.commit()
                print(f"Created admin user: {admin_email} / {admin_password}")

    candidate_models = [
        ROOT_DIR / "models" / "model_efficient.keras",
        ROOT_DIR / "models" / "model_efficient_final.keras",
        ROOT_DIR / "models" / "model.keras",
        ROOT_DIR / "models" / "model_final.keras",
    ]
    loaded_model = None
    for p in candidate_models:
        if p.exists():
            try:
                loaded_model = tf.keras.models.load_model(
                    p,
                    custom_objects={"preprocess_input": preprocess_input},
                    safe_mode=False,
                    compile=True,
                )
                break
            except Exception:
                # Try without custom_objects as a fallback
                try:
                    loaded_model = tf.keras.models.load_model(p)
                    break
                except Exception:
                    continue

    @app.route("/", methods=["GET"]) 
    def home():
        return render_template("home.html")

    @app.route("/upload", methods=["GET", "POST"]) 
    @login_required
    def upload():
        if request.method == "POST":
            file = request.files.get("video")
            if not file or file.filename == "":
                flash("Please select a video file.", "error")
                return redirect(url_for("upload"))
            save_path = Path(app.config["UPLOAD_FOLDER"]) / file.filename
            file.save(save_path)

            frames = extract_frames(str(save_path), max_frames=32, resize=(224, 224))
            if not frames:
                return render_template("result.html", prediction=None, confidence=None, error="Could not read frames from the video.")

            crops = [crop_primary_face(f, target_size=(160, 160)) for f in frames]
            seq = np.stack(crops).astype("float32") / 255.0  # (T,160,160,3)
            seq = np.expand_dims(seq, axis=0)  # (1,T,H,W,C)

            if loaded_model is None:
                return render_template("result.html", prediction=None, confidence=None, error="Model not found. Train the model first.")

            prob_fake = float(loaded_model.predict(seq, verbose=0)[0][0])
            prob_real = 1.0 - prob_fake
            pred_label = "Fake" if prob_fake >= 0.5 else "Real"
            confidence = prob_fake if pred_label == "Fake" else prob_real

            return render_template("result.html", prediction=pred_label, confidence=confidence)

        return render_template("upload.html")

    @app.route("/register", methods=["GET", "POST"]) 
    def register():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if not email or not password:
                flash("Email and password are required.", "error")
                return redirect(url_for("register"))
            if User.query.filter_by(email=email).first():
                flash("Email already registered.", "error")
                return redirect(url_for("register"))
            user = User(email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"]) 
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for("upload"))
            flash("Invalid credentials.", "error")
            return redirect(url_for("login"))
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("home"))

    @app.route("/analytics")
    def analytics():
        # Load training history if available
        history_path = ROOT_DIR / "outputs" / "history.csv"
        history = []
        if history_path.exists():
            import csv as _csv
            with open(history_path, "r") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    history.append({
                        "epoch": int(row.get("epoch", 0)),
                        "accuracy": float(row.get("accuracy", 0) or 0),
                        "val_accuracy": float(row.get("val_accuracy", 0) or 0),
                        "loss": float(row.get("loss", 0) or 0),
                        "val_loss": float(row.get("val_loss", 0) or 0),
                    })
        # Load classification report text if exists
        report_path = ROOT_DIR / "outputs" / "classification_report.txt"
        report_text = report_path.read_text() if report_path.exists() else None
        return render_template("analytics.html", history=history, report_text=report_text)

    # Admin helper functions
    def admin_required(f):
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or not current_user.is_admin:
                flash("Admin access required.", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated_function

    @app.route("/admin")
    @admin_required
    def admin_dashboard():
        # Get user statistics
        total_users = User.query.count()
        admin_users = User.query.filter_by(is_admin=True).count()
        regular_users = total_users - admin_users
        
        # Get recent users (last 10)
        recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
        
        return render_template("admin/dashboard.html", 
                             total_users=total_users,
                             admin_users=admin_users,
                             regular_users=regular_users,
                             recent_users=recent_users)

    @app.route("/admin/users")
    @admin_required
    def admin_users():
        page = request.args.get('page', 1, type=int)
        per_page = 10
        users = User.query.paginate(page=page, per_page=per_page, error_out=False)
        return render_template("admin/users.html", users=users)

    @app.route("/admin/users/<int:user_id>/toggle_admin", methods=["POST"])
    @admin_required
    def toggle_admin(user_id):
        user = User.query.get_or_404(user_id)
        if user.id == current_user.id:
            flash("Cannot modify your own admin status.", "error")
            return redirect(url_for("admin_users"))
        
        user.is_admin = not user.is_admin
        db.session.commit()
        
        status = "admin" if user.is_admin else "regular user"
        flash(f"User {user.email} is now a {status}.", "success")
        return redirect(url_for("admin_users"))

    @app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
    @admin_required
    def delete_user(user_id):
        user = User.query.get_or_404(user_id)
        if user.id == current_user.id:
            flash("Cannot delete your own account.", "error")
            return redirect(url_for("admin_users"))
        
        db.session.delete(user)
        db.session.commit()
        flash(f"User {user.email} has been deleted.", "success")
        return redirect(url_for("admin_users"))

    @app.route("/admin/analytics")
    @admin_required
    def admin_analytics():
        # Load training history if available
        history_path = ROOT_DIR / "outputs" / "history.csv"
        history = []
        if history_path.exists():
            import csv as _csv
            with open(history_path, "r") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    history.append({
                        "epoch": int(row.get("epoch", 0)),
                        "accuracy": float(row.get("accuracy", 0) or 0),
                        "val_accuracy": float(row.get("val_accuracy", 0) or 0),
                        "loss": float(row.get("loss", 0) or 0),
                        "val_loss": float(row.get("val_loss", 0) or 0),
                    })
        
        # Load classification report and parse metrics
        report_path = ROOT_DIR / "outputs" / "classification_report.txt"
        report_text = report_path.read_text() if report_path.exists() else None
        
        # Parse classification report for detailed metrics
        metrics = {}
        if report_text:
            lines = report_text.strip().split('\n')
            for line in lines:
                if 'real' in line and 'fake' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        metrics['real_precision'] = float(parts[1])
                        metrics['real_recall'] = float(parts[2])
                        metrics['real_f1'] = float(parts[3])
                        metrics['real_support'] = int(parts[4])
                        metrics['fake_precision'] = float(parts[5])
                        metrics['fake_recall'] = float(parts[6])
                        metrics['fake_f1'] = float(parts[7])
                        metrics['fake_support'] = int(parts[8])
                elif 'accuracy' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics['accuracy'] = float(parts[1])
                elif 'macro avg' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        metrics['macro_avg_precision'] = float(parts[2])
                        metrics['macro_avg_recall'] = float(parts[3])
                        metrics['macro_avg_f1'] = float(parts[4])
                elif 'weighted avg' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        metrics['weighted_avg_precision'] = float(parts[2])
                        metrics['weighted_avg_recall'] = float(parts[3])
                        metrics['weighted_avg_f1'] = float(parts[4])
        
        # Load additional metrics
        metrics_path = ROOT_DIR / "outputs" / "metrics.txt"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                for line in f:
                    if 'val_loss=' in line:
                        metrics['val_loss'] = float(line.split('=')[1])
                    elif 'val_accuracy=' in line:
                        metrics['val_accuracy'] = float(line.split('=')[1])
        
        # Load manifest for dataset statistics
        manifest_path = ROOT_DIR / "outputs" / "manifest.csv"
        dataset_stats = {"total_samples": 0, "real_samples": 0, "fake_samples": 0}
        if manifest_path.exists():
            import csv as _csv
            with open(manifest_path, "r") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    dataset_stats["total_samples"] += 1
                    if row.get("label") == "0":
                        dataset_stats["real_samples"] += 1
                    else:
                        dataset_stats["fake_samples"] += 1
        
        # Get user statistics for analytics
        total_users = User.query.count()
        admin_users = User.query.filter_by(is_admin=True).count()
        
        # Calculate model performance summary
        performance_summary = {
            "overall_accuracy": metrics.get('accuracy', 0),
            "best_epoch": len(history) if history else 0,
            "final_val_accuracy": history[-1]['val_accuracy'] if history else 0,
            "final_val_loss": history[-1]['val_loss'] if history else 0,
            "training_samples": dataset_stats["total_samples"],
            "real_samples": dataset_stats["real_samples"],
            "fake_samples": dataset_stats["fake_samples"]
        }
        
        return render_template("admin/analytics.html", 
                             history=history, 
                             report_text=report_text,
                             metrics=metrics,
                             dataset_stats=dataset_stats,
                             performance_summary=performance_summary,
                             total_users=total_users,
                             admin_users=admin_users)

    @app.route("/admin/confusion_matrix")
    @admin_required
    def confusion_matrix_image():
        confusion_matrix_path = ROOT_DIR / "outputs" / "confusion_matrix.png"
        if confusion_matrix_path.exists():
            return send_file(str(confusion_matrix_path), mimetype='image/png')
        else:
            return "Confusion matrix not found", 404

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


