from flask import Flask, render_template, request, redirect, url_for, session
import subprocess
import sys
import os

app = Flask(__name__)
app.secret_key = "temporary_secret_key"  # Needed for session usage

# === CONFIG ===
COMMENT_LOADER = "Comment_loader.py"
EMBED_CLUSTER = "Clusterize.py"


def predict(text: str) -> int:
    if not text or not text.strip():
        return 0
    return 1
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form.get("video_url")

        if not video_url:
            session["message"] = "❌ Please enter a YouTube video URL"
            return redirect(url_for("index"))

        try:
            # 1️⃣ Run Comment_loader.py
            subprocess.run(
                [sys.executable, COMMENT_LOADER, video_url],
                check=True
            )

            # 2️⃣ Run Clusterize.py
            subprocess.run(
                [sys.executable, EMBED_CLUSTER],
                check=True
            )

            # Store status message in session and redirect to clear POST state
            session["message"] = "✅ Success! Comments processed and plot generated."
            session["plot_ready"] = True
            return redirect(url_for("index"))

        except subprocess.CalledProcessError as e:
            session["message"] = f"❌ Error: {e}"
            session["plot_ready"] = False
            return redirect(url_for("index"))

    # === GET request ===
    message = session.pop("message", None)
    plot_ready = session.pop("plot_ready", False)
    return render_template("index.html", message=message, plot_ready=plot_ready)


@app.route("/plot")
def show_plot():
    plot_path = os.path.join("static", "plot.html")
    if os.path.exists(plot_path):
        return redirect("/static/plot.html")
    else:
        return "Plot not found. Please process a video first."

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.mkdir("static")

    app.run(host="0.0.0.0", port=5000, debug=True)
