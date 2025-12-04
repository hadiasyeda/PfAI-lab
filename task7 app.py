from flask import Flask, jsonify, render_template
import requests

app = Flask(__name__)

JOKE_API_URL = "https://official-joke-api.appspot.com/random_joke"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/joke", methods=["GET"])
def get_joke():
    try:
        response = requests.get(JOKE_API_URL)

        if response.status_code == 200:
            data = response.json()
            return jsonify({
                "status": "success",
                "setup": data["setup"],
                "punchline": data["punchline"]
            })
        else:
            return jsonify({"status": "error", "message": "API request failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
