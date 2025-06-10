from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llm import query_data


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("interface.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "Aucun prompt fourni."}), 400
    response = query_data(user_prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
