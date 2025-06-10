<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Pristini AI Chatbot</title>
  <!-- Google Fonts : Poppins -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Poppins', sans-serif;
      margin: 0;
      padding: 0;
      background: url("/static/background.jpg") no-repeat center center fixed;
      background-size: cover;
    }
    .overlay {
      background: rgba(255,255,255,0.9);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      padding-top: 30px;
    }
    .logo {
      width: 180px;
      margin-bottom: 20px;
    }
    .chat-container {
      max-width: 720px;
      width: 90%;
      background: #ffffff;
      padding: 30px;
      border-radius: 16px;
      box-shadow: 0 6px 30px rgba(0,0,0,0.2);
    }
    h1 {
      text-align: center;
      color: #003366;
      margin-bottom: 25px;
    }
    textarea {
      width: 100%;
      padding: 15px;
      border: 1px solid #d0d0d0;
      border-radius: 8px;
      margin-bottom: 20px;
      resize: vertical;
      font-size: 15px;
    }
    button {
      background: linear-gradient(45deg, #00AEEF, #00C19A);
      color: white;
      border: none;
      padding: 14px 28px;
      border-radius: 50px;
      font-size: 16px;
      cursor: pointer;
      transition: all 0.3s ease;
      display: block;
      margin: 0 auto;
    }
    button:hover {
      opacity: 0.9;
    }
    .response {
      margin-top: 30px;
      background: #f0f4f8;
      padding: 20px;
      border-radius: 8px;
      color: #333;
      line-height: 1.6;
      min-height: 80px;
    }
    .footer {
      margin-top: 40px;
      text-align: center;
      font-size: 13px;
      color: #999;
    }
    /* Loader */
    .loader {
      border: 5px solid #f3f3f3;
      border-top: 5px solid #00AEEF;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 0.9s linear infinite;
      margin: 20px auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>

<body>

  <div class="overlay">
    <img src="/static/logo.jpg" alt="Pristini Logo" class="logo">
    <div class="chat-container">
      <h1>🎓 Chatbot Pristini AI</h1>
      <textarea id="prompt" rows="4" placeholder="Posez votre question ici..."></textarea>
      <button onclick="sendPrompt()">Envoyer ma question</button>

      <div class="response" id="result"></div>
    </div>

    <div class="footer">
      © 2025 Pristini AI University 
    </div>
  </div>

  <script>
    async function sendPrompt() {
      const prompt = document.getElementById("prompt").value;
      const resultDiv = document.getElementById("result");
      resultDiv.innerHTML = "⏳ Chargement...";
      if (!prompt) {
        resultDiv.innerHTML = "<em>Merci de saisir une question avant d'envoyer.</em>";
        return;
      }

      // Loader animé
      resultDiv.innerHTML = '<div class="loader"></div>';

      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt })
      });

      const data = await response.json();

      // Affiche la réponse
      resultDiv.innerHTML = `<strong>Réponse :</strong><br>${data.response}`;
    }
  </script>

</body>
</html>
