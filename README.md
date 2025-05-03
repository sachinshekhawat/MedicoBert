###  Hosting Flask Backend & Connecting with Flutter Frontend
#### Train Model
Train the model on gpu for fast processing and store in same directory of other files as ./bertModel

#### Start Flask API  
Navigate to the directory containing `bert_mllm.py` and run:

```bash
python3 bert_mllm.py
```

#### Expose Flask Server via ngrok
Use ngrok to generate a public API URL:


```bash
ngrok http <port_number>
```

Replace <port_number> with the port your Flask app is running on (default is usually 5000).

#### Connect Flutter Frontend
Use the generated ngrok URL as the backend API endpoint in your Flutter app.
Make sure the Flask server is running and the correct API key (ngrok URL) is used.

#### API Requests
✅ POST requests are handled by the Flask server.

✅ GET requests fetch data directly from Firestore (no need for Flask server).
