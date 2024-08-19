# Medical-Report-Summarization# Configure MongoDB connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/reportdb"
mongo = PyMongo(app)

# Initialize HuggingFace model
HUGGINGFACEHUB_API_TOKEN = "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"
docker build -t medsumai.azurecr.io/medscribeai:latest . 