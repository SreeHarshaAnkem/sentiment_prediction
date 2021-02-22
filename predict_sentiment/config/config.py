## Training
INPUT_FIELDS = ["review", "sentiment"]
TEXT_FIELD = "review"
TARGET_FIELD = "sentiment"
VOCAB_SIZE = 3000
MAX_LEN = 50
EMBEDDING_DIM =100
EPOCHS =10
BATCH_SIZE=32

## Prediction
MODEL_URL = "http://sentiment_model:8501/v1/models/sentiment:predict"