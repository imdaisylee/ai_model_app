from app import app1
import os

if __name__ == "__main__":
    app1.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))