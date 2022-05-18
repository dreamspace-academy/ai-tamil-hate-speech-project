
from fastapi import FastAPI
import uvicorn

# fast api instance
app = FastAPI()


@app.post("/OmdenaNYU")
def detect_hate():
            return {"Test Succeeded"}



if __name__ == "__main__":
            uvicorn.run(app)
