from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import CropData, CropPredictor
from src.pipline.training_pipeline import TrainPipeline


# ==========================================
# Initialize FastAPI App
# ==========================================

app = FastAPI(title="KrishiLink Crop Recommendation System")

# Static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# ==========================================
# CORS Configuration
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Form Data Class
# ==========================================

class DataForm:

    def __init__(self, request: Request):

        self.request = request

        self.N: Optional[int] = None
        self.P: Optional[int] = None
        self.K: Optional[int] = None

        self.temperature: Optional[float] = None
        self.humidity: Optional[float] = None
        self.ph: Optional[float] = None
        self.rainfall: Optional[float] = None

    async def get_crop_data(self):

        form = await self.request.form()

        self.N = int(form.get("N"))
        self.P = int(form.get("P"))
        self.K = int(form.get("K"))

        self.temperature = float(form.get("temperature"))
        self.humidity = float(form.get("humidity"))
        self.ph = float(form.get("ph"))
        self.rainfall = float(form.get("rainfall"))


# ==========================================
# Home Page
# ==========================================

@app.get("/", response_class=Response)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None
        }
    )


# ==========================================
# Train Model Endpoint
# ==========================================

@app.get("/train")
async def train_model():

    try:

        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response(
            content="Training completed successfully!",
            media_type="text/plain"
        )

    except Exception as e:

        return Response(
            content=f"Training Failed\n\n{e}",
            media_type="text/plain"
        )


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/")
async def predict(request: Request):

    try:

        # Read form values
        form = DataForm(request)

        await form.get_crop_data()

        # Create CropData object
        crop_data = CropData(

            N=form.N,
            P=form.P,
            K=form.K,
            temperature=form.temperature,
            humidity=form.humidity,
            ph=form.ph,
            rainfall=form.rainfall

        )

        # Convert to DataFrame
        crop_df = crop_data.get_crop_input_dataframe()

        # Load model and predict
        predictor = CropPredictor()

        prediction = predictor.predict(crop_df)

        recommended_crop = prediction[0]

        # Render result
        return templates.TemplateResponse(

            "index.html",

            {

                "request": request,

                "prediction": recommended_crop

            }

        )

    except Exception as e:

        return templates.TemplateResponse(

            "index.html",

            {

                "request": request,

                "prediction": None,

                "error": str(e)

            }

        )


# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":

    app_run(

        app,

        host=APP_HOST,

        port=APP_PORT

    )