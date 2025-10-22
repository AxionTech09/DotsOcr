from fastapi import FastAPI, UploadFile, File, Form
from dots_ocr.model.inference import inference_with_vllm
from PIL import Image
import io
import uvicorn

app = FastAPI(title="DOTS OCR API", version="1.0")

@app.post("/inference/")
async def run_inference(
    file: UploadFile = File(...),
    prompt: str = Form("Extract text from this image")
):
    try:
        # Read the uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run the DOTS OCR inference function
        result = inference_with_vllm(
            image=image,
            prompt=prompt,
            ip="localhost",       # assuming local inference engine
            port=8000,            # match your running vLLM or API port
            model_name="rednote-hilab/dots.ocr",
            temperature=0.1,
            top_p=0.9,
            max_completion_tokens=32768
        )

        return {"status": "success", "result": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
