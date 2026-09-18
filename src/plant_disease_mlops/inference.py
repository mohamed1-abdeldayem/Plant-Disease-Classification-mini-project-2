from .config import settings
from .preprocessing import PlantDiseasePreprocessor
import onnxruntime as ort
from pathlib import Path
import numpy as np


class Inference:
    def __init__(self):
        self.model_path = Path(settings.base_dir_path, settings.MODEL_ONNX_PATH)
        self.preprocessor = PlantDiseasePreprocessor()
        self.session = ort.InferenceSession(self.model_path)

    def preprocess_image(self, image):
        return self.preprocessor.preprocess_image(image)

    def predict(self, image):
        session = self.session
        input_name = session.get_inputs()[0].name
        outputs = session.run(
            None,
            {input_name: image},
        )
        probabilities = outputs[0][0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        class_names = [
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___healthy",
        ]

        predicted_class = class_names[predicted_index]

        return predicted_class, confidence
