from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from functions import extract_text_from_image


class ImageCaptionTool(BaseTool):
    name: str = "image_captioner"  # No spaces
    description: str = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)
        print("DEBUG: Caption generated:", caption)  # Add this line

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name: str = "object_detector"  # No spaces
    description: str = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all detected objects. Each element in the list in the format: "
        "[x1, y1, x2, y2] class_name confidence_score."
    )

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Echoes back the input string."


class OCRTool(BaseTool):
    name: str = "image_text_extractor"
    description: str = (
        "Use this tool when given the path to an image and you want to extract any visible text from it using OCR."
    )

    def _run(self, img_path):
        return extract_text_from_image(img_path)

    def _arun(self, img_path):
        raise NotImplementedError("This tool does not support async")
