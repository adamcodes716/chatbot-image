from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import openai
import base64


def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    inputs = processor(image, return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def detect_objects(image_path):
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
    """
    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections


def extract_text_from_image(image_path):
    """
    Extracts text from an image using OpenAI GPT-4 Vision (current model).
    """
    import openai
    import os
    import base64

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return "OpenAI API key not found."

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "What text do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    image_path = '/home/phillip/Desktop/todays_tutorial/52_langchain_ask_questions_video/code/test.jpg'
    detections = detect_objects(image_path)
    print(detections)
