# Chat With an Image

I tried doing this with Ollama, but it does not handle the conversational piece of thi well so I abandoned it.

This is a Streamlit application that allows users to ask questions about an uploaded image and receive responses from a conversational AI agent. The agent uses the OpenAI GPT-3.5 Turbo model to generate answers based on the provided image and user input.

## installation

1. Clone the repository:

        git clone https://github.com/adamcodes716/chatbot-image
        
2. Create a Virtual environment in your project:

        Windows venv:
        ```
        python -m venv .env-windows
        .env-windows\Scripts\activate.bat
        ```

        Linux venv:
        ```
        python3 -m venv .env-linux
        source .env-linux/bin/activate
        ```
        
3. Install the required dependencies:

        pip install -r requirements.txt

4. Obtain an **OpenAI API key**. You can sign up for an API key at [OpenAI](https://platform.openai.com).

5. Replace the placeholder API key in the main.py file with your actual OpenAI API key:

        llm = ChatOpenAI(
            openai_api_key='YOUR_API_KEY',
            temperature=0,
            model_name="gpt-3.5-turbo"
        )

6. Run the Streamlit application:

        streamlit run src/main.py

        or if you want to specify the port:
      
        streamlit run src/main.py --server.port 10015

7. Open your web browser and go to http://localhost:8501 to access the application.

## usage

1. Upload an image by clicking the file upload button.

2. The uploaded image will be displayed.

3. Enter a question about the image in the text input field.

4. The conversational AI agent will generate a response based on the provided question and image.

5. The response will be displayed below the question input.

## tools

The application utilizes the following custom tools:

- **ImageCaptionTool**: Generates a textual caption for the uploaded image.
- **ObjectDetectionTool**: Performs object detection on the uploaded image and identifies the objects present.

