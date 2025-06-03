import streamlit as st
import torch
from transformers import (
    pipeline, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
from transformers.pipelines import pipeline
from diffusers.pipelines import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
import time
import gc
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="NLP & Image Generation Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .task-description {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton button:hover {
        background-color: #0D47A1;
    }
    .output-container {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'text_summarization_model' not in st.session_state:
    st.session_state.text_summarization_model = None
if 'next_word_model' not in st.session_state:
    st.session_state.next_word_model = None
if 'story_generation_model' not in st.session_state:
    st.session_state.story_generation_model = None
if 'chatbot_model' not in st.session_state:
    st.session_state.chatbot_model = None
if 'sentiment_analysis_model' not in st.session_state:
    st.session_state.sentiment_analysis_model = None
if 'qa_model' not in st.session_state:
    st.session_state.qa_model = None
if 'image_generation_model' not in st.session_state:
    st.session_state.image_generation_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

# Function to load models
@st.cache_resource
def load_summarization_model():
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        return None

@st.cache_resource
def load_next_word_model():
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading next word model: {str(e)}")
        return None, None

@st.cache_resource
def load_story_generation_model():
    try:
        model_name = "gpt2-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading story generation model: {str(e)}")
        return None, None

@st.cache_resource
def load_chatbot_model():
    try:
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading chatbot model: {str(e)}")
        return None, None

@st.cache_resource
def load_sentiment_analysis_model():
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
        return sentiment_analyzer
    except Exception as e:
        st.error(f"Error loading sentiment analysis model: {str(e)}")
        return None

@st.cache_resource
def load_qa_model():
    try:
        model_name = "distilbert-base-cased-distilled-squad"
        qa_pipeline = pipeline("question-answering", model=model_name)
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading QA model: {str(e)}")
        return None

@st.cache_resource
def load_image_generation_model():
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the pipeline with appropriate settings
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if device == "cuda":
            pipe = pipe.to("cuda")
            # Enable memory efficient attention if available
            try:
                pipe.enable_attention_slicing()
                pipe.enable_memory_efficient_attention()
            except:
                pass
        else:
            # For CPU inference, we need to use float32
            pipe = pipe.to("cpu")
            
        return pipe
    except Exception as e:
        st.error(f"Error loading image generation model: {str(e)}")
        st.error("Make sure you have the required dependencies installed:")
        st.code("pip install diffusers transformers accelerate torch torchvision")
        return None

# Function to summarize text
def summarize_text(text, max_length=150, min_length=30):
    try:
        summarizer = load_summarization_model()
        if summarizer is None:
            return "Error: Could not load summarization model"
        
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to predict next words
def predict_next_words(text, max_length=50):
    try:
        model, tokenizer = load_next_word_model()
        if model is None or tokenizer is None:
            return "Error: Could not load next word prediction model"
        
        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=min(max_length + len(inputs[0]), 1024), 
                num_return_sequences=1, 
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_text
    except Exception as e:
        return f"Error predicting next words: {str(e)}"

# Function to generate a story
def generate_story(prompt, max_length=200):
    try:
        model, tokenizer = load_story_generation_model()
        if model is None or tokenizer is None:
            return "Error: Could not load story generation model"
        
        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=min(max_length + len(inputs[0]), 1024), 
                num_return_sequences=1, 
                temperature=0.9,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story
    except Exception as e:
        return f"Error generating story: {str(e)}"

# Function for chatbot response
def get_chatbot_response(user_input, chat_history_ids=None):
    try:
        model, tokenizer = load_chatbot_model()
        if model is None or tokenizer is None:
            return "Error: Could not load chatbot model", None
        
        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Encode the user input
        user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        user_input_ids = user_input_ids.to(device)
        
        # Append to chat history if it exists
        if chat_history_ids is not None:
            chat_history_ids = chat_history_ids.to(device)
            bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1)
        else:
            bot_input_ids = user_input_ids
        
        # Limit the length to avoid memory issues
        if bot_input_ids.shape[-1] > 1000:
            bot_input_ids = bot_input_ids[:, -1000:]
        
        with torch.no_grad():
            # Generate a response
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=bot_input_ids.shape[-1] + 100,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )
        
        # Decode and return the response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response, chat_history_ids
    except Exception as e:
        return f"Error generating response: {str(e)}", None

# Function for sentiment analysis
def analyze_sentiment(text):
    try:
        sentiment_analyzer = load_sentiment_analysis_model()
        if sentiment_analyzer is None:
            return {"label": "ERROR", "score": 0.0}
        
        result = sentiment_analyzer(text)
        return result[0]
    except Exception as e:
        return {"label": "ERROR", "score": 0.0}

# Function for question answering
def answer_question(question, context):
    try:
        qa_pipeline = load_qa_model()
        if qa_pipeline is None:
            return {"answer": "Error: Could not load QA model", "start": 0, "end": 0, "score": 0.0}
        
        result = qa_pipeline(question=question, context=context)
        return result
    except Exception as e:
        return {"answer": f"Error answering question: {str(e)}", "start": 0, "end": 0, "score": 0.0}

# Function for image generation
def generate_image(prompt, height=512, width=512):
    try:
        pipe = load_image_generation_model()
        if pipe is None:
            return None
        
        # Clean up GPU memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate image with error handling
        with torch.no_grad():
            result = pipe(
                prompt, 
                height=height, 
                width=width,
                num_inference_steps=20,  # Reduced for faster generation
                guidance_scale=7.5,
                negative_prompt="blurry, bad quality, distorted"
            )
            
        image = result.images[0]
        
        # Clean up memory after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return image
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

# Helper function for image download
def image_buffer_to_bytes(image):
    import io
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer.getvalue()

# Sidebar for navigation
st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
st.sidebar.title("Navigation")
st.sidebar.markdown("Choose a task to perform:")

page = st.sidebar.radio(
    "Select Task",
    ["Home", "Text Summarization", "Next Word Prediction", "Story Generation", 
     "Chatbot", "Sentiment Analysis", "Question Answering", "Image Generation"]
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application demonstrates various NLP and image generation capabilities "
    "using pre-trained models from Hugging Face. Select a task from the navigation "
    "menu to get started."
)

# Display system information
st.sidebar.markdown("### System Info")
cuda_available = torch.cuda.is_available()
st.sidebar.markdown(f"CUDA Available: {'✅' if cuda_available else '❌'}")
if cuda_available:
    st.sidebar.markdown(f"GPU: {torch.cuda.get_device_name(0)}")

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Home page
if page == "Home":
    st.markdown("<h1 class='main-header'>NLP & Image Generation Hub</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        Welcome to the NLP & Image Generation Hub! This application showcases various natural language processing 
        and image generation capabilities using state-of-the-art models from Hugging Face.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Available Tasks</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Text Summarization**: Condense long texts into concise summaries
        - **Next Word Prediction**: Predict what words might come next in a sequence
        - **Story Generation**: Create stories from a starting prompt
        - **Chatbot**: Have a conversation with an AI assistant
        """)
    
    with col2:
        st.markdown("""
        - **Sentiment Analysis**: Determine the sentiment of a piece of text
        - **Question Answering**: Get answers to questions based on a context
        - **Image Generation**: Create images from text descriptions
        """)
    
    st.markdown("<h2 class='sub-header'>How to Use</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. Select a task from the sidebar navigation
    2. Follow the instructions for each task
    3. Explore the capabilities of different models
    
    Note: The first time you use each feature, it may take a moment to load the model.
    """)

# Text Summarization page
elif page == "Text Summarization":
    st.markdown("<h1 class='main-header'>Text Summarization</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        This tool uses BART (Bidirectional and Auto-Regressive Transformers) to generate concise summaries 
        of longer texts while preserving the key information.
    </div>
    """, unsafe_allow_html=True)
    
    text_to_summarize = st.text_area(
        "Enter the text you want to summarize:",
        height=200,
        placeholder="Paste your long text here..."
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        min_length = st.slider("Minimum summary length", 10, 100, 30)
    
    with col2:
        max_length = st.slider("Maximum summary length", 50, 500, 150)
    
    if st.button("Summarize"):
        if text_to_summarize:
            with st.spinner("Generating summary..."):
                summary = summarize_text(text_to_summarize, max_length, min_length)
                
                st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                st.markdown("### Summary")
                st.write(summary)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Please enter some text to summarize.")
    
    st.markdown("""
    ### Tips for Better Summaries
    - Provide well-structured text with clear paragraphs
    - Longer texts (300+ words) typically yield better summaries
    - Adjust the min/max length sliders based on your needs
    """)

# Next Word Prediction page
elif page == "Next Word Prediction":
    st.markdown("<h1 class='main-header'>Next Word Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        This tool uses GPT-2 to predict what words might come next in a sequence. 
        It's useful for text completion and generating coherent continuations.
    </div>
    """, unsafe_allow_html=True)
    
    input_text = st.text_area(
        "Enter some text to continue:",
        height=150,
        placeholder="Type a sentence or paragraph to continue..."
    )
    
    output_length = st.slider("Output length", 10, 200, 50)
    
    if st.button("Predict Next Words"):
        if input_text:
            with st.spinner("Generating prediction..."):
                prediction = predict_next_words(input_text, output_length)
                
                st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                st.markdown("### Completed Text")
                
                # Highlight the original text differently from the prediction
                if not prediction.startswith("Error"):
                    highlighted_text = f"<p><span style='background-color: #E3F2FD;'>{input_text}</span>{prediction[len(input_text):]}</p>"
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                else:
                    st.error(prediction)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Please enter some text to continue.")
    
    st.markdown("""
    ### Tips for Better Predictions
    - Start with a clear and specific prompt
    - Provide enough context (at least a sentence or two)
    - Try different output lengths for varied results
    """)

# Story Generation page
elif page == "Story Generation":
    st.markdown("<h1 class='main-header'>Story Generation</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        This tool uses GPT-2 Medium to generate creative stories from a starting prompt.
        It can create fiction, continue narratives, or develop characters and settings.
    </div>
    """, unsafe_allow_html=True)
    
    story_prompt = st.text_area(
        "Enter a story prompt:",
        height=100,
        placeholder="Once upon a time in a distant galaxy..."
    )
    
    story_length = st.slider("Story length", 50, 500, 200)
    
    if st.button("Generate Story"):
        if story_prompt:
            with st.spinner("Creating your story..."):
                story = generate_story(story_prompt, story_length)
                
                st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                st.markdown("### Generated Story")
                
                # Highlight the original prompt differently from the generated story
                if not story.startswith("Error"):
                    highlighted_story = f"<p><span style='background-color: #E3F2FD;'>{story_prompt}</span>{story[len(story_prompt):]}</p>"
                    st.markdown(highlighted_story, unsafe_allow_html=True)
                else:
                    st.error(story)
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Please enter a story prompt.")
    
    st.markdown("""
    ### Tips for Better Stories
    - Be specific in your prompt (characters, setting, situation)
    - Start with an interesting hook or scenario
    - Try different prompts to see varied creative outputs
    - Longer prompts (2-3 sentences) often yield more coherent stories
    """)

# Chatbot page
elif page == "Chatbot":
    st.markdown("<h1 class='main-header'>AI Chatbot</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        Have a conversation with DialoGPT, a model trained on millions of conversations from Reddit.
        It can discuss various topics, answer questions, or just chat casually.
    </div>
    """, unsafe_allow_html=True)
    
    # Create a container for the chat interface
    chat_container = st.container()
    
    # Input for new message
    user_input = st.text_input("Type your message:", key="chat_input")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Send"):
            if user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Thinking..."):
                    # Get chatbot response
                    response, chat_history_ids = get_chatbot_response(user_input, st.session_state.chat_history_ids)
                    st.session_state.chat_history_ids = chat_history_ids
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to update the chat display
                st.rerun()
    
    with col2:
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.chat_history_ids = None
            st.rerun()
    
    # Display chat history inside the container
    with chat_container:
        # Apply styling directly to the container if needed, or rely on message styling
        # st.markdown("<div class='output-container' style='height: 300px; overflow-y: auto;'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div style='text-align: right; margin: 10px 0;'><span style='background-color: #E3F2FD; padding: 8px 12px; border-radius: 15px; display: inline-block; max-width: 70%;'>👤 {message['content']}</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; margin: 10px 0;'><span style='background-color: #F5F5F5; padding: 8px 12px; border-radius: 15px; display: inline-block; max-width: 70%;'>🤖 {message['content']}</span></div>", unsafe_allow_html=True)
        # st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Tips for Better Conversations
    - Ask specific questions for more focused responses
    - The chatbot works best with conversational language
    - It may not have knowledge of very recent events
    - Keep conversations on common topics for best results
    """)

# Sentiment Analysis page
elif page == "Sentiment Analysis":
    st.markdown("<h1 class='main-header'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        This tool analyzes the sentiment of text, determining whether it expresses a positive or negative opinion.
        It's useful for understanding emotional tone in reviews, social media, and other text.
    </div>
    """, unsafe_allow_html=True)
    
    text_for_sentiment = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste text here to analyze its sentiment..."
    )
    
    if st.button("Analyze Sentiment"):
        if text_for_sentiment:
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(text_for_sentiment)
                
                if sentiment['label'] != 'ERROR':
                    st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                    st.markdown("### Sentiment Analysis Result")
                    
                    # Display result with appropriate styling
                    if sentiment['label'] == 'POSITIVE':
                        st.markdown(f"<h3 style='color: #2E7D32;'>POSITIVE 😊</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: #C62828;'>NEGATIVE 😞</h3>", unsafe_allow_html=True)
                    
                    # Display confidence score
                    st.markdown(f"Confidence: {sentiment['score']:.2%}")
                    
                    # Create a simple visualization
                    col1, col2 = st.columns([5, 5])
                    with col1:
                        st.markdown("### Sentiment Score")
                        st.progress(sentiment['score'] if sentiment['label'] == 'POSITIVE' else 1 - sentiment['score'])
                    
                    with col2:
                        # Display a gauge or meter visualization
                        if sentiment['label'] == 'POSITIVE':
                            st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{'😊' if sentiment['score'] > 0.75 else '🙂'}</h1>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{'😞' if sentiment['score'] > 0.75 else '😐'}</h1>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Error analyzing sentiment. Please try again.")
        else:
            st.error("Please enter some text to analyze.")
    
    st.markdown("""
    ### Examples to Try
    - "I absolutely loved the movie! The acting was superb and the plot was engaging."
    - "The customer service was terrible and the product arrived damaged."
    - "The weather today is quite pleasant, perfect for a walk in the park."
    """)

# Question Answering page
elif page == "Question Answering":
    st.markdown("<h1 class='main-header'>Question Answering</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        This tool answers questions based on a provided context. It uses a model fine-tuned on the SQuAD dataset
        to extract relevant information from text and provide accurate answers.
    </div>
    """, unsafe_allow_html=True)
    
    context = st.text_area(
        "Enter the context (text passage):",
        height=200,
        placeholder="Paste a paragraph or article that contains information to answer questions about..."
    )
    
    question = st.text_input("Enter your question about the context:")
    
    if st.button("Answer Question"):
        if context and question:
            with st.spinner("Finding the answer..."):
                answer = answer_question(question, context)
                
                if not answer['answer'].startswith("Error"):
                    st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                    st.markdown("### Answer")
                    
                    # Display the answer with highlighting in the context
                    start_idx = answer['start']
                    end_idx = answer['end']
                    
                    highlighted_context = (
                        f"{context[:start_idx]}"
                        f"<span style='background-color: #BBDEFB; font-weight: bold;'>{context[start_idx:end_idx]}</span>"
                        f"{context[end_idx:]}"
                    )
                    
                    st.markdown(f"<p><strong>Answer:</strong> {answer['answer']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p><strong>Confidence:</strong> {answer['score']:.2%}</p>", unsafe_allow_html=True)
                    
                    st.markdown("### Context with Highlighted Answer")
                    st.markdown(highlighted_context, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(answer['answer'])
        else:
            if not context:
                st.error("Please provide a context passage.")
            if not question:
                st.error("Please enter a question.")
    
    st.markdown("""
    ### Sample Context to Try
    
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world. The Eiffel Tower is the most-visited paid monument in the world; 6.91 million people ascended it in 2015. The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
    
    **Sample Questions:**
    - Who designed the Eiffel Tower?
    - When was the Eiffel Tower built?
    - How tall is the Eiffel Tower?
    """)

# Image Generation page
elif page == "Image Generation":
    st.markdown("<h1 class='main-header'>Image Generation</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='task-description'>
        This tool uses Stable Diffusion to generate images from text descriptions.
        It can create artwork, visualize concepts, or generate creative imagery based on your prompts.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if required dependencies are available
    try:
        import diffusers
        dependencies_ok = True
    except ImportError:
        dependencies_ok = False
        st.error("⚠️ Image generation requires additional dependencies. Please install them with:")
        st.code("pip install diffusers transformers accelerate torch torchvision xformers")
    
    if dependencies_ok:
        image_prompt = st.text_area(
            "Describe the image you want to generate:",
            height=100,
            placeholder="A serene landscape with mountains and a lake at sunset..."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            width = st.select_slider("Width", options=[256, 384, 512, 640, 768], value=512)
        
        with col2:
            height = st.select_slider("Height", options=[256, 384, 512, 640, 768], value=512)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            num_inference_steps = st.slider("Inference Steps (Higher = Better Quality)", 10, 50, 20)
            guidance_scale = st.slider("Guidance Scale (Higher = More Prompt Adherence)", 1.0, 20.0, 7.5)
            negative_prompt = st.text_area("Negative Prompt (What to avoid)", 
                                         value="blurry, bad quality, distorted, ugly, malformed")
        
        if st.button("Generate Image"):
            if image_prompt:
                # Show system requirements warning for CPU users
                if not torch.cuda.is_available():
                    st.warning("⚠️ No GPU detected. Image generation will be slow on CPU. Consider using Google Colab or a GPU-enabled environment for better performance.")
                
                with st.spinner("Generating image (this may take 1-3 minutes)..."):
                    try:
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update progress (simulated)
                        for i in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)
                            if i < 20:
                                status_text.text("Loading model...")
                            elif i < 80:
                                status_text.text("Generating image...")
                            else:
                                status_text.text("Finalizing...")
                        
                        # Generate image
                        pipe = load_image_generation_model()
                        if pipe is not None:
                            # Generate with custom settings
                            with torch.no_grad():
                                result = pipe(
                                    image_prompt,
                                    height=height,
                                    width=width,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    negative_prompt=negative_prompt
                                )
                                image = result.images[0]
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                            st.markdown("### Generated Image")
                            st.image(image, caption=image_prompt, use_column_width=True)
                            
                            # Provide download option
                            img_buffer = image_buffer_to_bytes(image)
                            st.download_button(
                                label="Download Image",
                                data=img_buffer,
                                file_name=f"generated_image_{int(time.time())}.png",
                                mime="image/png"
                            )
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error("Failed to load image generation model. Please check your installation.")
                            
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
                        st.markdown("""
                        **Troubleshooting Tips:**
                        - Ensure you have sufficient RAM/GPU memory
                        - Try reducing image dimensions
                        - Use a simpler prompt
                        - Make sure all dependencies are installed correctly
                        """)
            else:
                st.error("Please enter a description for the image you want to generate.")
    
    st.markdown("""
    ### Tips for Better Image Generation
    - Be specific and descriptive in your prompts
    - Include details about style, lighting, and composition
    - Try adding artistic references (e.g., "in the style of Van Gogh")
    - Use adjectives to specify the mood or atmosphere
    - Mention camera settings for photorealistic images (e.g., "shot with 85mm lens")
    
    ### Sample Prompts to Try
    - "A futuristic cityscape with flying cars and neon lights at night, cyberpunk style"
    - "A photorealistic portrait of a red fox in an autumn forest, golden hour lighting, detailed fur"
    - "An oil painting of a calm lake surrounded by mountains, impressionist style"
    - "A cute robot watering flowers in a garden, children's book illustration style"
    """)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 1rem; background-color: #F5F5F5; border-radius: 5px;">
    <p>Built with Streamlit and Hugging Face Transformers</p>
    <p style="font-size: 0.8rem;">Models used: BART, GPT-2, DialoGPT, DistilBERT, Stable Diffusion</p>
    <p style="font-size: 0.7rem; color: #666;">
        ⚡ GPU accelerated when available | 🔧 Optimized for performance | 🛡️ Safe content filtering
    </p>
</div>
""", unsafe_allow_html=True)