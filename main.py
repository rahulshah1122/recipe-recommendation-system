import streamlit as st
import tensorflow as tf
import numpy as np
import os
import requests

# Set page configuration
st.set_page_config(
    page_title="Recipe Recommendation: Detect & Cook",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388e3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .recipe-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ingredient-tag {
        background-color: #c8e6c9;
        color: #2e7d32;
        padding: 5px 10px;
        border-radius: 20px;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    .recipe-title {
        color: #2e7d32;
        font-size: 1.3rem;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
    }
    .emoji-icon {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #388e3c;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f1f8e9;
    }
    .upload-section {
        border: 2px dashed #81c784;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .vegetable-info {
        padding: 10px;
        border-radius: 10px;
        background-color: #2a453b;
        margin-top: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f8e9;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #2e7d32;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c8e6c9;
        border-radius: 4px 4px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Tensorflow Model Prediction (no changes needed)
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")  # Correct path
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Vegetable nutrition database
vegetable_info = {
    "cucumber": {
        "emoji": "🥒",
        "nutrition": "High in water, low in calories, contains vitamins K, C, and potassium",
        "benefits": "Hydrating, helps with weight management, supports digestive health"
    },
    "carrot": {
        "emoji": "🥕",
        "nutrition": "Rich in beta-carotene, fiber, vitamin K1, potassium, and antioxidants",
        "benefits": "Good for eye health, lowers cholesterol, improves skin health"
    },
    "capsicum": {
        "emoji": "🫑",
        "nutrition": "Excellent source of vitamins A, C, and B6, folate and potassium",
        "benefits": "Supports immune function, promotes eye health, anti-inflammatory"
    },
    "onion": {
        "emoji": "🧅",
        "nutrition": "Contains vitamin C, folate, vitamin B6, potassium",
        "benefits": "Anti-inflammatory, antibacterial properties, heart health benefits"
    },
    "potato": {
        "emoji": "🥔",
        "nutrition": "Good source of fiber, potassium, vitamin C, and vitamin B6",
        "benefits": "Provides energy, supports digestive health, contains antioxidants"
    },
    "tomato": {
        "emoji": "🍅",
        "nutrition": "Good source of vitamin C, potassium, folate, and vitamin K",
        "benefits": "Rich in lycopene, promotes heart health, improves skin health"
    },
    # Default for other vegetables
    "default": {
        "emoji": "🥗",
        "nutrition": "Various vitamins and minerals essential for health",
        "benefits": "Part of a balanced diet, provides fiber and nutrients"
    }
}

# Get vegetable info with default fallback
def get_vegetable_info(vegetable_name):
    # Handle special cases and format inconsistencies
    vegetable_name = vegetable_name.lower()
    if vegetable_name == "bell pepper":
        vegetable_name = "capsicum"
    
    return vegetable_info.get(vegetable_name, vegetable_info["default"])

# Sidebar styling and navigation
with st.sidebar:
    st.markdown('<h1 style="color:#2e7d32;">🥦 Recipe Recommendation</h1>', unsafe_allow_html=True)
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    app_mode = st.selectbox("📋 Navigation", ["Home","Vegetable Detection", "About Project"])

# Main content
if app_mode == "Home":
    st.markdown('<h1 class="main-header">🥦 Recipe Recommendation</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Detect Vegetables & Discover Delicious Recipes</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📸 Vegetable Detection")
        st.write("Upload an image of a vegetable and our AI will identify it!")
        st.write("You'll get:")
        st.markdown("- ✅ Accurate vegetable identification")
        st.markdown("- 🍽️ Personalized recipe recommendations")
        st.markdown("- 🥗 Nutrition information and health benefits")
        st.markdown("- 🎬 Recipe videos when available")
        
        if st.button("Try it now!", key="try_detection"):
            st.session_state.app_mode = "Vegetable Detection"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        image_path = "home_img.png"  # Replace with actual path
        try:
            st.image(image_path, use_container_width=True)
        except:
            st.info("Home image not found. Please add 'home_img.png' to your app directory.")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌟 How It Works")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### 1. Upload")
        st.markdown("Take a photo of any vegetable and upload it to our app")
    with cols[1]:
        st.markdown("#### 2. AI Analysis")
        st.markdown("Our advanced AI recognizes the vegetable with high accuracy")
    with cols[2]:
        st.markdown("#### 3. Get Recipes")
        st.markdown("Discover delicious recipes using your vegetable")
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "About Project":
    st.markdown('<h1 class="main-header">About Project</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 About Dataset")
    st.write("This dataset contains images of the following food items:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fruits:**")
        st.markdown("- 🍌 Banana")
        st.markdown("- 🍎 Apple")
        st.markdown("- 🍐 Pear")
        st.markdown("- 🍇 Grapes")
        st.markdown("- 🍊 Orange")
        st.markdown("- 🥝 Kiwi")
        st.markdown("- 🍉 Watermelon")
        st.markdown("- 🍑 Pomegranate")
        st.markdown("- 🍍 Pineapple")
        st.markdown("- 🥭 Mango")
    
    with col2:
        st.markdown("**Vegetables:**")
        st.markdown("- 🥒 Cucumber, 🥕 Carrot, 🫑 Capsicum")
        st.markdown("- 🧅 Onion, 🥔 Potato, 🍋 Lemon")
        st.markdown("- 🍅 Tomato, Radish, Beetroot")
        st.markdown("- 🥬 Cabbage, Lettuce, Spinach")
        st.markdown("- Soy Bean, Cauliflower, Bell Pepper")
        st.markdown("- 🌶️ Chilli Pepper, Turnip, 🌽 Corn")
        st.markdown("- Sweet Corn, Sweet Potato, Paprika")
        st.markdown("- Jalapeño, Ginger, Garlic")
        st.markdown("- Peas, 🍆 Eggplant")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📁 Dataset Structure")
    st.markdown("This dataset contains three folders:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Training")
        st.markdown("- 100 images per class")
        st.markdown("- Used to train the AI model")
    with col2:
        st.markdown("#### Testing")
        st.markdown("- 10 images per class")
        st.markdown("- Used to evaluate the model")
    with col3:
        st.markdown("#### Validation")
        st.markdown("- 10 images per class")
        st.markdown("- Used to fine-tune the model")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Model Architecture")
    st.write("This project uses a deep learning model based on convolutional neural networks to classify vegetables with high accuracy.")
    st.markdown("The model was trained on thousands of vegetable images and can identify various types with accuracy exceeding 95%.")
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Vegetable Detection":
    st.markdown('<h1 class="main-header">Vegetable Detection & Recipes</h1>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="emoji-icon">📸</span> **Upload a Vegetable Image**', unsafe_allow_html=True)
        
        # File uploader with improved styling
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
        
        if test_image is not None:
            # Show image preview
            st.image(test_image, use_container_width=True)
            
            # Predict button with enhanced styling
            predict_col1, predict_col2 = st.columns([1, 1])
            with predict_col1:
                predict_btn = st.button("🔍 Identify Vegetable")
            with predict_col2:
                clear_btn = st.button("🗑️ Clear Identification")
                if clear_btn:
                 st.session_state.pop("predicted_vegetable", None)
                 st.session_state.pop("prediction_done", None)
                 st.session_state.pop("uploaded_file", None)  # Clear uploaded file
                 st.rerun()  # Force UI refresh
                    
        else:
            st.info("Please upload an image of a vegetable to get started")
            
            # Sample vegetables for quick testing
            st.markdown("### Try with sample vegetables:")
            sample_cols = st.columns(3)
            with sample_cols[0]:
                st.markdown("🥕 Carrot")
            with sample_cols[1]:
                st.markdown("🥦 Broccoli")
            with sample_cols[2]:
                st.markdown("🍅 Tomato")
            st.write("(Upload your own image for actual detection)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dietary preferences
        if test_image is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="emoji-icon">⚙️</span> **Customize Recommendations**', unsafe_allow_html=True)
            
            diet_pref = st.multiselect(
                "Dietary Preferences",
                ["Vegetarian", "Vegan", "Gluten-Free", "Low-Carb", "High-Protein"]
            )
            
            cooking_time = st.slider("Max Cooking Time (minutes)", 15, 60, 30)
            
            exclude_ingredients = st.text_input("Ingredients to Exclude (comma separated)")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column for results and recipes
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="emoji-icon">🍽️</span> **Results & Recommendations**', unsafe_allow_html=True)
        
        if test_image is not None and predict_btn:
            st.session_state.prediction_done = True
            
            with st.spinner("Identifying vegetable..."):
                try:
                    # Run model prediction
                    result_index = model_prediction(test_image)
                    with open("labels.txt") as f:  # Correct path
                        content = f.readlines()
                    label = [i[:-1] for i in content]
                    predicted_vegetable = label[result_index]
                    st.session_state.predicted_vegetable = predicted_vegetable
                    
                    # Get vegetable info
                    veg_info = get_vegetable_info(predicted_vegetable)
                    
                    # Display success message with styled box
                    st.markdown(f"""
                    <div class="vegetable-info">
                        <h3>✅ Detected: {predicted_vegetable.capitalize()} {veg_info['emoji']}</h3>
                        <p><b>Nutrition:</b> {veg_info['nutrition']}</p>
                        <p><b>Health Benefits:</b> {veg_info['benefits']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.snow()  # Keep the snow effect
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.warning("Please make sure your model and label files are in the correct location.")
                    if 'predicted_vegetable' in st.session_state:
                        del st.session_state.predicted_vegetable
        
        # Recipe section
        if 'predicted_vegetable' in st.session_state and 'prediction_done' in st.session_state:
            st.markdown(f"### Recipes with {st.session_state.predicted_vegetable.capitalize()} {get_vegetable_info(st.session_state.predicted_vegetable)['emoji']}")
            
            # Recipe API call
            try:
                api_url = "https://www.themealdb.com/api/json/v1/1/search.php"
                params = {"s": st.session_state.predicted_vegetable}
                
                response = requests.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data and data["meals"]:
                    # Create tabs for multiple recipes if available
                    num_recipes = min(3, len(data["meals"]))
                    recipe_tabs = st.tabs([f"Recipe {i+1}" for i in range(num_recipes)])
                    
                    for i in range(num_recipes):
                        with recipe_tabs[i]:
                            meal = data["meals"][i]
                            
                            # Recipe card with styling
                            st.markdown(f'<h3 class="recipe-title">{meal["strMeal"]}</h3>', unsafe_allow_html=True)
                            
                            # Image and details in columns
                            img_col, details_col = st.columns([1, 1])
                            
                            with img_col:
                                st.image(meal['strMealThumb'], width=300)
                            
                            with details_col:
                                # Display ingredients with tags
                                st.markdown("**Ingredients:**")
                                ingredients_html = "<div>"
                                for j in range(1, 21):
                                    ingredient = meal.get(f"strIngredient{j}")
                                    measure = meal.get(f"strMeasure{j}")
                                    if ingredient and ingredient.strip() and measure and measure.strip():
                                        ingredients_html += f'<span class="ingredient-tag">{measure.strip()} {ingredient.strip()}</span>'
                                ingredients_html += "</div>"
                                st.markdown(ingredients_html, unsafe_allow_html=True)
                                
                                # Category and area
                                st.markdown(f"**Category:** {meal.get('strCategory', 'Not specified')}")
                                st.markdown(f"**Cuisine:** {meal.get('strArea', 'Not specified')}")
                                
                                # YouTube link
                                youtube_link = meal.get('strYoutube')
                                if youtube_link:
                                    st.markdown(f"<a href='{youtube_link}' target='_blank'>📺 Watch Recipe Video</a>", unsafe_allow_html=True)
                            
                            # Instructions
                            st.markdown("**Instructions:**")
                            instructions = meal.get('strInstructions', "Instructions not available.")
                            # Format instructions as steps
                            steps = instructions.split('\r\n')
                            steps = [step for step in steps if step.strip()]
                            for idx, step in enumerate(steps):
                                st.markdown(f"{idx+1}. {step}")
                
                else:
                    # No recipes found
                    st.info(f"No specific recipes found for {st.session_state.predicted_vegetable}.")
                    st.write("Try searching for a general recipe with this vegetable:")
                    search_term = st.text_input("Search term:", value=st.session_state.predicted_vegetable)
                    if st.button("Search Recipes"):
                        st.markdown(f"<a href='https://www.google.com/search?q={search_term}+recipes' target='_blank'>Search for {search_term} recipes</a>", unsafe_allow_html=True)
            
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching recipe: {e}")
            except (KeyError, IndexError) as e:
                st.error(f"Error processing recipe data: {e}. API might have changed or data is missing.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        
        elif test_image is None:
            st.info("Upload a vegetable image to see recipe recommendations")
            
            # Show sample recipe card
            st.markdown("### Sample Recipe Preview")
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown("<h3 class='recipe-title'>Vegetable Stir Fry</h3>", unsafe_allow_html=True)
            
            sample_col1, sample_col2 = st.columns([1, 1])
            with sample_col1:
                st.markdown("**Ingredients will appear here**")
                st.markdown('<span class="ingredient-tag">ingredient 1</span>', unsafe_allow_html=True)
                st.markdown('<span class="ingredient-tag">ingredient 2</span>', unsafe_allow_html=True)
            
            with sample_col2:
                st.markdown("**Recipe details will appear here**")
                st.markdown("Instructions and steps will be shown after vegetable detection")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">VeggieSense - Detect, Learn, Cook!</div>', unsafe_allow_html=True)

# Handle session state navigation if needed
if 'app_mode' in st.session_state:
    app_mode = st.session_state.app_mode
    del st.session_state.app_mode
    st.rerun()