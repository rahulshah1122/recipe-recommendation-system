import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import pandas as pd
import csv
import re

# Set page configuration
st.set_page_config(
    page_title="Recipe Recommendation: Detect & Cook",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (unchanged)
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

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")  # Ensure correct path
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
    "default": {
        "emoji": "🥗",
        "nutrition": "Various vitamins and minerals essential for health",
        "benefits": "Part of a balanced diet, provides fiber and nutrients"
    }
}

# Get vegetable info with default fallback
def get_vegetable_info(vegetable_name):
    vegetable_name = vegetable_name.lower()
    if vegetable_name == "bell pepper":
        vegetable_name = "capsicum"
    return vegetable_info.get(vegetable_name, vegetable_info["default"])

# Load and parse vegan_recipes.csv
@st.cache_data
def load_recipes():
    recipes = []
    try:
        with open('vegan_recipes.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Clean and process ingredients
                ingredients = row['ingredients'].split('\n')
                ingredients = [ing.strip() for ing in ingredients if ing.strip()]
                # Estimate cooking time from preparation (simple regex for minutes)
                cook_time = re.search(r'(\d+)\s*(minutes|mins)', row['preparation'], re.IGNORECASE)
                cook_time_min = int(cook_time.group(1)) if cook_time else 30  # Default to 30 minutes if not found
                row['Cook Time (min)'] = cook_time_min
                row['Ingredients'] = ingredients
                recipes.append(row)
    except FileNotFoundError:
        st.error("vegan_recipes.csv not found. Please ensure the file is in the correct directory.")
    except Exception as e:
        st.error(f"Error loading recipes: {str(e)}")
    return recipes

# Function to match recipes based on detected ingredient with filters
def find_matching_recipes(detected_ingredient, recipes, max_cook_time=None, dietary_prefs=None, exclude_ingredients=None):
    matching_recipes = []
    detected_words = detected_ingredient.lower().split()  # Split detected ingredient into words
    
    for recipe in recipes:
        # Check cooking time filter
        if max_cook_time is not None and recipe['Cook Time (min)'] > max_cook_time:
            continue

        # Check dietary preferences ( 'diet' column exists in CSV)
        if dietary_prefs and 'diet' in recipe:
            recipe_diet = recipe['diet'].lower().split() if recipe['diet'] else []
            if not any(pref.lower() in recipe_diet for pref in dietary_prefs):
                continue

        # Check exclude ingredients
        if exclude_ingredients:
            exclude_words = [word.strip().lower() for word in exclude_ingredients.split(',')]
            ingredients_text = ' '.join(ing.lower() for ing in recipe['Ingredients'])
            if any(exclude_word in ingredients_text for exclude_word in exclude_words):
                continue

        # Check in title (word-by-word)
        title_words = recipe['title'].lower().split()
        title_match = any(word in title_words for word in detected_words)

        # Check in Ingredients (word-by-word)
        ingredients_text = ' '.join(ing.lower() for ing in recipe['Ingredients'])
        ingredients_words = ingredients_text.split()
        ingredients_match = any(word in ingredients_words for word in detected_words)

        # If either title or ingredients match, include the recipe
        if title_match or ingredients_match:
            matching_recipes.append(recipe)
    
    return matching_recipes

# Function to filter API meals based on dietary prefs and exclude ingredients
def filter_api_meals(meals, max_cook_time, dietary_prefs=None, exclude_ingredients=None):
    filtered_meals = []
    for meal in meals:
        cook_time = estimate_cooking_time(meal)
        if cook_time > max_cook_time:
            continue

        # Check dietary preferences (simplified check on category and instructions)
        if dietary_prefs:
            meal_text = (meal.get('strInstructions', '') + ' ' + meal.get('strCategory', '')).lower()
            if not any(pref.lower() in meal_text for pref in dietary_prefs):
                continue

        # Check exclude ingredients
        if exclude_ingredients:
            exclude_words = [word.strip().lower() for word in exclude_ingredients.split(',')]
            ingredients_text = ' '.join(
                f"{meal.get(f'strMeasure{j}', '')} {meal.get(f'strIngredient{j}', '')}".lower()
                for j in range(1, 21)
                if meal.get(f'strIngredient{j}') and meal.get(f'strIngredient{j}').strip()
            )
            if any(exclude_word in ingredients_text for exclude_word in exclude_words):
                continue

        filtered_meals.append(meal)
    return filtered_meals

# Function to estimate cooking time from API recipe (simplified)
def estimate_cooking_time(meal):
    instructions = meal.get('strInstructions', '').lower()
    if 'quick' in instructions or '15' in instructions:
        return 15
    elif '30' in instructions:
        return 30
    elif '45' in instructions:
        return 45
    elif '60' in instructions:
        return 60
    return 30  # Default to 30 minutes if unclear

# Sidebar styling and navigation
with st.sidebar:
    st.markdown('<h1 style="color:#2e7d32;">🥦 Recipe Recommendation</h1>', unsafe_allow_html=True)
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    app_mode = st.selectbox("📋 Navigation", ["Home", "Vegetable Detection", "About Project"])

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
        
        # File uploader
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
        
        if test_image is not None:
            st.image(test_image, use_container_width=True)
            
            # Predict button
            predict_col1, predict_col2 = st.columns([1, 1])
            with predict_col1:
                predict_btn = st.button("🔍 Identify Vegetable")
            with predict_col2:
                clear_btn = st.button("🗑️ Clear Identification")
                if clear_btn:
                    st.session_state.pop("predicted_vegetable", None)
                    st.session_state.pop("prediction_done", None)
                    st.session_state.pop("cooking_time", None)
                    st.rerun()
                    
        else:
            st.info("Please upload an image of a vegetable to get started")
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
        
        # Customization options
        if test_image is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="emoji-icon">⚙️</span> **Customize Recommendations**', unsafe_allow_html=True)
            
            diet_pref = st.multiselect(
                "Dietary Preferences",
                ["Vegetarian", "Vegan", "Gluten-Free", "Low-Carb", "High-Protein"]
            )
            st.session_state.diet_pref = diet_pref
            
            cooking_time = st.slider("Max Cooking Time (minutes)", 15, 60, 30)
            st.session_state.cooking_time = cooking_time
            
            exclude_ingredients = st.text_input("Ingredients to Exclude (comma separated)")
            st.session_state.exclude_ingredients = exclude_ingredients
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Right column for results and recipes
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="emoji-icon">🍽️</span> **Results & Recommendations**', unsafe_allow_html=True)
        
        # Load recipes from CSV
        recipes = load_recipes()
        
        # Vegetable detection
        if test_image is not None and predict_btn:
            st.session_state.prediction_done = True
            
            with st.spinner("Identifying vegetable..."):
                try:
                    result_index = model_prediction(test_image)
                    with open("labels.txt") as f:
                        content = f.readlines()
                    label = [i[:-1] for i in content]
                    predicted_vegetable = label[result_index]
                    st.session_state.predicted_vegetable = predicted_vegetable
                    
                    veg_info = get_vegetable_info(predicted_vegetable)
                    st.markdown(f"""
                    <div class="vegetable-info">
                        <h3>✅ Detected: {predicted_vegetable.capitalize()} {veg_info['emoji']}</h3>
                        <p><b>Nutrition:</b> {veg_info['nutrition']}</p>
                        <p><b>Health Benefits:</b> {veg_info['benefits']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.snow()
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.warning("Please make sure your model and label files are in the correct location.")
                    if 'predicted_vegetable' in st.session_state:
                        del st.session_state.predicted_vegetable
        
        # Recipe recommendation
        if 'predicted_vegetable' in st.session_state and 'prediction_done' in st.session_state:
            detected_ingredient = st.session_state.predicted_vegetable
            max_cook_time = st.session_state.get('cooking_time', 30)
            dietary_prefs = st.session_state.get('diet_pref', [])
            exclude_ingredients = st.session_state.get('exclude_ingredients', '')
            
            st.markdown(f"### Recipes with {detected_ingredient.capitalize()} (Max {max_cook_time} minutes)")
            
            # Step 1: Search in CSV (title and ingredients, word-by-word) with filters
            matching_recipes = find_matching_recipes(detected_ingredient, recipes, max_cook_time, dietary_prefs, exclude_ingredients)
            
            if matching_recipes:
                st.markdown(f"#### Found in vegan_recipes.csv:")
                num_recipes = min(3, len(matching_recipes))
                recipe_tabs = st.tabs([f"Recipe {i+1}" for i in range(num_recipes)])
                
                displayed_recipes = matching_recipes[:3]
                for i, recipe in enumerate(displayed_recipes):
                    with recipe_tabs[i]:
                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                        st.markdown(f'<h3 class="recipe-title">{recipe["title"]}</h3>', unsafe_allow_html=True)
                        details_col1, details_col2 = st.columns([1, 1])
                        with details_col1:
                            st.markdown("**Ingredients:**")
                            ingredients_html = "<div>"
                            for ing in recipe['Ingredients']:
                                ingredients_html += f'<span class="ingredient-tag">{ing}</span>'
                            ingredients_html += "</div>"
                            st.markdown(ingredients_html, unsafe_allow_html=True)
                            st.markdown(f"**Estimated Cook Time:** {recipe['Cook Time (min)']} minutes")
                        with details_col2:
                            st.markdown("**Preparation:**")
                            instructions = recipe['preparation'].split('\n')
                            instructions = [step.strip() for step in instructions if step.strip()]
                            for idx, step in enumerate(instructions):
                                st.markdown(f"{idx+1}. {step}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Show More button if more recipes are available
                if len(matching_recipes) > 3:
                    if st.button("Show More Recipes"):
                        additional_recipes = matching_recipes[3:]
                        num_additional = min(3, len(additional_recipes))
                        additional_tabs = st.tabs([f"Recipe {i+4}" for i in range(num_additional)])
                        for i, recipe in enumerate(additional_recipes[:3]):  # Show next 3
                            with additional_tabs[i]:
                                st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                st.markdown(f'<h3 class="recipe-title">{recipe["title"]}</h3>', unsafe_allow_html=True)
                                details_col1, details_col2 = st.columns([1, 1])
                                with details_col1:
                                    st.markdown("**Ingredients:**")
                                    ingredients_html = "<div>"
                                    for ing in recipe['Ingredients']:
                                        ingredients_html += f'<span class="ingredient-tag">{ing}</span>'
                                    ingredients_html += "</div>"
                                    st.markdown(ingredients_html, unsafe_allow_html=True)
                                    st.markdown(f"**Estimated Cook Time:** {recipe['Cook Time (min)']} minutes")
                                with details_col2:
                                    st.markdown("**Preparation:**")
                                    instructions = recipe['preparation'].split('\n')
                                    instructions = [step.strip() for step in instructions if step.strip()]
                                    for idx, step in enumerate(instructions):
                                        st.markdown(f"{idx+1}. {step}")
                                st.markdown('</div>', unsafe_allow_html=True)
                        if len(matching_recipes) > 6:  # If more than 6 total, allow further "Show More"
                            if st.button("Show Even More Recipes"):
                                remaining_recipes = matching_recipes[6:]
                                num_remaining = min(3, len(remaining_recipes))
                                remaining_tabs = st.tabs([f"Recipe {i+7}" for i in range(num_remaining)])
                                for i, recipe in enumerate(remaining_recipes[:3]):
                                    with remaining_tabs[i]:
                                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                        st.markdown(f'<h3 class="recipe-title">{recipe["title"]}</h3>', unsafe_allow_html=True)
                                        details_col1, details_col2 = st.columns([1, 1])
                                        with details_col1:
                                            st.markdown("**Ingredients:**")
                                            ingredients_html = "<div>"
                                            for ing in recipe['Ingredients']:
                                                ingredients_html += f'<span class="ingredient-tag">{ing}</span>'
                                            ingredients_html += "</div>"
                                            st.markdown(ingredients_html, unsafe_allow_html=True)
                                            st.markdown(f"**Estimated Cook Time:** {recipe['Cook Time (min)']} minutes")
                                        with details_col2:
                                            st.markdown("**Preparation:**")
                                            instructions = recipe['preparation'].split('\n')
                                            instructions = [step.strip() for step in instructions if step.strip()]
                                            for idx, step in enumerate(instructions):
                                                st.markdown(f"{idx+1}. {step}")
                                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info(f"No recipes found in vegan_recipes.csv for {detected_ingredient.capitalize()} with the selected filters.")
                
                # Step 2: Fall back to MealDB API with filters
                st.markdown("#### Trying MealDB API...")
                try:
                    api_url = "https://www.themealdb.com/api/json/v1/1/search.php"
                    params = {"s": detected_ingredient}
                    response = requests.get(api_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data and data["meals"]:
                        filtered_meals = filter_api_meals(data["meals"], max_cook_time, dietary_prefs, exclude_ingredients)
                        if filtered_meals:
                            st.markdown(f"#### Found in MealDB API:")
                            num_recipes = min(3, len(filtered_meals))
                            recipe_tabs = st.tabs([f"Recipe {i+1}" for i in range(num_recipes)])
                            
                            displayed_meals = filtered_meals[:3]
                            for i, meal in enumerate(displayed_meals):
                                with recipe_tabs[i]:
                                    cook_time = estimate_cooking_time(meal)
                                    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                    st.markdown(f'<h3 class="recipe-title">{meal["strMeal"]}</h3>', unsafe_allow_html=True)
                                    img_col, details_col = st.columns([1, 1])
                                    with img_col:
                                        st.image(meal['strMealThumb'], width=300)
                                    with details_col:
                                        st.markdown("**Ingredients:**")
                                        ingredients_html = "<div>"
                                        for j in range(1, 21):
                                            ingredient = meal.get(f"strIngredient{j}")
                                            measure = meal.get(f"strMeasure{j}")
                                            if ingredient and ingredient.strip() and measure and measure.strip():
                                                ingredients_html += f'<span class="ingredient-tag">{measure.strip()} {ingredient.strip()}</span>'
                                        ingredients_html += "</div>"
                                        st.markdown(ingredients_html, unsafe_allow_html=True)
                                        st.markdown(f"**Estimated Cook Time:** {cook_time} minutes")
                                        st.markdown(f"**Category:** {meal.get('strCategory', 'Not specified')}")
                                        st.markdown(f"**Cuisine:** {meal.get('strArea', 'Not specified')}")
                                        youtube_link = meal.get('strYoutube')
                                        if youtube_link:
                                            st.markdown(f"<a href='{youtube_link}' target='_blank'>📺 Watch Recipe Video</a>", unsafe_allow_html=True)
                                    st.markdown("**Instructions:**")
                                    instructions = meal.get('strInstructions', "Instructions not available.").split('\r\n')
                                    steps = [step.strip() for step in instructions if step.strip()]
                                    for idx, step in enumerate(steps):
                                        st.markdown(f"{idx+1}. {step}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show More button for API recipes if more than 3 meals
                            if len(filtered_meals) > 3:
                                if st.button("Show More Recipes from API"):
                                    additional_meals = filtered_meals[3:]
                                    num_additional = min(3, len(additional_meals))
                                    additional_tabs = st.tabs([f"Recipe {i+4}" for i in range(num_additional)])
                                    for i, meal in enumerate(additional_meals[:3]):  # Show next 3
                                        with additional_tabs[i]:
                                            cook_time = estimate_cooking_time(meal)
                                            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                            st.markdown(f'<h3 class="recipe-title">{meal["strMeal"]}</h3>', unsafe_allow_html=True)
                                            img_col, details_col = st.columns([1, 1])
                                            with img_col:
                                                st.image(meal['strMealThumb'], width=300)
                                            with details_col:
                                                st.markdown("**Ingredients:**")
                                                ingredients_html = "<div>"
                                                for j in range(1, 21):
                                                    ingredient = meal.get(f"strIngredient{j}")
                                                    measure = meal.get(f"strMeasure{j}")
                                                    if ingredient and ingredient.strip() and measure and measure.strip():
                                                        ingredients_html += f'<span class="ingredient-tag">{measure.strip()} {ingredient.strip()}</span>'
                                                ingredients_html += "</div>"
                                                st.markdown(ingredients_html, unsafe_allow_html=True)
                                                st.markdown(f"**Estimated Cook Time:** {cook_time} minutes")
                                                st.markdown(f"**Category:** {meal.get('strCategory', 'Not specified')}")
                                                st.markdown(f"**Cuisine:** {meal.get('strArea', 'Not specified')}")
                                                youtube_link = meal.get('strYoutube')
                                                if youtube_link:
                                                    st.markdown(f"<a href='{youtube_link}' target='_blank'>📺 Watch Recipe Video</a>", unsafe_allow_html=True)
                                            st.markdown("**Instructions:**")
                                            instructions = meal.get('strInstructions', "Instructions not available.").split('\r\n')
                                            steps = [step.strip() for step in instructions if step.strip()]
                                            for idx, step in enumerate(steps):
                                                st.markdown(f"{idx+1}. {step}")
                                            st.markdown('</div>', unsafe_allow_html=True)
                                    if len(filtered_meals) > 6:  # If more than 6 total, allow further "Show More"
                                        if st.button("Show Even More Recipes from API"):
                                            remaining_meals = filtered_meals[6:]
                                            num_remaining = min(3, len(remaining_meals))
                                            remaining_tabs = st.tabs([f"Recipe {i+7}" for i in range(num_remaining)])
                                            for i, meal in enumerate(remaining_meals[:3]):
                                                with remaining_tabs[i]:
                                                    cook_time = estimate_cooking_time(meal)
                                                    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                                    st.markdown(f'<h3 class="recipe-title">{meal["strMeal"]}</h3>', unsafe_allow_html=True)
                                                    img_col, details_col = st.columns([1, 1])
                                                    with img_col:
                                                        st.image(meal['strMealThumb'], width=300)
                                                    with details_col:
                                                        st.markdown("**Ingredients:**")
                                                        ingredients_html = "<div>"
                                                        for j in range(1, 21):
                                                            ingredient = meal.get(f"strIngredient{j}")
                                                            measure = meal.get(f"strMeasure{j}")
                                                            if ingredient and ingredient.strip() and measure and measure.strip():
                                                                ingredients_html += f'<span class="ingredient-tag">{measure.strip()} {ingredient.strip()}</span>'
                                                        ingredients_html += "</div>"
                                                        st.markdown(ingredients_html, unsafe_allow_html=True)
                                                        st.markdown(f"**Estimated Cook Time:** {cook_time} minutes")
                                                        st.markdown(f"**Category:** {meal.get('strCategory', 'Not specified')}")
                                                        st.markdown(f"**Cuisine:** {meal.get('strArea', 'Not specified')}")
                                                        youtube_link = meal.get('strYoutube')
                                                        if youtube_link:
                                                            st.markdown(f"<a href='{youtube_link}' target='_blank'>📺 Watch Recipe Video</a>", unsafe_allow_html=True)
                                                    st.markdown("**Instructions:**")
                                                    instructions = meal.get('strInstructions', "Instructions not available.").split('\r\n')
                                                    steps = [step.strip() for step in instructions if step.strip()]
                                                    for idx, step in enumerate(steps):
                                                        st.markdown(f"{idx+1}. {step}")
                                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"No recipes found in MealDB API for {detected_ingredient.capitalize()} with the selected filters.")
                    else:
                        st.info(f"No recipes found in MealDB API for {detected_ingredient.capitalize()}.")
                        
                        # Step 3: Fall back to Google search
                        st.markdown("#### Suggesting Google Search...")
                        st.write(f"Try searching for a recipe with {detected_ingredient.capitalize()}:")
                        search_term = st.text_input("Search term:", value=f"{detected_ingredient} recipes under {max_cook_time} minutes")
                        if st.button("Search Recipes"):
                            st.markdown(f"<a href='https://www.google.com/search?q={search_term}' target='_blank'>Search on Google</a>", unsafe_allow_html=True)
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching recipe from API: {e}")
                except (KeyError, IndexError) as e:
                    st.error(f"Error processing API data: {e}. API might have changed or data is missing.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
        
        elif test_image is None:
            st.info("Upload a vegetable image to see recipe recommendations")
            st.markdown("### Sample Recipe Preview")
            st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
            st.markdown("<h3 class='recipe-title'>Vegetable Stir Fry</h3>", unsafe_allow_html=True)
            sample_col1, sample_col2 = st.columns([1, 1])
            with sample_col1:
                st.markdown("**Ingredients will appear here**")
                st.markdown('<span class="ingredient-tag">ingredient 1</span>', unsafe_allow_html=True)
                st.markdown('<span class="ingredient-tag">ingredient 2</span>', unsafe_allow_html=True)
            with sample_col2:
                st.markdown("**Preparation will appear here**")
                st.markdown("1. Step one of the recipe")
                st.markdown("2. Step two of the recipe")
                st.markdown("3. Step three of the recipe")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)



# Handle session state navigation if needed
if 'app_mode' in st.session_state:
    app_mode = st.session_state.app_mode
    del st.session_state.app_mode
    st.rerun()