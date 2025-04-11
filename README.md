---

# Recipe Recommendation System

A machine learning-powered application that recommends vegan recipes based on image recognition of fruits and vegetables. Built with Python, TensorFlow, and Streamlit, it offers an interactive interface for users to discover recipes tailored to their identified ingredients.

## Features

- **Image-Based Ingredient Recognition**: Utilizes a trained convolutional neural network (CNN) to identify fruits and vegetables from user-uploaded images.
- **Vegan Recipe Suggestions**: Provides recipe recommendations based on the recognized ingredients, sourced from a curated dataset.
- **Interactive Web Interface**: Employs Streamlit for a user-friendly and responsive application experience.

## File Structure

- `main.py`: Main application script integrating image recognition and recipe recommendation functionalities.
- `trained_model.h5`: Pre-trained CNN model for classifying fruits and vegetables.
- `labels.txt`: List of class labels corresponding to the model's output categories.
- `vegan_recipes.csv`: Dataset containing vegan recipes mapped to specific ingredients.
- `Training_fruit_vegetable.ipynb`: Jupyter Notebook detailing the model training process.
- `Testing_fruit_veg_recognition.ipynb`: Notebook for testing and validating the trained model.
- `home_img.jpg` & `home_img.png`: Images used for the application's homepage or interface.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries:
  - `streamlit`
  - `tensorflow`
  - `pandas`
  - `numpy`
  - `Pillow`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rahulshah1122/recipe-recommendation-system.git
   ```



2. Navigate to the project directory:

   ```bash
   cd recipe-recommendation-system
   ```



3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```



4. Run the application:

   ```bash
   streamlit run main.py
   ```



## Usage

- Upon running the application, a new browser window will open displaying the interface.
- Upload an image of a fruit or vegetable.
- The application will identify the ingredient and suggest relevant vegan recipes.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

*Note: This project is a basic implementation and may not cover all aspects of a full-fledged recipe recommendation system.*

--- 
