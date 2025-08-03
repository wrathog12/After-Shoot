# AI Teeth Whitener

An AI-powered application to automatically detect teeth in an image and apply a realistic, adjustable whitening filter.

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/wrathog12/After-Shoot.git](https://github.com/wrathog12/After-Shoot.git)
    cd After-Shoot
    ```

2.  **Download Model Weights**
    - **[Download `unet_teeth_v1.pth` from this Google Drive Link](https://drive.google.com/file/d/1unj8ETlKz5mZ0qushsbt_yAfbJgG6ZjF/view?usp=sharing)**
    - Place the downloaded `unet_teeth_v1.pth` file inside the `After-Shoot` project folder.

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Navigate to the project folder in your terminal and run:
```bash
streamlit run app.py


/After-Shoot/
|
|-- app.py
|-- whitener.py
|-- unet_teeth_v1.pth
|-- requirements.txt
|-- .gitignore
`-- README.md
app.py: Your main Streamlit application file.

whitener.py: The backend module with the U-Net model and filter logic.

unet_teeth_v1.pth: Your trained model weights file (hosted on Google Drive).

requirements.txt: The list of necessary Python libraries.

.gitignore: Tells Git to ignore the large .pth file.

README.md: The instructions for setting up and running your project.
