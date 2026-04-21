cat <<EOF > README.md
# Few-Shot Uniform Detection Model
<img width="1682" height="1177" alt="image" src="https://github.com/user-attachments/assets/a9588d3d-d7fd-4bc8-8200-125cde55dc03" />


A specialized Computer Vision model designed to distinguish between individuals wearing uniforms and those who are not, using a **Few-Shot Learning** approach. 

## 🚀 Core Advantage
Unlike traditional deep learning models that require thousands of labeled images, this model can learn to identify new uniform types with **just a few samples**. This makes it highly adaptable for diverse corporate, educational, and security environments.

## 🛠 Key Features
- **Few-Shot Learning:** Efficiently generalizes uniform features from minimal training data.
- **Action & Identity Segmentation:** Specifically tuned to separate uniform patterns from complex backgrounds.
- **Metric Learning (Triplet Loss):** Uses advanced distance-based loss functions to maximize the difference between "Uniform" and "Non-Uniform" feature vectors.
- **PyTorch Implementation:** Built on a robust and scalable architecture.

## 📁 Project Structure
- \`train.py\`: Core training script for the few-shot learning iterations.
- \`models/\`: Contains the backbone architecture and few-shot classification heads.
- \`inference.py\`: Test the model on new images/videos with limited reference samples.
- \`configs/\`: Configuration files for shot-count and feature extraction settings.

## 🏃 Usage
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/diyorbekxidirov-hub/Uniform.git
   cd Uniform
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Run the few-shot inference:
   \`\`\`bash
   python inference.py --samples path/to/uniform_samples/ --test path/to/target_image.jpg
   \`\`\`

cat <<EOF >> README.md

## 🔒 Model Checkpoints
The trained model weights are securely hosted on **Hugging Face**.
- **Model URL:** [huggingface.co/diorbek2002/Uniform](https://huggingface.co/diorbek2002/Uniform)
- **Status:** Private (Access granted upon request or with valid HF Token).

### How to use:
To load the model in your code, ensure you have the \`huggingface_hub\` library installed and are logged in:
\`\`\`python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="diorbek2002/Uniform", filename="best.pt", use_auth_token=True)
\`\`\`
EOF

# GitHub-ga yuborish
git add README.md
git commit -m "Update README with private model info"
git push
EOF

# GitHub-ga yuborish
git add README.md
git commit -m "Update README with Few-Shot focus"
git push
