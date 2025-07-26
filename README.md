# Mosquito-OCR-Local

A lightweight, local OCR application written in a single Python script using [EasyOCR](https://github.com/JaidedAI/EasyOCR). Supports CPU, NVIDIA CUDA GPUs, and Apple Silicon (MSP-X) acceleration—all on your machine, no cloud required.

---
### The compiled version can be downloaded here:
[Mosquito OCR Compiled](https://drive.google.com/file/d/1oSSqwr4wJ8A_Y83QtBkmfOZ1PxSSs094/view?usp=sharing)
##### The release was compiled using pyinstaller.
---

## 🔍 Features

* **EasyOCR-powered**: High-accuracy text recognition with deep learning models.
* **Local Processing**: No internet connection or external services.
* **Multi-Backend**:

  * CPU (default)
  * CUDA (NVIDIA GPU)
  * MSP-X (Apple M1/M2)
* **Simple GUI**: Single-script, PySimpleGUI interface, no web browser needed.

---

## 🛠 Prerequisites

* **Python**: 3.12
* **Git** (optional): to clone this repo
* **NVIDIA CUDA Toolkit** (optional): for GPU acceleration

---

## ⚙️ Installation

1. **Clone or download** this repository.

   ```bash
   git clone https://github.com/FiredMosquito831/Mosquito-Ocr-Local.git
   cd Mosquito-Ocr-Local
   ```

2. **Create a virtual environment** and activate it:

   ```bash
   python -m venv venv
   # Windows
   .\\venv\\Scripts\\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requiremets.txt
   ```

4. **(Optional: CUDA)** If you want GPU support, install the CUDA-enabled PyTorch:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

---

## 🚀 Usage

---
![App Screenshot](https://cdn.discordapp.com/attachments/535071254399680523/1398388788278988850/2747576E-1CCC-4BDF-9145-A2D20C06FDDB.png?ex=68852e8b&is=6883dd0b&hm=a4172435588c8106a4be969b75f343206817790628aa7efa132042b9438bab28&)
---



1. **Launch the app**:

   ```bash
   python "Mosquito OCR 1.0.py"
   ```

2. **Use the GUI**:
   *  Ocr clipboard Image
       * The image within clipboard will have OCR applied to it and the text will be available within the text box
   *  Ocr local image
       * It will open a browse window and after selecting an image the OCR will start then the text will be available within the text box
   * Easy selection of languages to be used during OCR
       * A selected languages list can be modified with add/removoe languages from a dropdown/text input table and using the buttons
       * Languages can be added by writing the name of the language directly instead of using the dropdown as well
   * Dark and white mode (Dark mode looks pretty bad it is a work in progress)
---

## 📄 License

This project is licensed under the **AGPL-3.0** License. See [LICENSE](LICENSE) for details.

*Last updated: July 2025*
