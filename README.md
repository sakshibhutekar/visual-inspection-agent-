# 🔍 Visual Inspection Agent

An AI-powered visual inspection pipeline that detects manufacturing defects using **YOLOv8**, analyzes them with **LLaVA (Large Language and Vision Assistant)**, and presents results through an interactive **Streamlit** dashboard.

---

## 🚀 Features

- **Defect Detection** — Fine-tuned YOLOv8 model trained on the NEU Surface Defect Dataset
- **AI Analysis** — LLaVA multimodal LLM provides natural language descriptions of detected defects
- **5-Module Pipeline** — Modular architecture covering detection, classification, analysis, reporting, and visualization
- **Automated Reports** — Generates structured inspection reports with defect details and confidence scores
- **Interactive UI** — Streamlit-based dashboard for real-time image inspection

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Vision-Language Model | LLaVA via Ollama |
| Computer Vision | OpenCV |
| Frontend | Streamlit |
| Model Training | HuggingFace + custom dataset |

---

## 📁 Project Structure

```
visual-inspection-agent/
│
├── app.py                  # Main Streamlit application
├── defect_detector.py      # YOLOv8 defect detection module
├── llava_agent.py          # LLaVA vision-language analysis
├── report_generator.py     # Automated inspection report generator
├── opencv_basics.py        # Image preprocessing utilities
├── utils.py                # Helper functions
└── requirements.txt        # Project dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/sakshibhutekar/visual-inspection-agent-.git
cd visual-inspection-agent-
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull LLaVA model
```bash
# Install Ollama from https://ollama.ai
ollama pull llava
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Input** — User uploads an image via the Streamlit UI
2. **Detection** — YOLOv8 model scans the image and draws bounding boxes around defects
3. **Analysis** — LLaVA analyzes the cropped defect regions and generates descriptions
4. **Report** — System compiles detection results into a structured inspection report
5. **Output** — Annotated image + report displayed on the dashboard

---

## 📊 Model Details

- **Base Model:** YOLOv8n (nano)
- **Dataset:** NEU Surface Defect Dataset
- **Classes:** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches
- **Format:** Custom trained `.pt` weights

---

## 📦 Requirements

```
ultralytics
streamlit
opencv-python
Pillow
requests
ollama
```

---

## 👩‍💻 Author

**Sakshi Bhutekar**
- 📧 sakshibhtekar.work@gmail.com
- 🐙 GitHub: [@sakshibhutekar](https://github.com/sakshibhutekar)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
