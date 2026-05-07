# Report Structure — Multi-Modal Deepfake Detection System
### Ahanaf Alam | CSI-6-CSP: Honours Computer Science Project
### London South Bank University (LSBU) | School of Engineering | 2024/25

---

## FORMATTING SPECIFICATION

| Element | Specification |
|---|---|
| Chapter Headings (Heading 1) | 12pt, Times New Roman, **Bold** |
| Section Headings (Heading 2) | 12pt, Times New Roman, **Bold** |
| Body Text (Normal) | 12pt, Times New Roman, 1.5 line spacing |
| Margins | 25 mm — Top, Bottom, Left, Right |
| Page numbers — Prelims | Roman numerals (i, ii, iii…) |
| Page numbers — Main text & Appendices | Arabic numerals (1, 2, 3…) |
| Page 1 | First page of Chapter 1 (Introduction) |
| Submission format | Electronic only (Word + PDF) |
| Section numbering | Maximum 3 levels (e.g., 3.2.1) |
| Figures | Caption below: "Figure X: [Title]" — must be referenced in text |
| Tables | Caption above: "Table X: [Title]" — must be referenced in text |
| Referencing | Harvard style throughout |
| Writing style | Third person academic ("it was found that…", "the system was designed to…") |

> **Note on font size for headings:** LSBU slide guidelines state headings should generally use font sizes *larger* than 12pt. However, per the project formatting instruction provided, 12pt Bold Times New Roman is applied consistently. Confirm with supervisor if a distinction is preferred.

---

## WORD COUNT TARGET

**12,000 words ±10%** (10,800 – 13,200 words)
Appendices are **excluded** from the word count.

| Chapter | Target words |
|---|---|
| 1. Introduction | ~900 |
| 2. Literature Review | ~1,800 |
| 3. Technical Review | ~1,500 |
| 4. Methodology | ~800 |
| 5. Requirements | ~1,000 |
| 6. Design | ~1,200 |
| 7. Implementation | ~2,000 |
| 8. Testing and Evaluation | ~1,500 |
| 9. Conclusion and Reflection | ~700 |
| **Total** | **~11,400** |

---
---

## FRONT MATTER (Preliminary Pages — Roman Numerals)

---

### Title Page *(page i — unnumbered)*
```
London South Bank University
School of Engineering

Multi-Modal Deepfake Detection System:
A Late-Fusion Approach Across Image, Audio, and Text Modalities

A dissertation submitted in partial fulfilment of
the requirements for the degree of

BSc (Honours) Computer Science

Ahanaf Alam
[Student ID]

Supervisor: [Supervisor Name]
Second Assessor: [Second Assessor Name]

Academic Year 2024/25

Module: CSI-6-CSP — Honours Computer Science Project
```

---

### Declaration *(page ii)*
The following text must appear verbatim and be signed:

> *"This dissertation is my own original work and has not been submitted elsewhere in fulfilment of the requirements of this or any other award. Any passages taken from my own previous work or other people's work have been quoted and acknowledged by clear referencing to author, source and page(s). Any non-original illustrations are also referenced. I understand that failure to do this amounts to plagiarism and will be considered grounds for failure in this dissertation and the degree as a whole."*

**Signature:** ____________________________________
**Date:** ____________________________________

---

### Abstract *(page iii — ~200 words)*
Standalone summary covering:
- The problem: the growing threat of multimodal deepfake content (image, audio, text) and limitations of unimodal detection systems
- The approach: three independently trained deep learning models (EfficientNet-B0 for images, ResNet18 on Mel-spectrograms for audio, DeBERTa-v3-small for text) combined through a late-fusion weighted ensemble
- Datasets: approximately 39,428 images, 66 audio recordings, and 29,145 text samples sourced from Kaggle
- Key results: image detection accuracy 98.86% (F1=0.9885, AUC=0.9993); audio detection accuracy 85.71% (AUC=1.0); text model trained on 29,145 samples
- Conclusion: the late-fusion architecture improves robustness over any single modality; the modular design allows independent component upgrades

---

### Acknowledgements *(page iv — optional)*
Thank supervisor, second assessor, and any individuals or institutions who supported the work.

---

### Contents *(page v)*
Auto-generated table listing all chapters, sections, subsections, and page numbers.

---

### List of Figures *(page vi)*

| Figure | Title | Chapter |
|---|---|---|
| Figure 1 | Taxonomy of Deepfake Generation Techniques | Chapter 2 |
| Figure 2 | Comparison of Multimodal Fusion Strategies | Chapter 3 |
| Figure 3 | Iterative Development Methodology — Phased Plan | Chapter 4 |
| Figure 4 | High-Level System Architecture Diagram | Chapter 6 |
| Figure 5 | Component-Based Architecture Diagram | Chapter 6 |
| Figure 6 | UML Use Case Diagram | Chapter 6 |
| Figure 7 | UML Sequence Diagram — End-to-End Detection Workflow | Chapter 6 |
| Figure 8 | Late-Fusion vs Early-Fusion Architecture Comparison | Chapter 6 |
| Figure 9 | Image Dataset Sample Grid: Real vs Fake Faces | Chapter 7 |
| Figure 10 | Audio Waveform and Mel-Spectrogram: Real vs Fake Voice | Chapter 7 |
| Figure 11 | Text Dataset Class Distribution Bar Chart | Chapter 7 |
| Figure 12 | Training and Validation Curves — Image Model (EfficientNet-B0) | Chapter 7 |
| Figure 13 | Training and Validation Curves — Audio Model (ResNet18) | Chapter 7 |
| Figure 14 | Training and Validation Curves — Text Model (DeBERTa-v3-small) | Chapter 7 |
| Figure 15 | Confusion Matrices — All Three Modalities | Chapter 8 |
| Figure 16 | ROC Curves — All Three Modalities | Chapter 8 |
| Figure 17 | Unimodal vs Fused System Performance Comparison | Chapter 8 |

---

### List of Tables *(page vii)*

| Table | Title | Chapter |
|---|---|---|
| Table 1 | Comparison of Image Deepfake Detection Approaches | Chapter 3 |
| Table 2 | Comparison of Audio Deepfake Detection Approaches | Chapter 3 |
| Table 3 | Comparison of Text / AI-Generated Content Detection Approaches | Chapter 3 |
| Table 4 | Comparison of Deep Learning Frameworks and Tools | Chapter 3 |
| Table 5 | Functional Requirements — MoSCoW Prioritisation | Chapter 5 |
| Table 6 | Non-Functional Requirements — MoSCoW Prioritisation | Chapter 5 |
| Table 7 | Requirements Traceability Matrix | Chapter 5 |
| Table 8 | Dataset Summary: Modality, Source, Size, Class Distribution | Chapter 7 |
| Table 9 | Data Split Summary: Train / Validation / Test per Modality | Chapter 7 |
| Table 10 | Model Architecture Summary (Backbone, Input, Parameter Count) | Chapter 7 |
| Table 11 | Training Configuration per Model (LR, Epochs, Optimiser, Batch Size) | Chapter 7 |
| Table 12 | Unimodal Model Test Results (Accuracy, F1-Score, AUC-ROC) | Chapter 8 |
| Table 13 | Fusion System vs Unimodal Baselines | Chapter 8 |
| Table 14 | Comparison with Published Deepfake Detection Benchmarks | Chapter 8 |
| Table 15 | Requirements Satisfaction Analysis | Chapter 8 |

---

### Glossary *(page viii)*

| Term | Definition |
|---|---|
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve — measures binary classifier quality independently of classification threshold |
| CNN | Convolutional Neural Network — a deep learning architecture for processing grid-like data such as images |
| DeBERTa | Decoding-enhanced BERT with Disentangled Attention — a transformer model for NLP tasks |
| EfficientNet | A family of CNNs scaled uniformly across width, depth, and resolution |
| EER | Equal Error Rate — the point at which false acceptance rate equals false rejection rate |
| F1-Score | Harmonic mean of Precision and Recall |
| GAN | Generative Adversarial Network — a generative model that pits a generator against a discriminator |
| Late Fusion | A multimodal strategy in which individual modality models make independent predictions that are combined at the decision level |
| Mel-Spectrogram | A time-frequency representation of audio using a perceptually motivated Mel frequency scale |
| ResNet | Residual Network — a deep CNN architecture that uses skip connections to ease training |
| Transformer | A neural network architecture based on self-attention mechanisms, widely used in NLP and increasingly in vision tasks |

---
---

## MAIN CHAPTERS (Arabic Page Numbering begins at page 1)

---

### Chapter 1: Introduction *(~900 words)*

**1.1 Background and Context** *(~250 words)*
Introduce the deepfake phenomenon: its origins in GAN-based synthesis and its rapid escalation across image, video, audio, and text. Describe real-world harms — political disinformation, financial fraud, non-consensual media, and erosion of public trust in digital content. Establish the scale of the problem with recent statistics or reported incidents.

**1.2 Problem Statement and Research Motivation** *(~200 words)*
Articulate the core gap: most existing detection systems target a single modality. A sophisticated deepfake attack may simultaneously deploy fake video, cloned voice, and AI-written text. Motivate the need for a unified multimodal detection architecture capable of addressing all three simultaneously.

**1.3 Aim and Objectives** *(~200 words)*
State the overall aim clearly. List the specific, measurable objectives:
1. Design and implement an image deepfake detector using EfficientNet-B0
2. Design and implement an audio deepfake detector using ResNet18 on Mel-spectrograms
3. Design and implement a text deepfake detector using DeBERTa-v3-small
4. Develop a late-fusion engine to combine unimodal predictions into a unified classification
5. Evaluate each modality detector and the fused system against standard performance metrics (Accuracy, F1-Score, AUC-ROC)
6. Assess the complete system against defined functional and non-functional requirements

**1.4 Scope, Constraints, and Assumptions** *(~100 words)*
Scope: pre-recorded media only; three modalities (image, audio, text); Kaggle-sourced datasets; binary classification (real vs fake). Explicitly excluded: live-stream detection, legal forensic certification, social media platform integration, video-level temporal analysis.

**1.5 Ethical Considerations** *(~100 words)*
Note that Kaggle datasets are publicly available (no primary research involving the public). Flag that some datasets contain images/audio of real public figures — discuss privacy and potential misuse implications. Reference completed Ethics Form (Appendix E). Mention that the system itself could theoretically be misused as an adversarial tool; acknowledge this as a limitation and ethical responsibility of the developer.

**1.6 Structure of the Dissertation** *(~50 words)*
One sentence per chapter describing its content. Direct the reader through the report.

---

### Chapter 2: Literature Review *(~1,800 words)*

**2.1 Introduction to the Chapter** *(~100 words)*
Explain the scope of the review and the key topics investigated, with a brief rationale for why each is relevant to the project.

**2.2 Deepfake Generation Techniques** *(~350 words)*
Review the evolution of synthetic media generation: GAN architectures (DC-GAN, StyleGAN, FaceSwap), Variational Autoencoders, and diffusion models. Explain how each method generates realistic synthetic content and why this makes detection increasingly challenging. Reference **Figure 1** (Taxonomy of Deepfake Generation Techniques).

**2.3 Image and Video Deepfake Detection** *(~400 words)*
Survey CNN-based (XceptionNet, EfficientNet), 3D-CNN, and Vision Transformer approaches. Discuss benchmark datasets (FaceForensics++, CelebDF). Compare reported accuracies and limitations. Identify the gap: high accuracy on seen datasets, but poor generalisation across manipulation types.

**2.4 Audio Deepfake Detection** *(~300 words)*
Review voice cloning and speech synthesis detection. Cover RNN/LSTM-based approaches and spectrogram-based CNN models. Discuss the ASVspoof challenge datasets. Note class imbalance as a recurring challenge in audio deepfake datasets.

**2.5 Text and AI-Generated Content Detection** *(~300 words)*
Survey stylometric analysis, statistical methods, and transformer-based classifiers (BERT, RoBERTa, DeBERTa). Discuss the challenges of detecting LLM-generated text (GPT-2/3/4, etc.) as models improve. Cover relevant benchmark datasets.

**2.6 Multimodal and Cross-Modal Approaches** *(~200 words)*
Survey early fusion, late fusion, cross-modal attention, and hybrid strategies for multimodal deepfake detection. Identify why late fusion offers advantages for modular systems and missing-modality robustness.

**2.7 Summary and Research Gaps** *(~150 words)*
Synthesise the key findings. Identify the specific gap this project addresses: a modular, maintainable, multimodal late-fusion system validated across all three modalities with independent per-modality evaluations. Justify the choices made in the project design.

---

### Chapter 3: Technical Review *(~1,500 words)*

> **Note:** The LSBU module guide groups this with the Literature Review as a single chapter. Based on the sample reports and the technical complexity of this project (three distinct modalities), splitting into two chapters is recommended and aligns with sample report patterns. Confirm with supervisor.

**3.1 Introduction to the Chapter** *(~80 words)*
Explain that this chapter evaluates technical options for meeting each modality-specific requirement and justifies the technology choices made.

**3.2 Image Detection: Architecture Options** *(~350 words)*
Identify the requirement: a pretrained CNN capable of detecting subtle facial manipulation artefacts from images at 224×224 resolution. Describe and evaluate alternatives: ResNet50, VGG-16, InceptionV3, EfficientNet-B0, ViT (Vision Transformer). Reference **Table 1** (Comparison of Image Deepfake Detection Approaches). Justify selection of EfficientNet-B0 based on accuracy/efficiency trade-off, established performance on FaceForensics++, and computational suitability for the available hardware.

**3.3 Audio Detection: Architecture Options** *(~300 words)*
Identify the requirement: a classifier for Mel-spectrogram representations of audio recordings. Describe and evaluate alternatives: LSTM on raw audio, 1D-CNN on MFCC features, 2D-CNN on spectrograms, ResNet18 adapted for single-channel input. Reference **Table 2** (Comparison of Audio Detection Approaches). Justify ResNet18 selection: transfer learning from ImageNet weights, established adaptation to single-channel spectrogram input, and manageable parameter count for the small audio dataset (66 files).

**3.4 Text Detection: Architecture Options** *(~300 words)*
Identify the requirement: a binary classifier for short-to-medium length text distinguishing human-written from AI-generated content. Describe and evaluate alternatives: TF-IDF with logistic regression, BERT, RoBERTa, ALBERT, DeBERTa-v3-small. Reference **Table 3** (Comparison of Text Detection Approaches). Justify DeBERTa-v3-small: superior performance on text classification benchmarks versus BERT at similar size; disentangled attention mechanism better captures semantic and positional information.

**3.5 Frameworks, Libraries, and Tools** *(~300 words)*
Evaluate frameworks for implementation: TensorFlow vs PyTorch, HuggingFace Transformers vs manual implementation, librosa vs other audio libraries. Reference **Table 4** (Comparison of Frameworks and Tools). Justify final selections: PyTorch (dynamic computation graphs, native GPU support, active research community), HuggingFace Transformers (pre-trained DeBERTa weights), librosa (industry standard for audio processing, Mel-spectrogram computation). Compare **Figure 2** (Fusion Strategy Comparison).

**3.6 Summary of Technical Choices** *(~170 words)*
Produce a concise justified summary mapping each technical requirement to the chosen solution. Reference the evaluation evidence in sections 3.2–3.5.

---

### Chapter 4: Methodology *(~800 words)*

**4.1 Research Approach and Paradigm** *(~150 words)*
Applied/experimental research with primary research through hypothesis testing: "a late-fusion multimodal system achieves higher detection accuracy than any single-modality classifier." Supervised binary classification framework.

**4.2 Development Methodology** *(~250 words)*
Agile iterative approach adopted, organised into 8 defined phases (environment setup → data download and analysis → preprocessing → model training → evaluation → fusion implementation → system integration → final testing). Justify agile for ML projects: accommodates model iteration, hyperparameter adjustment, and incremental testing. Reference **Figure 3** (Iterative Development Methodology — Phased Plan).

**4.3 Data Collection Strategy** *(~150 words)*
All datasets sourced from Kaggle (secondary data, no primary data collection required). Justify Kaggle sources: publicly available, reproducible, academic use permitted. Note limitations: audio dataset imbalance (8 real vs 58 fake), fixed historical snapshots not representative of latest GAN outputs.

**4.4 Ethical Considerations in Methodology** *(~150 words)*
GDPR note on datasets containing images/audio of real individuals. Data not stored beyond project use. No questionnaires or contact with the public. Reference completed Ethics Form in Appendix E.

**4.5 Risk and Limitation Acknowledgement** *(~100 words)*
Brief note on identified risks: dataset class imbalance in audio, GPU availability for text training, potential overfitting on small audio set. Mitigation strategies: stratified splits, class-weighted loss, early stopping.

---

### Chapter 5: Requirements *(~1,000 words)*

**5.1 Requirements Elicitation Approach** *(~100 words)*
Requirements derived from literature review findings, technical review conclusions, and analysis of existing systems. No end-user questionnaires required (system targets researchers/analysts).

**5.2 Stakeholder Identification** *(~150 words)*
Identify stakeholders: academic supervisor and second assessor (technical rigour, academic contribution); researchers and content analysts (accuracy, interpretability, reproducibility); indirect stakeholders (platform providers, policymakers, general public affected by deepfake harms).

**5.3 Functional Requirements (MoSCoW)** *(~350 words)*
Reference **Table 5** (Functional Requirements — MoSCoW Prioritisation). Cover:
- **Must Have:** Input processing for each of three modalities; binary real/fake classification with confidence score; independent unimodal evaluation; late-fusion engine; preprocessing pipeline per modality; model training and saving capability
- **Should Have:** Batch processing; modular architecture (components independently replaceable); logging of training metrics; visualisation of training curves and evaluation plots
- **Could Have:** Web-based or GUI interface; attention/saliency map visualisation for explainability; multilingual text detection; near real-time processing
- **Won't Have:** Live-stream video analysis; legal/forensic certification; watermarking; social media API integration

**5.4 Non-Functional Requirements (MoSCoW)** *(~250 words)*
Reference **Table 6** (Non-Functional Requirements — MoSCoW Prioritisation). Cover:
- **Must Have:** Image detection accuracy ≥ 90%; audio detection AUC ≥ 0.85; text detection F1 ≥ 0.85; GDPR-compliant data handling; reproducibility (fixed random seed); modular codebase
- **Should Have:** GPU-accelerated training; processing time under 30 seconds per input; generalisation to unseen samples
- **Could Have:** Visualisation dashboard; cloud scalability
- **Won't Have:** Real-time stream performance; forensic-grade certification

**5.5 Requirements Traceability** *(~150 words)*
Reference **Table 7** (Requirements Traceability Matrix). Show mapping from each requirement to the design chapter element, implementation component, and test case that verifies it.

---

### Chapter 6: Design *(~1,200 words)*

**6.1 Overall System Architecture** *(~250 words)*
Describe the end-to-end pipeline: input layer (media file upload) → modality router → parallel preprocessing streams → three independent unimodal detectors → fusion engine → decision engine → output (classification + confidence score). Reference **Figure 4** (High-Level System Architecture Diagram).

**6.2 Component-Based Architecture** *(~200 words)*
Break the system into discrete components: Input Handler, Image Preprocessing Module, Audio Preprocessing Module, Text Preprocessing Module, Image Detector (EfficientNet-B0), Audio Detector (ResNet18), Text Detector (DeBERTa-v3-small), Fusion Engine, Output Module. Reference **Figure 5** (Component-Based Architecture Diagram).

**6.3 UML Modelling**

*6.3.1 Use Case Diagram (100 words)*
Describe the primary actor (User/Researcher) and system use cases: submit media for analysis, view classification result, view per-modality confidence scores, view evaluation metrics. Reference **Figure 6** (UML Use Case Diagram).

*6.3.2 Sequence Diagram (100 words)*
Walk through the detection flow: media upload → modality detection and routing → concurrent preprocessing → concurrent model inference → score collection by fusion engine → weighted ensemble computation → decision threshold applied → result returned to user. Reference **Figure 7** (UML Sequence Diagram).

**6.4 Multimodal Fusion Design** *(~250 words)*
Justify late fusion over early fusion: each modality's feature space is incompatible for direct concatenation; late fusion preserves per-modality interpretability; missing or noisy modalities can be handled gracefully. Describe the weighted ensemble logic: each unimodal detector outputs a softmax probability score; the fusion engine computes a composite score weighted by validation performance of each model. Apply a decision threshold to produce the final binary classification. Reference **Figures 8** (Fusion Architecture Comparison).

**6.5 Design Justification Against Requirements** *(~200 words)*
Explicitly map key design decisions to the requirements from Chapter 5. Demonstrate that the architecture satisfies the Must Have requirements and accounts for identified risks.

---

### Chapter 7: Implementation *(~2,000 words)*

**7.1 Development Environment and Dataset Setup** *(~200 words)*
Specify environment: Python 3.x, PyTorch, HuggingFace Transformers, librosa, OpenCV, facenet-pytorch, scikit-learn; CUDA GPU support. Reference **Table 11** (Training Configuration). Dataset download via KaggleHub. Reference **Table 8** (Dataset Summary) and **Table 9** (Data Split Summary: 70/10/20 stratified splits).

*Note: Reference **Figure 9** (Image Dataset Sample Grid), **Figure 10** (Audio Waveform and Mel-Spectrogram), **Figure 11** (Text Class Distribution) to illustrate dataset characteristics.*

**7.2 Image Preprocessing and Data Augmentation** *(~200 words)*
Describe transformations: resize to 224×224, normalise using ImageNet mean/std (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Training augmentation: random horizontal flip (p=0.5), random rotation (±10°), colour jitter (brightness and contrast ±0.2). Justify augmentation for improving generalisation on 27,599 training images.

**7.3 Audio Preprocessing** *(~200 words)*
Describe audio loading at 16kHz, fixed duration of 3 seconds (padding with zeros or trimming), conversion to Mel-spectrogram (128 Mel bins), power-to-dB conversion. Justify Mel-spectrogram over raw waveforms and MFCCs for CNN compatibility. Address the small dataset size (45 training samples): note limited augmentation potential and flag as a known limitation.

**7.4 Text Preprocessing** *(~150 words)*
Describe tokenisation using DeBERTa-v3-small tokenizer; maximum sequence length 256; padding to max length; tensor conversion. Note dtype fix applied (float32) to avoid Half/Float mismatch error encountered during development.

**7.5 Image Detection Module** *(~300 words)*
EfficientNet-B0 backbone with ImageNet pretrained weights. Modified classifier head: Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→2). AdamW optimiser (lr=1e-4, weight_decay=0.01). 10 epochs, ReduceLROnPlateau scheduler (patience=2). Best validation accuracy: 98.86%, F1=0.9885, AUC=0.9993. Reference **Figure 12** (Image Training Curves) and **Table 10** (Model Architecture Summary).

**7.6 Audio Detection Module** *(~300 words)*
ResNet18 backbone adapted for single-channel input (Conv1 modified: 3-channel → 1-channel). Classifier head: Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→2). AdamW (lr=1e-4). 10 epochs. Best validation accuracy: 85.71%, AUC=1.0. Note class imbalance challenge (8 real vs 58 fake). Reference **Figure 13** (Audio Training Curves).

**7.7 Text Detection Module** *(~250 words)*
DeBERTa-v3-small backbone with CLS token representation. Classifier head: Dropout(0.3) → Linear(hidden_size→128) → ReLU → Dropout(0.2) → Linear(128→2). AdamW (lr=2e-5). 5 epochs, batch size 16. dtype fix applied to resolve Half/Float mismatch. Reference **Figure 14** (Text Training Curves).

**7.8 Fusion Engine and Model Export** *(~200 words)*
Describe the late-fusion weighted ensemble: each model produces softmax probability scores for the real/fake classes; the composite score is computed as a performance-weighted average; the final decision is made by applying a 0.5 classification threshold. Models saved as .pth files with metadata.json (architecture name, accuracy, F1, AUC per model). Reference **Table 13** for fusion performance.

---

### Chapter 8: Testing and Evaluation *(~1,500 words)*

**8.1 Testing Strategy** *(~200 words)*
Describe the hold-out test set approach (20% stratified split, fixed seed=42 for reproducibility). Hardware and software environment for evaluation. Confirm test sets were unseen throughout training and validation. Reference **Table 9** (Data Split Summary).

**8.2 Evaluation Metrics** *(~250 words)*
Define and justify each metric:
- **Accuracy:** overall correct classifications as a proportion of total predictions
- **F1-Score:** harmonic mean of Precision and Recall; particularly important for imbalanced classes
- **AUC-ROC:** area under the Receiver Operating Characteristic curve; threshold-independent measure of discriminative ability; preferred for imbalanced datasets
- **EER:** Equal Error Rate — the operating point where false acceptance equals false rejection; relevant for security-critical applications

**8.3 Unimodal Model Test Results** *(~350 words)*
Present test set results per modality. Reference **Table 12** (Unimodal Model Test Results). Reference **Figure 15** (Confusion Matrices) and **Figure 16** (ROC Curves). Discuss each model's performance, error patterns, and any unexpected results. Specifically address the audio model's small test set (14 samples) and what this means for confidence in results.

**8.4 Fusion System Evaluation** *(~300 words)*
Present the performance of the combined late-fusion system. Reference **Table 13** (Fusion vs Unimodal) and **Figure 17** (Performance Comparison Chart). Discuss whether the fusion meaningfully improves over the best unimodal result and why (or why not). Assess robustness: what happens when one modality is absent or noisy.

**8.5 Comparison with Published Benchmarks** *(~200 words)*
Compare results against published deepfake detection benchmarks from the literature. Reference **Table 14** (Benchmark Comparison). Discuss generalisability: the project models were tested on the same distribution as training data; out-of-distribution performance is unknown and is flagged as a limitation.

**8.6 Requirements Satisfaction Analysis** *(~200 words)*
Evaluate which functional and non-functional requirements were met, partially met, or not met. Reference **Table 15** (Requirements Satisfaction Analysis). Trace outcomes back to objectives stated in Chapter 1, Section 1.3.

---

### Chapter 9: Conclusion and Reflection *(~700 words)*

**9.1 Summary of Work Undertaken** *(~150 words)*
Restate the problem and give a concise narrative summary of the work completed across all phases — from dataset preparation through model training and fusion implementation to evaluation.

**9.2 Achievement of Aims and Objectives** *(~200 words)*
Map each objective from Chapter 1.3 to its outcome. Include quantitative results (accuracy, F1, AUC) as the evidence of achievement. Be honest about objectives that were only partially achieved (e.g., text model training completed but not yet fully integrated into the fusion pipeline).

**9.3 Limitations of the System** *(~150 words)*
Address: audio dataset severely imbalanced and too small (66 files total; 14 test samples); models trained on specific Kaggle distributions with unknown out-of-distribution performance; no live-stream or real-time capability; late fusion requires all three modalities to be available; no explainability mechanism to indicate which region/feature triggered detection.

**9.4 Future Work** *(~100 words)*
Suggest: collect a larger, more balanced audio dataset; deploy via a Flask or Streamlit web interface with REST API; add a GradCAM/attention visualisation layer for explainability; explore cross-modal attention-based fusion; evaluate on out-of-distribution datasets (FaceForensics++, ASVspoof); investigate video-level temporal detection.

**9.5 Personal Reflection and Learning Outcomes** *(~100 words)*
Reflect on skills developed: multimodal deep learning pipeline design, PyTorch proficiency, transformer fine-tuning, dataset handling under class imbalance, and academic project management under the LSBU CSI-6-CSP framework.

---

### References
Harvard referencing style. All sources cited in Chapters 2, 3, and throughout. Must include:
- Deepfake generation: GAN, diffusion model foundational papers
- Detection benchmarks: FaceForensics++, ASVspoof, LLM detection papers
- Model architectures: EfficientNet (Tan & Le, 2019), ResNet (He et al., 2016), DeBERTa (He et al., 2021)
- Fusion strategies
- Any sources cited from the progress report literature review

---
---

## APPENDICES *(excluded from word count)*

**Appendix A — Source Code / Notebook**
Full deepfake_detection.ipynb notebook or link to GitHub repository. Include all phases (environment setup through model export).

**Appendix B — Dataset Details and Licences**
Full details of each Kaggle dataset: title, author, URL, licence, access date, and class distribution.

**Appendix C — Model Export Metadata**
Contents of metadata.json: per-model architecture name, test accuracy, F1, AUC.

**Appendix D — Training Log**
Excerpt or summary from training_log.txt confirming completed training runs.

**Appendix E — Ethics Form**
Completed and signed LSBU Ethics Form. Note: all data from publicly available Kaggle datasets; no contact with members of the public; datasets contain images/audio of real public figures — noted in the introduction.

**Appendix F — Project Logbook / Meeting Notes**
Summary record of supervisor meetings and progress milestones, as required by the CSI-6-CSP module.

---
---

## NOTES FOR WRITING

**On the literature vs technical review split:** The module guide slide lists these as one chapter ("Literature and Technical Review"). However, all four sample reports split them. Given the breadth of Ahanaf's project (three modalities, each with distinct technical options), two separate chapters is strongly recommended. Confirm with supervisor.

**On ethics:** The ethics form must accompany both the Initial Progress Report (Week 10) and the Final Dissertation. Include it as Appendix E. Even though no primary research is involved, the project builds a system that could be misused — this ethical consideration should be discussed in Chapters 1 and 4 to gain credit.

**On the Declaration:** The exact text from the LSBU guidelines must appear verbatim and be signed. Do not paraphrase it.

**On word count:** The 12,000 word limit is strict (±10%). Appendices are excluded. Keep Implementation chapter concise — describe decisions and justify them; do not paste raw code into the main body.

**On referencing:** Every figure and table must be explicitly referenced in the main body text (e.g., "as shown in Figure 4…", "Table 12 presents…"). Non-original figures must include a source citation.

---

*Structure prepared: 24 March 2026*
*Based on: LSBU CSI-6-CSP Module Guide 2024/25 v8; LSBU Final Report Guidelines (02-ethics-and-final-report-guidelines-2425.pptx); Literature Review Guidelines (04-literature-review-2425.pptx); Technical Review Guidelines (05-technical-review-2425.pptx); sample dissertations: Juan Carlos Blanco Delgado (Inventory Management), Jake Wakelin (AIM93), Erblin Mulkurti (AM Prevention), Raima Saad Butt (eBay Store Management); project files from Ahanaf's workspace (README.md, progress.docx, deepfake_detection.ipynb, training_log.txt).*
