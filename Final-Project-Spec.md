![Project-Spec-Header-Image](https://github.com/user-attachments/assets/fd9b566b-b5fb-4a4b-b786-9439d19fb68a)

# CS4445 Human Centric Computing: AI Minor – Final Project Specifications

## 1. Project Objectives
1. **Demonstrate Practical Mastery**: Showcase your ability to design, implement, and evaluate ML/AI models using the techniques introduced in class (e.g., data preprocessing, CNN/RNN, regularization, interpretability, fairness).
2. **Highlight Human-Centric Elements**: Incorporate considerations such as interpretability, explainability, fairness, and user-friendly interactions wherever possible.
3. **Apply Sound Methodology**: Manage datasets, perform feature engineering, choose appropriate metrics, and document your process thoroughly.

## 2. Deliverables
Each group must submit the following:
1. **Detailed Project Report** (as a well-formatted PDF file).
2. **Source Code** (with clear comments or Markdown cells if using notebooks).

### 2.1. Report Requirements
Your report should follow standard technical project guidelines and include at least the sections listed below (feel free to add any relevant subsections):

1. **Title Page**  
   - Project title, team members’ names, student IDs, date of submission.

2. **Table of Contents**  
   - Organized with page numbers for easy navigation (tip: you can generate this from the Insert menu in either MS Word or Google Docs!).

3. **Abstract** (150–250 words)  
   - Brief summary of your problem statement, methodology, and key results.

4. **Introduction**  
   - Project motivation and objectives.
   - **Delegation of group work** - important.

5. **Dataset**  
   - Description of the source of data, its nature (structured, unstructured), size, and any relevant attributes.  
   - **Exploratory Data Analysis (EDA)**: Overview of data distribution, missing values, class imbalance, etc.  
   - **Visualizations**: Graphs or plots that illustrate key insights.  
   - **Feature Engineering (if any)**: Explanation of new features created or rationale for feature selection.

6. **Preprocessing**  
   - Detailed steps taken to clean and transform the data (handling outliers, normalization, encoding categorical variables, etc.).  
   - Justification for why each step was necessary.
     - In the case that no preprocessing was done, justify that.

7. **ML Model / Network Structure**  
   - Description of the chosen algorithms/model architectures (e.g., CNN, RNN, ML pipeline, etc.).  
   - **Hyperparameters**: List and explanation of key hyperparameters.  
   - **Loss Function and Optimizer**: Justify the choices and mention any variations or customizations.

8. **Experiments & Evaluation**  
   - **Experimental Setup**: How you conducted your training/validation/testing.  
   - **Metrics**: Which metrics were used (accuracy, F1-score, AUC, etc.) and why.  
   - **Results**: Tables, charts, confusion matrices, or other visual aids to interpret performance.  
   - **Interpretability**: If relevant, demonstration of LIME, SHAP, or other explanation frameworks.  
   - **Fairness/Bias Mitigation**: If applicable, describe any techniques used to identify or reduce unfairness.

9. **Discussion and Conclusion**  
   - Interpretation of the results.  
   - Challenges faced (e.g., class imbalance, data quality).  
   - How results align with or differ from initial expectations.
   - Possible future enhancements or next steps.

10.  **References**  
   - Citations for all external sources, including datasets, research papers, libraries, etc.

### 2.2. Source Code Requirements
- Must be submitted in a clear format (i.e., Python project, Jupyter notebooks).  
- Well-commented to explain key sections, functions, and logic and using Markdown cells (in notebooks) to describe each major step where applicable.  
- Include a short **README** that outlines how to run your project, library dependencies (requirements.txt), and any special instructions.

---

## 3. Grading Rubric (Total 55% of Module Grade)

| **Category** <br>*(Max Points)*                                     | **Beginner** <br>*(Points Range)*                                                                                                                                              | **Developing** <br>*(Points Range)*                                                                                                                                                   | **Accomplished** <br>*(Points Range)*                                                                                                                                                                             | **Exemplary** <br>*(Points Range)*                                                                                                                                                                                      |
|:--------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Project Setup & Data Handling** <br>*(10 pts)*                 | **0–3 pts**<br>• Minimal or no data cleaning/preprocessing<br>• Insufficient or unclear dataset description<br>• Little to no feature engineering or EDA                                                             | **4–6 pts**<br>• Basic data cleaning with some omissions<br>• Limited EDA and feature engineering<br>• Partial understanding of data issues                                                                                 | **7–8 pts**<br>• Good data cleaning approach with clear explanation<br>• Adequate EDA and visualizations<br>• Solid feature engineering or handling of missing/outlier values                                                                         | **9–10 pts**<br>• Thorough exploration of the dataset with creative EDA<br>• Comprehensive feature engineering tied to project objectives<br>• Well-justified data handling decisions throughout                                                             |
| **2. Model Design & Implementation** <br>*(15 pts)*                 | **0–5 pts**<br>• Model choice not justified<br>• Incorrect or incomplete architecture<br>• Minimal hyperparameter tuning                                                                                               | **6–9 pts**<br>• Appropriate model selection but limited rationale<br>• Basic implementation with some tuning<br>• Partial understanding of optimizer/loss                                                                 | **10–12 pts**<br>• Sound rationale for chosen model<br>• Competent implementation with relevant hyperparam tuning<br>• Demonstrates good understanding of training strategies                                                                           | **13–15 pts**<br>• Innovative or well-optimized architecture<br>• Excellent hyperparameter exploration and justification<br>• High level of technical proficiency in ML design and coding                                                                    |
| **3. Evaluation & Analysis** <br>*(10 pts)*                         | **0–3 pts**<br>• Relies on a single or inappropriate metric<br>• Little to no discussion of results<br>• No interpretability/fairness considerations                                                                  | **4–6 pts**<br>• Uses some standard metrics (e.g., precision, recall) with limited explanation<br>• Basic result interpretation<br>• Minimal mention of interpretability/fairness                                            | **7–8 pts**<br>• Applies relevant metrics (e.g., F1, AUC) with clear discussion<br>• Demonstrates basic interpretability (LIME/SHAP) or fairness checks if applicable<br>• Results well-documented                     | **9–10 pts**<br>• Comprehensive evaluation with multiple metrics<br>• Insightful analysis linking metrics to methodology<br>• Advanced interpretability/fairness assessment; acknowledges limitations                                                        |
| **4. Report Quality & Structure** <br>*(10 pts)*                    | **0–3 pts**<br>• Disorganized or missing major sections<br>• Poor clarity, minimal references<br>• Little attention to professional formatting                                                                         | **4–6 pts**<br>• Basic structure present (intro, methods, etc.) but lacks detail<br>• Some inconsistencies in writing or unclear explanations<br>• Partial referencing                                                     | **7–8 pts**<br>• Well-structured report covering all major sections<br>• Clear writing style and logical flow<br>• Proper citations and references                                                                                                    | **9–10 pts**<br>• Exceptionally organized and professional<br>• Excellent clarity with thorough explanations<br>• Meticulous referencing and well-integrated visuals                                                                                       |
| **5. Code Quality & Robustness** <br>*(5 pts)*               | **0–1 pt**<br>• Poorly structured or difficult-to-read code<br>• Sparse or no comments/documentation<br> • Frequently breaks or incomplete| **2–3 pts**<br>• Basic structure but improvements needed<br>• Some comments, not comprehensive<br>• Occasional errors or lack of error handling | **4 pts**<br>• Well-organized and reasonably modular code<br>• Clear inline comments or Markdown explanations<br>• Mostly robust with minor oversights in edge cases or error handling                            | **5 pts**<br>• Excellent code structure with modular, reusable components<br>• Thorough in-line documentation or Markdown cells<br>• Highly robust—handles edge cases gracefully with minimal or no runtime errors |

---

#### 6. Outstanding Project Bonus (**5 points**)

This bonus category is meant to **reward exceptional, out-of-the-box** endeavors that exceed baseline expectations. Up to **5 additional points** can be awarded for extraordinary creativity, complexity, or human-computer interaction (HCI) integration.

| **Level**    | **Points Range** | **Description**                                                                                                                                                                                    |
|:-------------|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Not Awarded** | 0 pts         | • Meets requirements but does not go beyond them.                                                                                                                                                  |
| **Partial Bonus** | 1–3 pts      | • Partial bonus for extra creativity/advanced approach<br>• Some unique elements in ML/HCI design                                                                                          |
| **Full Bonus**    | 4–5 pts      | • Truly innovative solution or domain application. <br>• Exceptional demonstration of advanced techniques or HCI integration.|

**Total = 55 points (55% of module grade)**

---

## 4. Sample Project Ideas

1. **Text Sentiment Analysis with Fairness Considerations**  
   - Explore tweets, product reviews, or social media data.  
   - Investigate demographic bias in sentiment or language use.

2. **Predictive Maintenance for IoT Sensor Data**  
   - Use time-series sensor data to predict machinery failure.  
   - RNN or advanced time-series models with interpretability for system engineers.

3. **Automated Resume Screening**  
   - Classify or rank resumes for relevant job skills.  
   - Incorporate fairness checks and bias detection (e.g., name anonymization or reweighing).

4. **Fashion Item Recommendation**  
   - Build a recommendation system using user and product features.  
   - Evaluate fairness in recommendations across demographic groups.

5. **Crop Yield Prediction & Visualization**  
   - Use environmental data (weather, soil conditions) to predict yield.  
   - Provide interpretability for farmers and highlight usage of real-time data.

6. **Handwritten Digit/Character Recognition**  
   - Build a CNN-based recognition system.  
   - Test interpretability methods and potentially incorporate multi-lingual sets.

> The above project ideas are just suggestions. Feel free to propose your own original idea, especially if it aligns with your interests or any prior domain knowledge.

---

## 5. Suggested Dataset Sources

- [**Kaggle**](https://www.kaggle.com/datasets) – Large variety of domain-specific datasets (finance, health, image, text).  
- [**UCI Machine Learning Repository**](https://archive.ics.uci.edu/ml) – Classic datasets for academic projects.  
- **Government Open Data Portals** (e.g., [data.gov](https://data.gov), [data.gov.uk](data.gov.uk), [data.gov.in](data.gov.in)) – Real-world public datasets on various topics.  
- [**Open Images Dataset**](https://storage.googleapis.com/openimages/web/index.html) or [**COCO Dataset**](https://cocodataset.org/#home) for computer vision tasks.  
- **Public APIs** (e.g., [Twitter](https://developer.x.com/en/docs/x-api), [Reddit](https://www.reddit.com/dev/api/)) for text data (ensure T&Cs compliance).  

---

## 6. Additional Guidelines

1. **Group Work**: When working in the group, ensure roles and responsibilities are clearly communicated. Briefly describe how this was handled in the introduction section of the report i.e., either delegated specific tasks/everyone participated in all aspects/etc.
2. **Plagiarism**: All submissions must be original. Cite any code blocks or data sources that are not your own.  
3. **Submission Format**:  
   - **Report**: PDF (submitted to project assignment in Brightspace).
   - **Code**: Ensure that both the below checklist items are available.
     - Link to your project repository in the CS4445 Github Classroom should be included in your report on the title page.
     - The zip of your repository (submitted to project assignment in Brightspace).
4. **Due Date**: Projects must be submitted by **14/03/2025 23:59 (Friday of Week 9)** to the [project submission page on Brightspace](https://learn.ul.ie/d2l/le/lessons/49281/topics/906587). Late submissions may incur penalties.

---

**Best of luck, and have fun with your final project!**
