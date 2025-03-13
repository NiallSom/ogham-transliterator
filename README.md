[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/XqvnLU5y)
# ᚌᚔᚈ ᚌᚑᚑᚅᚄ
Repository containing files and source code for the CS4445 AI module's final capstone project.

**Group name:** ᚌᚔᚈ ᚌᚑᚑᚅᚄ 
  
**Team members:**
- Fred Sheppard - 23361433
- Tóla Bowen MacCurtáin - 23383372
- Niall Somers - 23371455

## Installation/Pre-requisites

### Environment Setup
- Python 3.8 or higher
- pip package manager

### Required Libraries
```bash
pip install -r requirements.txt
```

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-project-team.git
   cd ai-project-team
   ```
2. Install dependencies as mentioned above
3. If you are planning to train a model run synthesise_data.py

4. Select the options:
- `--model`: Path to a custom trained model 
- `--image`: Path to the image you would like to run the defined model on
- `--labels`: Path to defined labels json file
- `--heatmap`: Define whether you want the heatmap of focus points or not

Example:
```bash
python model/client.py --model best_ogham_hybrid_model.h5 --image G.png --heatmap 
python model/client.py --model best_ogham_hybrid_model.h5 --image test.png --train
python model/client.py --model best_ogham_hybrid_model.h5 --image test.png --heatmap    
python model/client.py --model best_ogham_hybrid_model.h5 --image test.png --train --heatmap   
python model/client.py --model best_ogham_hybrid_model.h5 --image test.png --labels model/label_to_letter.json #only used if labels cannot be found 
```
