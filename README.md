## MLFlow Experiment Tracking Project

MLflow for machine learning experiment tracking, providing a structured approach to log parameters, metrics, and artifacts during model development. 

Key features of this project:

- Tracking experiments with different machine learning models

- Logging parameters, metrics, and model artifacts

- Comparing model performance across different runs

- Organizing experiments with proper naming and tagging

- Saving trained models with all dependencies for reproducibility

The project serves as a template for implementing DVC in ML projects, showing best practices for data and model versioning, pipeline automation, and experiment reproducibility.


### ​ Project Structure

```bash
Building-DVC-Pipeline/
│
├── malartifacts/           # logging data file with exprements
│    
├── mlruns                  # logging data file 
│
├── src/                    # Source code
│   ├── app.py/             # Mlflow logging sample code
│   ├── app._1.ipynb/       # Mlflow logging sample code with GridSearchCV
│   └── autolog.py/         # Mlflow logging sample code with autolog 
│              
│
├── .gitignore              # Specifies intentionally untracked files
├── Confustion-matrix.png   #Confustion-matrix.png
└── README.md               # Project documentation
```

 mlflow ui :-
`Opens MLflow tracking UI at http://127.0.0.1:5000`




