Drug Disintegration Time Prediction using Machine Learning

My proposed platform aims to explore how machine learning models can be used to predict the disintegration time of fast disintegrating tablets based on excipient combination and physicochemical properties. Disintegration time is a critical quality attribute in oral dosage forms, especially for fast-dissolving tablets. By applying predictive modeling to existing datasets, this project identifies potential features influencing rapid disintegration and streamlines experimental design in pharmaceutical development.

The data is adopted from an agregation of data study performed by Momeni, Mehri et al. 



 🎯 Objectives

- Establish a clean and structured dataset of formulation features
- Perform exploratory data analysis (EDA) to reveal patterns
- Train basic ML models to predict `DISINTEGRATION_TIME`
- Evaluate model performances and identify key influencing variables
- Export reproducible code and outputs from JupyterLab to GitHub

🧪 Dataset

This project utilizes a curated dataset with 75+ formulation and physical property features, including:

- API descriptors: Molecular weight, LogP, H-bond donors/acceptors, etc.
- Physical characteristics: Angle of repose, tapped density, bulk density
- Excipients: Microcrystalline cellulose, mannitol, magnesium stearate, etc.
- Outcome variable: `DISINTEGRATION_TIME` (seconds)

🔍 Key Columns

Features include (not exhaustive):

['Molecular Weight', 'XLogP3-AA', 'Rotational bond count', 
 'Carrs Compressibility Index', 'Microcrystalline Cellulose',
 'Sodium starch glycolate', 'Sucralose', 'HARDNESS', 
 'FRIABILITY', 'Drug content', 'Water absorption ratio', 'DISINTEGRATION_TIME']

Reference: Dataset development of pre-formulation tests on fast disintegrating tablets (FDT): data aggregation.” BMC research notes vol. 16,1 131. 3 Jul. 2023, doi:10.1186/s13104-023-06416-w