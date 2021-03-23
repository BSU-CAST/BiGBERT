# BiGBERT

BiGBERT is a pre-trained deep learning model that uses website URLs and their respective descriptions to identify educational 
resources.

## Requirements Setup

To begin using BiGBERT, install the dependencies (recommended done inside a virtual environment):

```bash
pip install -r requirements.txt
```

## Data Prep

BiGBERT expects a `pandas.DataFrame` as input with two columns: `"url"` and `"description"`. 


## Usage

```python
import numpy as np
import pandas as pd
from BiGBERT import BiGBERT
from sklearn.metrics import accuracy_score

# This file should have "url", "description" along with "target" columns
data = pd.read_csv("some/data/file.csv")
y = data["target"]
X = data.drop(columns=["target"], inplace=True)

model = BiGBERT()
y_pred = model.predict(X)
print(accuracy_score(y, np.argmax(y_pred, axis=1)))
```


## **Citation**

If you use BiGBERT in a scientific publication, please include the following citation (provided in BibTeX format):

```
@inproceedings{allen2021bigbert,   
  title={BiGBERT: Classifying Educational WebResources for Kindergarten-12$^{th}$ Grades},
  author={Allen, Garrett and Downs, Brody and Shukla, Aprajita and Kennington, Casey and Fails, Jerry Alan and Wright, Katherine Landau and Pera, Maria Soledad},
  booktitle={European Conference on Information Retrieval},
  pages={To Appear},
  year={2021},
  organization={Springer}
}
```