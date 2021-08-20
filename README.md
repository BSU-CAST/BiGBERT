# BiGBERT

BiGBERT is a pre-trained deep learning model that uses website URLs and their respective descriptions to identify educational 
resources.

## Installation

To begin using BiGBERT, install the PyPi package:

```bash
pip install bigbert
```

#### Important Note: #### 

The installation size of the package is relatively small, but the first time you instantiate an instance of BiGBERT, two large files need to be downloaded. Details for these files, and their sizes, are provided in the table below.

| File | Size | Purpose |
|---|---|---|
| edu2vec.txt | 5.2 GB | Word embeddings infused with educational standards domain knowledge. Used by the URL vectorizer component internally. |
| bertedu_1e-6lr.p | 438.0 MB | A BERT model fine-tuned with educational domain knowledge. Used for the snippet vectorizer internally. |

## Data Prep

BiGBERT expects a `pandas.DataFrame` as input with two columns: `"url"` and `"description"`. 


## Usage

```python
import numpy as np
import pandas as pd
from bigbert.bigbert import BiGBERT
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

If you use BiGBERT in a research publication, please include the following citation (provided in BibTeX format):

```
@inproceedings{allen2021bigbert,
  title={BiGBERT: Classifying Educational WebResources for Kindergarten-12$^{th}$ Grades},
  author={Allen, Garrett and Downs, Brody and Shukla, Aprajita and Kennington, Casey and Fails, Jerry Alan and Wright, Katherine Landau and Pera, Maria Soledad},
  booktitle={European Conference on Information Retrieval},
  pages={176-184},
  year={2021},
  organization={Springer}
}
```

## **License**

BiGBERT is available under the MIT License.