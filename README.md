# Captioning of Images using Attention Mechanism

Author: [Preetham Ganesh](https://www.linkedin.com/in/preethamganesh/)

## Contents

- [Description](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#description)
- [Dataset](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#dataset)
- [Usage](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#usage)
	- [Requirement Installation](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#requirement-installment)
	- [Model Training and Testing](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#model-training-and-testing)
	- [How to run the application?](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#how-to-run-the-application?)
- [Future Work](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#future-work)
- [Support](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#support)
- [License](https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism#license)

## Description

- Developed a full-stack web application for predicting captions of an image given by the user and hosted the model on cloud.
- Pre-processed & tokenized captions using Subword tokenizer.
- Used pre-trained InceptionV3 model to extract spatial features.
- Trained & tested Attention-based Seq2Seq model which produced a BLEU score of 32.5962.

## Dataset

- The data was downloaded from MS-COCO dataset website [[Link]](https://cocodataset.org/#download).
- Due to the size of the data, it is not available in the repository.
- The models in this project were trained & evaluated using the train2017 & val2017 datasets.
- After downloading the data, it should be extracted (without changing names) and saved in the data folder with the sub-folder named 'original_data'.

## Usage

### Requirement Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

Requires: Python 3.6.

```bash
# Clone this repository
git clone https://github.com/preetham-ganesh/image-captioning-using-attention-mechanism.git
cd image-captioning-using-attention-mechanism

# Create a Conda environment with dependencies
conda env create -f environment.yml
conda activate dr_env
pip install -r requirements.txt
```

## Future Work

- Due to computational complexities, the application is executed on localhost.
- In the future, the application will be modified for hosting on an ML cloud server such as AWS or Google Cloud.

## Support

For any queries regarding the repository contact 'preetham.ganesh2015@gmail.com'.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

