# minbert Assignment
by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig

This is an exercise in developing a minimalist version of BERT, part of Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html).

In this assignment, you will implement some important components of the BERT model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with the BERT model.

## Assignment Details

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
    * Please run it with `source setup.sh` instead of `./setup.sh`.
* We will run your code with the following commands, so make sure that whatever your best results are reproducible using these commands (where you replace ANDREWID with your lowercase Andrew ID):
    * Do not change any of the existing command options (including defaults) or add any new required parameters
```
mkdir -p ANDREWID

python3 classifier.py --option [freeze/finetune] --epochs NUM_EPOCHS --lr LR --train data/sst-train.csv --dev data/sst-dev.csv --test data/sst-test.csv
```
## Reference accuracies: 

Frozen for SST:
Dev Accuracy: 0.391 (0.007)
Test Accuracy: 0.403 (0.008)

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

Finetuning for SST:
Dev Accuracy: 0.515 (0.004)
Test Accuracy: 0.526 (0.008)

Finetuning for CFIMDB:
Dev Accuracy: 0.966 (0.007)
Test Accuracy: - (test labels are withheld)

### Submission

We are asking you to submit in two ways:
1. *Canvas:* a full code package, which will be checked by the TAs in the 1-2 weeks 
   after the assignment for its executability.
2. *Kaggle:* we will ask you to submit your system outputs to Kaggle, in case there are any problems executing your code.

#### Canvas Submission

For submission via [Canvas](https://canvas.cmu.edu/),
the submission file should be a zip file with the following structure (assuming the
lowercase Andrew ID is ``ANDREWID``):
```
ANDREWID/
├── base_bert.py
├── bert.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── sst-dev-output.csv 
├── sst-test-output.csv 
├── cfimdb-dev-output.csv 
├── cfimdb-test-output.csv 
└── setup.py
```

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It
will throw assertion errors if the format is not expected, and *submissions that fail
this check will be graded down*.

Usage:
1. To create and check a zip file with your outputs, run
   `python3 prepare_submit.py path/to/your/output/dir ANDREWID`
2. To check your zip file, run
   `python3 prepare_submit.py path/to/your/submit/zip/file.zip ANDREWID`

Please double check this before you submit to Canvas; most recently we had about 10/100
students lose a 1/3 letter grade because of an improper submission format.

#### Kaggle Submission

We will have you submit your system outputs to Kaggle in case of any problems with running your code.

Please create a Kaggle account using your CMU email and submit your *output.csv files for each of the four evaluation sets to the following Kaggle pages. 

* sst-dev: https://www.kaggle.com/t/490a691dfb9d4e54ba9e045fd78ac8c0
* sst-test: https://www.kaggle.com/t/0f66dc8812b94027a198d0ea148763e7
* cfimdb-dev: https://www.kaggle.com/t/db7ad612e5ba4117bf5a0f10fb2daa0e
* cfimdb-test: https://www.kaggle.com/t/84f6e5b5b62541dca741812045a190de

While each Kaggle page has a leaderboard, _your rank on it does not count in any way toward your score on this assignment_. We’re just using Kaggle as a way for you to verify your scores immediately, and as a backup in case we cannot run the code you submit to Canvas.


### Grading
* A+: You additionally implement something else on top of the requirements for A, and achieve significant accuracy improvements. Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the MLM objective to do domain adaptation
    * try [alternative fine-tuning algorithms](https://www.aclweb.org/anthology/2020.acl-main.197)
    * add other model components on top of the model
* A: You implement all the missing pieces and the original ``classifier.py`` with ``--option freeze`` and ``--option finetune`` code that achieves comparable accuracy to our reference implementation.
* A-: You implement all the missing pieces and the original ``classifier.py`` with ``--option freeze`` and ``--option finetune`` code but accuracy is not comparable to the reference.
* B+: All missing pieces are implemented and pass tests in ``sanity_check.py`` (bert implementation) and ``optimizer_test.py`` (optimizer implementation)
* B or below: Some parts of the missing pieces are not implemented.

If your results can be confirmed through the submission of your outputs, but there are problems with your
code submitted through Canvas, such as not being properly formatted, not executing in
the appropriate amount of time, etc., you will be graded down 1/3 grade.

All assignments must be done individually and we will be running plagiarism detection
on your code. If we confirm that any code was plagiarized from that of other students
in the class, you will be subject to strict measure according to CMUs academic integrity
policy.

### Acknowledgement
Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
