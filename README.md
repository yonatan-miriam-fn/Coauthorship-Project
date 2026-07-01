This repository contains all code used during my final project.

The file <code>data_download.py</code> contains the code for downloading, pre-processing, and splitting the data into 
train, dev, and test sets.
The file <code>data_splitter.py</code> contains helper functions for <code>data_download.py</code>.

The <code>data</code> folder is where that data is located.

The <code>trainer.py</code> file contains all the code for fine-tuning that I used in my project. In order to run it,
a few command line arguments are provided:
- <code>-m</code>, or <code>--model</code> specifies which 4bit quantized model to use as the base. Default is 
"unsloth/llama-3-8b-Instruct-bnb-4bit".
- <code>-r</code>, or <code>--rank</code>, is the LoRA rank. Defaults to 16.
- <code>-a</code>, or <code>--alpha</code>, is the scaling factor for the LoRA adapters. Defaults to 16.
- <code>-lr</code>, or <code>--learning-rate</code>, is the learning rate of the trainer. Defaults to 2e-4.
- <code>-t</code>, or <code>--template</code>, is the index of the chat template to use. 0 is the template described in
the writeup as "medium detail", and is the default 1 is the one that was described as "simple detail", and 2 is the
template that was only used for a single run.
- <code>-f</code>, or <code>--full</code>, if included, will cause the code to go through a full epoch of training as
opposed to the default 60 update steps.

If the <code>requirements.txt</code> file is not working, the following imports should be used:
- <code>pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"</code>
- <code>pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes</code>
- If intending to run <code>data_download.py</code>: <code>pip install kagglehub</code>

The command line may error when trying to install <code>xformers</code>. I don't understand why this happens but it
appears to work sometimes even when that error appears.