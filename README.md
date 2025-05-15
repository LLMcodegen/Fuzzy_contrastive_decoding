# :zap: Fuzzy-Assisted Contrastive Decoding Improving Code Generation of Large Language Models

### Environment

```bash
pip install transformers>=4.25.1
pip install accelerate>=0.13.2
pip install datasets>=2.6.1
pip install evaluate>=0.3.0
pip install pyext==0.7
pip install mosestokenizer==1.0.0
pip install huggingface_hub>=0.11.1
pip install fsspec<2023.10.0
```

### Code

The main framework of our code is based on [bigcode-project](https://github.com/bigcode-project/bigcode-evaluation-harness). Here is an example demonstrating how to apply Fuzzy-Assisted Contrastive Decoding to a code generation dataset.

```bash
cd $current dir
bash run_main.sh
```

For detailed usage instructions, please refer to the [bigcode documentation](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#documentation).

### Data

We provide the following data used in our experiments:

- Evaluation benchmarks:
    - [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval): The HumanEval benchmark includes 164 manually written Python programming problems, with a primary emphasis on language understanding, algorithms, and fundamental mathematics. It is mainly used to assess the function completion abilities of large language models (LLMs).
    - [MBPP](https://huggingface.co/datasets/nus-yam/mbpp): The MBPP (Mostly Basic Python Problems) benchmark is designed to primarily assess the proficiency of large language models (LLMs) in generating functional code. Specifically, it evaluates their ability to produce correct and efficient Python functions based on given problem descriptions. The test set for this benchmark is composed of 500 carefully curated samples, each consisting of Python programming problems that challenge the models' understanding of syntax, logic, and problem-solving capabilities. These samples cover a range of basic to intermediate coding tasks, ensuring a comprehensive evaluation of the LLMs' function generation skills.
    - [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E): The MultiPL-E initiative extends the HumanEval and MBPP benchmarks by translating them into eighteen additional programming languages, including, but not limited to, C++, C#, Java, PHP, and Bash. This expansion ensures that the evaluation of large language models' (LLMs) code generation capabilities, as originally tested in Python through these benchmarks, can be comprehensively assessed across a diverse set of programming languages. By incorporating languages with varying syntax, paradigms, and use cases, MultiPL-E facilitates a broader and more robust analysis of LLMs' ability to generate functional and accurate code in multiple programming environments.

### Model

We provide the following models used in our experiments:

- Models:
    - [Llama3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B): Llama3-8b is an efficient, lightweight large language model developed by Meta AI, part of the Llama 3 series, with 8 billion parameters. It excels in natural language processing tasks, making it ideal for research and applications requiring low computational resources. Llama3-8B demonstrates strong performance on various benchmarks, such as MMLU, HumanEval, and MBPP, often matching or surpassing larger models in language understanding and code generation. Compared to its predecessors, it features improved training data and architecture, enhancing inference efficiency and task generalization. Primarily designed for research purposes, it is subject to Meta AI’s usage license and is widely used in academic studies, text generation, and dialogue systems.
    - [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf): The CodeLlama-7b model is a specialized version of the Llama model, fine-tuned to enhance its capabilities for specific tasks, such as code generation and code comprehension. This model, developed by Meta AI, leverages the foundational architecture of Llama but incorporates targeted optimizations to improve its performance in programming-related activities. It is designed to excel in generating accurate and functional code across various programming languages, as well as understanding and analyzing code structures, making it a valuable tool for developers and researchers working on coding tasks and software development projects.
    - [StarCoder](https://huggingface.co/bigcode/starcoder): The StarCoder model, boasting 15.5 billion parameters, is a sophisticated large language model meticulously trained on a diverse dataset encompassing over 80 programming languages sourced from Stack (version 1.2). This extensive training enables StarCoder to excel in a wide range of coding tasks, leveraging the rich and varied codebases available in the Stack dataset. Designed to support advanced code generation and comprehension, the model benefits from its broad exposure to multiple programming paradigms and syntaxes, making it a powerful tool for developers and researchers working across numerous programming environments.
    - [WizardCoder-15b](https://huggingface.co/WizardLMTeam/WizardCoder-15B-V1.0): The WizardCoder-15b model is an advanced large language model that has been meticulously fine-tuned using the Evol-Instruct methodology, specifically tailored for enhancing the capabilities of Code-centric LLMs. This fine-tuning process optimizes the model’s performance in programming-related tasks, such as code generation, debugging, and code comprehension, by leveraging the Evol-Instruct approach to refine its understanding of coding patterns and problem-solving strategies. With 15 billion parameters, WizardCoder-15b is designed to deliver superior results in software development and computational tasks, making it a powerful tool for developers and researchers working on complex coding challenges.
