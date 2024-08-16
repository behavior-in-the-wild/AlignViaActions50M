# Align Via Actions : Learning Behavior Aligns LLMs With Human Opinions in Zero-Shot

## Overview

This repository includes tools and datasets for evaluating Large Language Models (LLMs) on various behavioral tasks described in the paper and also run evaluations on the OpinionQA-XL dataset.

### Folders Description

#### Evaluation on OpinionQA-XL

The ```OQA-XL``` folder contains necessary files for evaluating LLMs on the OpinionQA-XL dataset. 

**About OpinionQA-XL:**
OpinionQA-XL significantly expands the original OpinionQA dataset to include PEW survey questions up to the final survey conducted in November 2022. The questions are extracted from survey PDFs using optical character recognition technology. Errors in extraction are corrected using GPT-4-turbo, followed by manual verification and correction. OpinionQA-XL introduces 68 new topics, such as Climate Change, Space Tourism, and Digital Economy, thereby greatly expanding the dataset's scope and relevance.

#### Evaluation on AlignViaActions50M

The ```eval_train_tasks``` provides scripts to evaluate LLMs on validation split of the AlignViaActions50M dataset proposed in the associated work. It contains a script to generate LLM responses, and two scripts to evaluate the generated responses on either predictive or generative tasks, as described in the paper.

### Citation

If you find this work useful for your research, please cite the it as follows:

```bibtex
@online{bhattacharyya2024align,
  author = {Bhattacharyya, Aanisha and Agrawal, Susmit and Singla, Yaman K and SR, Nikitha and Menta, Tarun Ram and Krishnamurthy, Balaji},
  title = {Measuring and Improving Persuasive Abilities of Generative Models},
  year = {2024},
  url = {https://behavior-in-the-wild.github.io/align-via-actions}
}
```
