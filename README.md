# CAMeLBERT_morphosyntactic_tagger
Codebase for "[Morphosyntactic Tagging with Pre-trained Language Models for Arabic and its Dialects](https://aclanthology.org/2022.findings-acl.135/)". Findings of ACL, 2022.

Some of the models are already part of the newer version of [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools). Please check out the repository if you want to try out our tagger! Currently, unfactored MSA, EGY, and GLF models are available through CAMeL Tools.

## Requirements
```bash
git clone https://github.com/CAMeL-Lab/CAMeLBERT_morphosyntactic_tagger.git
cd CAMeLBERT_morphosyntactic_tagger

conda create -n CAMeLBERT_morphosyntactic_tagger python=3.7
conda activate CAMeLBERT_morphosyntactic_tagger

pip install -r requirements.txt

# install the latest camel tools
git clone https://github.com/CAMeL-Lab/camel_tools.git
cd camel_tools
# Install from source
pip install -e .
# download models
camel_data -i disambig-bert-unfactored-all
```

## Example: How to tag a sentence
```python
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

# MSA
MSA_unfactored = BERTUnfactoredDisambiguator.pretrained(model_name='msa')
MSA_text = simple_word_tokenize('كيف حالك ؟')

# tag with the analyzer
MSA_unfactored.tag_sentence(MSA_text)

# without the analyzer
MSA_unfactored.tag_sentence(MSA_text, use_analyzer=False)
```
* The morphological analyzer used in the example is not the same as the one in the paper.

## Experiments
This repo is organized as follows:
- [data](https://github.com/CAMeL-Lab/CAMeLBERT_morphosyntactic_tagger/releases/tag/v0.0.1): models and preprocessed datasets used in our experiments.
- [scripts](https://github.com/CAMeL-Lab/CAMeLBERT_morphosyntactic_tagger/tree/main/scripts): scripts used to fine-tune [CAMeLBERT-MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa) and [CAMeLBERT-Mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix) for morphosyntactic tagging task.
 

## Citation

```bibtex
@inproceedings{inoue-etal-2022-morphosyntactic,
    title = "Morphosyntactic Tagging with Pre-trained Language Models for Arabic and its Dialects",
    author = "Inoue, Go  and
      Khalifa, Salam  and
      Habash, Nizar",
    booktitle = "Proceedings of the Findings of the Association for Computational Linguistics: ACL2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    abstract = "We present state-of-the-art results on morphosyntactic tagging across different varieties of Arabic using fine-tuned pre-trained transformer language models. Our models consistently outperform existing systems in Modern Standard Arabic and all the Arabic dialects we study, achieving 2.6% absolute improvement over the previous state-of-the-art in Modern Standard Arabic, 2.8% in Gulf, 1.6% in Egyptian, and 8.3% in Levantine. We explore different training setups for fine-tuning pre-trained transformer language models, including training data size, the use of external linguistic resources, and the use of annotated data from other dialects in a low-resource scenario. Our results show that strategic fine-tuning using datasets from other high-resource dialects is beneficial for a low-resource dialect Additionally, we show that high-quality morphological analyzers as external linguistic resources are beneficial especially in low-resource settings."
}
```