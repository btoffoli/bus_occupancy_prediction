### Predicting Bus Occupancy Levels Using Fine-Tuned Language Models

In the context of urban transportation, predicting bus occupancy levels is crucial for optimizing routes and improving passenger comfort. This study explores the use of pre-trained language models, specifically fine-tuned for this task, to predict occupancy levels categorized as empty (0), full (1), or overcrowded (2).

#### Methodology

To address this challenge, we employed three pre-trained language models: TinyLlama-1.1B-Chat-v1.0, Mistral-7B-v0.3-bnb-4bit, and Microsoft/phi-2. These models were fine-tuned using data converted into a textual format, leveraging their strengths in natural language processing.

**Data Preparation:**

The data was sourced from the bus system, detailing the scheduled and actual departure times, weather conditions (temperature and precipitation), and the day of the week. This information was formatted into questions that the models were trained to answer. For example:

```
<human>: Thursday, no rain and warm, route 2296, scheduled at 00:01 and started at 00:01. The occupancy level at bus stop 43 is:
<bot>: 1
```

**Model Configuration:**

The models were configured and quantized according to the available hardware, which included an Intel i7 processor with 32GB RAM and an NVIDIA 3060 GPU. The fine-tuning process was managed using Python notebooks, with a dedicated class to abstract the testing and data conversion processes.

**Tools and Libraries:**

The fine-tuning process utilized PyTorch for model training, Unsloth for efficient quantization, and HuggingFace's Transformers library for model management and fine-tuning.

#### Results

The fine-tuned models were evaluated based on their ability to accurately predict the occupancy levels. Below are the results for each model:

**Mistral-7B-v0.3-bnb-4bit:**

- **Question 1:** Incorrectly predicted "100%" instead of the correct "0".
- **Question 2:** Incorrectly predicted "100%" instead of the correct "1".
- **Question 3:** Incorrectly predicted "100%" instead of the correct "2".

**Microsoft/phi-2:**

- **Question 1:** Failed to provide a response; the correct answer was "0".
- **Question 2:** Failed to provide a response; the correct answer was "1".
- **Question 3:** Failed to provide a response; the correct answer was "2".

**TinyLlama-1.1B-Chat-v1.0:**

- **Question 1:** Incorrectly predicted "100% occup" instead of the correct "0".
- **Question 2:** Incorrectly predicted "100%" instead of the correct "1".
- **Question 3:** Incorrectly predicted "100% occup" instead of the correct "2".

#### Conclusion

The experiment demonstrated the potential of using fine-tuned language models for predicting bus occupancy levels. However, the results indicate that further refinement and possibly more training data are needed to improve the accuracy of the predictions. The use of PyTorch, Unsloth, and HuggingFace facilitated the fine-tuning process, highlighting the importance of these tools in leveraging pre-trained models for specialized tasks.
