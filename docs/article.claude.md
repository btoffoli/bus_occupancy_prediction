## Methodology: Fine-tuning Language Models for Bus Occupancy Prediction

### Problem Formulation

In this research, we address the challenge of predicting bus occupancy levels categorized into three discrete states: empty (0), full (1), and overcrowded (2). Rather than employing traditional regression or classification approaches, we explore the potential of transformer-based language models, which have demonstrated remarkable capabilities in various prediction tasks when properly fine-tuned.

### Data Preparation

The data for this study was sourced from a bus transportation system's operational records. To leverage pre-trained language models for this task, we converted the structured transportation data into a text-based format. Each query was formulated to include contextual information such as:

- Day of the week
- Weather conditions (temperature: warm > 27°C, normal 18-27°C, cold < 18°C)
- Precipitation status (no rain, light rain, heavy rain)
- Bus route number
- Scheduled departure time
- Actual departure time
- Bus stop identifier

This information was structured in natural language format to serve as input for the language models. The target output was the occupancy level (0, 1, or 2) at the specified bus stop. An example of this query-response format used for fine-tuning is:

```
<human>: Thursday, no rain and warm, route 2296, scheduled at 00:01 and started at 00:01. The occupancy level at bust stop 43 is:
<bot>: 1
```

### Model Selection and Implementation

We selected three language models of varying architectures and parameter counts to evaluate their efficacy for this task:

1. **TinyLlama-1.1B-Chat-v1.0**: A compact model with approximately 1.1 billion parameters
2. **mistral-7b-v0.3-bnb-4bi**: A medium-sized model with 7 billion parameters, quantized for efficiency
3. **microsoft/phi-2**: A newer model with advanced capabilities and efficient architecture

All models were appropriately quantized to operate within our available hardware constraints: an Intel i7 processor with 32GB RAM and an NVIDIA RTX 3060 GPU. We utilized PyTorch as our deep learning framework, Unsloth for optimization and efficient fine-tuning, and the Hugging Face ecosystem for model management and implementation.

The implementation was carried out in Python notebooks, with a dedicated class abstracting the data conversion, testing procedures, and model construction, while respecting the specific configuration requirements and hardware limitations for each model.

### Fine-tuning Process

The fine-tuning process involved adapting these general-purpose language models to the specific task of predicting bus occupancy. By providing examples of queries and their corresponding correct answers, we guided the models to learn the patterns and relationships between contextual variables (day, weather, route information) and resulting occupancy levels.

This approach takes advantage of the models' pre-existing knowledge and pattern recognition capabilities while specializing them for our transportation domain. The text-based formulation allows us to represent complex, multi-dimensional inputs in a form these models can effectively process.

### Experimental Results

Our testing revealed varying performance across the three models. Here we present example predictions for three test cases:

#### Test Case 1

```
<human>: Saturday, no rain and warm, route 1865, scheduled at 00:01 and started at 00:04. The occupancy level at bust stop 1446 is:
```

- **Ground truth**: 0 (empty)
- **mistral-7b**: Predicted "100%" (incorrect)
- **phi-2**: Failed to generate a proper response (incorrect)
- **TinyLlama-1.1B**: Predicted "100% occup" (incorrect)

#### Test Case 2

```
<human>: Saturday, no rain and warm, route 1865, scheduled at 00:01 and started at 00:04. The occupancy level at bust stop 185 is:
```

- **Ground truth**: 1 (full)
- **mistral-7b**: Predicted "100%" (incorrect)
- **phi-2**: Failed to generate a proper response (incorrect)
- **TinyLlama-1.1B**: Predicted "100%" (incorrect)

#### Test Case 3

```
<human>: Saturday, no rain and warm, route 4040, scheduled at 00:50 and started at 00:54. The occupancy level at bust stop 755 is:
```

- **Ground truth**: 2 (overcrowded)
- **mistral-7b**: Predicted "100%" (incorrect)
- **phi-2**: Failed to generate a proper response (incorrect)
- **TinyLlama-1.1B**: Predicted "100% occup" (incorrect)

These preliminary results indicate significant challenges in the fine-tuning process. All three models exhibited difficulty in properly generating the expected numeric responses (0, 1, or 2), instead producing text like "100%" which, while related to occupancy, does not match our defined categorical labels. The microsoft/phi-2 model in particular struggled to generate any coherent response for the given queries.

These findings suggest that additional refinement of the fine-tuning process is necessary. Potential improvements could include:

1. Reformulating the output format to avoid confusion between percentage and categorical values
2. Extending the training dataset with more diverse examples
3. Adjusting hyperparameters such as learning rate and training epochs
4. Exploring prompt engineering techniques to better guide the models' responses

Despite these initial challenges, the approach of using fine-tuned language models for transportation prediction tasks remains promising, as it offers a flexible framework for incorporating diverse contextual factors in natural language format.
