# LangBack: Native Language Identification (NLI) with BERT, LLaMA 3.2, and Qwen 2.5

---

LangBack is a project that identifies a speaker's **native language (L1)** based on their **English (L2) text** using **BERT, LLaMA 3.2, and Qwen 2.5**. This task, known as **Native Language Identification (NLI)**, has applications in **language identification**, **linguistic analysis**, and **language learning systems**.

The project fine-tunes BERT using **full training** while using **prompt-based fine-tuning** for LLaMA and Qwen. The models are evaluated on English learner texts from speakers of **Chinese**, **Arabic**, and **Korean**.

## 🛠️ **Setup**

1. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Scripts**
   ```bash
    python train_bert_finetuning.py
    python train_llama_finetuning.py
    python train_qwen_finetuning.py
   ```

## 📊 **Data Format**

The **train_data.csv** and **test_data.csv** files have the following structure:

| **text**                    | **label** |
| --------------------------- | --------- |
| He very likes to eat pizza. | Chinese   |
| He likes very to eat pizza. | Arabic    |
| He likes pizza very much.   | Korean    |

- **text**: Sentence written by a non-native English speaker (L2).
- **label**: Native language (L1) of the speaker (e.g., **Chinese**, **Arabic**, or **Korean**).

## 🤖 **Models Used**

| **Model**     | **Training Type**    | **Description**                          |
| ------------- | -------------------- | ---------------------------------------- |
| **BERT**      | **Full fine-tuning** | Traditional BERT fine-tuning             |
| **LLaMA 3.2** | **Prompt-based**     | Prompt used to guide LLaMA's predictions |
| **Qwen 2.5**  | **Prompt-based**     | Prompt used to guide Qwen's predictions  |

#### **Prompt for LLaMA and Qwen**

_“You are an expert linguist identifying native language backgrounds of non-native English speakers. Given their written English text, predict their native language.”_

## 📢 **Contact**

For issues or suggestions, please email ph474@cornell.edu.
