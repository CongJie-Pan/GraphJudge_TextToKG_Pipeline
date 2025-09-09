# Evaluation Metrics

## The problem of Conventional Evaluation

- The Conventional way : Rule-Based. Resemblance between predictions and ground-truth KGs through strict string matching.
- Conventional way limitations : The triple meaning is correct, but the word is different lead to the falut.
- e.g. 
  - `<Albert Einstein, born in, Germany>`
  - `<Einstein, was born in, Germany>`
- Meaning is the same, but word different. 

## Solution : Following PiVe's Approach

Instead of rigid string matching, they want to use more flexible evaluation methods that generated graphs are semantically equivalent to the ground truth. Hence, in the evaluation, it used :
- One semantic level metrics - G-BERTScore(G-BS)
  - Use embeddings or semantic similarity to determine if triples have the same meaning.
- Two soft string matching metrics - G-BLEU(G-BL) , G-ROUGE(G-RO)
  - More flexible than exact matching, allowing for minor variations in wording.

### Run the eval process

- First prepare the pre-data of the eval.py
  - pred.txt
  - gold.txt
  - Using the command to get the two files: `python eval_preDataProcess.py --csv "..\..\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration2\pred_instructions_context_gemini_itr2.csv" --output_dir examples --verbose`

- 