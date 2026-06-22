# E-VQA Evidence Check
- Add a new column for data_id , starting from E-VQA_0 in the original csv file fixing/evqa/test_evqa.csv.
- The KB section coould be too lengthy, may use the original paragraph structures(split by '\n')
- Some annotated answer could be extremely short, but considering the context provided in the question, it may also be considered as correct.
- There could be duplicated answers(concated with '|'), which could be removed.

## Ground the QA pair with fine-grained evidence.
1. For templated questions, there could be evidence in provided csv file.
- Check if it is shown in the corresponding section in KB.
- The evidence field in csv may also be properly extended to bridge QA in a more natural way.

2. For automatic questions, there is no direct evidence in provided csv file.
- Go through the whole corresponding section in the KB to locate the evidence senteces.

3. For multi-answer questions(connected with &&)
- If direct evidence exists, follow pipeline for templated questions to examine every candidate answer.
- If not, follow pipeline for automatic questions to find evidence and exmine QA for every candidate answer.
- If there are only partial candidate answers are supported by the evidence, then tag it as improvable(partial correct.)

## Audit annotation quality
1. Consider the context provided in the question, which could be helpful to shorten the answer.
2. Faithfully follow the evidence, the priority of original annotated answer can be lower than the evidence if it is considered supporting.
3. As long as the answer is derived from the supporting evidence, we can consider to make it more precise to the question to avoid vague or coarse-grained but useless answer.